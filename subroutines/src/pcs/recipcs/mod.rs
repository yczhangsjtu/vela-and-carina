//! ReciPCS — reciprocal polynomial commitment scheme for multilinear
//! polynomials.
//!
//! ReciPCS encodes the coefficient vector of a mu-variate multilinear
//! polynomial f as a univariate polynomial f_v(X) = sum_i a_i X^i (N = 2^mu
//! coefficients). For an evaluation point r it builds the tensor polynomial
//!   T_r(X) = prod_{k=0}^{mu-1} ((1-r_k) + r_k X^{2^k}),  [X^j] T_r = eq(j; r),
//! so that h(X) = f_v(X) T_r(X^{-1}) has constant Laurent coefficient h_0 =
//! f(r).
//!
//! With claimed value y, the balance polynomial L(X) = h(X) + h(X^{-1}) - 2y is
//! symmetric under X -> X^{-1}, and the half-length reciprocal witness
//!   hbar(X) = sum_{i=0}^{N-2} L_{i+1} X^i    (degree N-2, N-1 coefficients)
//! satisfies  L(X) - L_0 = X hbar(X) + X^{-1} hbar(X^{-1}).  Since char != 2,
//! L_0 = 2(f(r) - y) vanishes iff y = f(r), giving the verifier identity
//!   z hbar(z) + z^{-1} hbar(z^{-1}) + 2y = f_v(z) T_r(z^{-1}) + f_v(z^{-1})
//! T_r(z).
//!
//! The evaluation argument is a single KZG opening of C(X) = f_v(X) + alpha
//! hbar(X) at the two reciprocal points {z, z^{-1}} with vanishing polynomial
//!   Z(X) = (X-z)(X-z^{-1}) = X^2 - (z+z^{-1}) X + 1,
//! verified as  e(cm_f + alpha cm_hbar - [R(tau)]_1, [1]_2) = e(pi, [Z(tau)]_2)
//! with [Z(tau)]_2 = [tau^2]_2 - (z+z^{-1}) [tau]_2 + [1]_2 (one G2 scalar
//! mul).
//!
//! Fiat-Shamir order (critical):
//!   1. z is derived after cm_hbar is absorbed;
//!   2. alpha is derived after all four claimed evaluations are absorbed.
//!
//! Proof: 2 G1 elements (cm_hbar, pi) + 4 field elements (the four
//! evaluations).

use crate::pcs::{
    multilinear_kzg::batching::{batch_verify_internal, multi_open_internal, BatchProof},
    prelude::{Commitment, PCSError},
    PolynomialCommitmentScheme, StructuredReferenceString,
};
use arithmetic::DenseMultilinearExtension;
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    AffineRepr, CurveGroup,
};
use ark_ff::Field;
use ark_poly::MultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::Borrow, format, marker::PhantomData, rand::Rng, string::ToString, sync::Arc, vec,
    vec::Vec, One,
};
use transcript::IOPTranscript;

pub mod srs;

use srs::{ReciProverParam, ReciUniversalParams, ReciVerifierParam};

/// ReciPCS scheme handle.
pub struct ReciPCS<E: Pairing> {
    phantom: PhantomData<E>,
}

/// ReciPCS opening proof: 2 group elements + 4 field elements.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct ReciProof<E: Pairing> {
    /// Commitment to the reciprocal witness hbar.
    pub cm_hbar: E::G1Affine,
    /// Batched KZG opening proof at {z, z^{-1}}.
    pub pi: E::G1Affine,
    /// f_v(z)
    pub f_z: E::ScalarField,
    /// f_v(z^{-1})
    pub f_z_inv: E::ScalarField,
    /// hbar(z)
    pub hbar_z: E::ScalarField,
    /// hbar(z^{-1})
    pub hbar_z_inv: E::ScalarField,
    /// number of variables mu (bound into the transcript, checked by verifier)
    pub mu: usize,
}

const LABEL_MU: &[u8] = b"recipcs::mu";
const LABEL_CM_F: &[u8] = b"recipcs::cm_f";
const LABEL_POINT: &[u8] = b"recipcs::point";
const LABEL_VALUE: &[u8] = b"recipcs::value";
const LABEL_CM_HBAR: &[u8] = b"recipcs::cm_hbar";
const LABEL_Z: &[u8] = b"recipcs::z";
const LABEL_FZ: &[u8] = b"recipcs::f_z";
const LABEL_FZI: &[u8] = b"recipcs::f_z_inv";
const LABEL_HZ: &[u8] = b"recipcs::hbar_z";
const LABEL_HZI: &[u8] = b"recipcs::hbar_z_inv";
const LABEL_ALPHA: &[u8] = b"recipcs::alpha";

// ════════════════════════════════════════════════════════════════════
// PolynomialCommitmentScheme trait
// ════════════════════════════════════════════════════════════════════

impl<E: Pairing> PolynomialCommitmentScheme<E> for ReciPCS<E> {
    type ProverParam = ReciProverParam<E>;
    type VerifierParam = ReciVerifierParam<E>;
    type SRS = ReciUniversalParams<E>;
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    type Commitment = Commitment<E>;
    type Proof = ReciProof<E>;
    type BatchProof = BatchProof<E, Self>;

    fn gen_srs_for_testing<R: Rng>(rng: &mut R, nv: usize) -> Result<Self::SRS, PCSError> {
        ReciUniversalParams::<E>::gen_srs_for_testing(rng, nv)
    }

    fn trim(
        srs: impl Borrow<Self::SRS>,
        _supported_degree: Option<usize>,
        supported_num_vars: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        let nv = supported_num_vars
            .ok_or_else(|| PCSError::InvalidParameters("need num_var".to_string()))?;
        let n = 1usize << nv;
        srs.borrow().trim(n - 1)
    }

    fn commit(
        pp: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let pp = pp.borrow();
        let nv = poly.num_vars;
        if nv >= usize::BITS as usize - 1 {
            return Err(PCSError::InvalidParameters(format!(
                "num_vars {} too large for platform word size",
                nv
            )));
        }
        let n = 1usize << nv;
        if pp.max_degree < n - 1 {
            return Err(PCSError::InvalidParameters(format!(
                "SRS max degree {} insufficient for N-1 = {}",
                pp.max_degree,
                n - 1
            )));
        }
        let coeffs = poly.to_evaluations();
        Ok(Commitment(pp.commit(&coeffs)?))
    }

    fn open(
        pp: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let pp = pp.borrow();
        if point.len() != poly.num_vars() {
            return Err(PCSError::InvalidParameters(format!(
                "point length {} != mu {}",
                point.len(),
                poly.num_vars()
            )));
        }
        let value = poly
            .evaluate(point)
            .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;
        let cm_f = ReciPCS::<E>::commit(pp, poly)?;
        let mut t = new_transcript::<E>(poly.num_vars(), &cm_f, point, &value)?;
        recipcs_open(pp, poly, point, &value, &mut t)
    }

    fn verify(
        vp: &Self::VerifierParam,
        com: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        let mut t = new_transcript::<E>(proof.mu, com, point, value)?;
        recipcs_verify(vp, com, point, value, proof, &mut t)
    }

    fn multi_open(
        prover_param: impl Borrow<Self::ProverParam>,
        polynomials: &[Self::Polynomial],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Self::BatchProof, PCSError> {
        multi_open_internal::<E, Self>(
            prover_param.borrow(),
            polynomials,
            points,
            evals,
            transcript,
        )
    }

    fn batch_verify(
        verifier_param: &Self::VerifierParam,
        commitments: &[Self::Commitment],
        points: &[Self::Point],
        batch_proof: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        batch_verify_internal::<E, Self>(
            verifier_param,
            commitments,
            points,
            batch_proof,
            transcript,
        )
    }
}

// ════════════════════════════════════════════════════════════════════
// Transcript setup (binds the full statement)
// ════════════════════════════════════════════════════════════════════

fn new_transcript<E: Pairing>(
    mu: usize,
    cm_f: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
) -> Result<IOPTranscript<E::ScalarField>, PCSError> {
    let mut t = IOPTranscript::new(b"recipcs-v1");
    t.append_field_element(LABEL_MU, &E::ScalarField::from(mu as u64))?;
    t.append_serializable_element(LABEL_CM_F, &cm_f.0)?;
    t.append_serializable_element(LABEL_POINT, &point.to_vec())?;
    t.append_field_element(LABEL_VALUE, value)?;
    Ok(t)
}

// ════════════════════════════════════════════════════════════════════
// Laurent / witness helpers
// ════════════════════════════════════════════════════════════════════

/// Compute h(X) = f_v(X) * T_r(X^{-1}) as a Laurent coefficient buffer of
/// length 2N-1 (index i + (N-1) holds the coefficient of X^i, i in
/// -(N-1)..(N-1)), by mu structured right-shifts. O(N*mu) field operations,
/// FFT-free.
///
/// Thin wrapper over the shared [`crate::pcs::laurent`] kernel so the
/// reciprocal formula lives in exactly one place (also used by the Mercury
/// backend).
fn compute_laurent_h<F: Field>(coeffs: &[F], mu: usize, r: &[F]) -> Vec<F> {
    crate::pcs::laurent::mul_by_reciprocal_tensor(coeffs, mu, r)
}

/// hbar_i = L_{i+1} = h_{i+1} + h_{-(i+1)}, i = 0..N-2. Length N-1, degree N-2.
fn compute_hbar<F: Field>(laurent_h: &[F], offset: usize) -> Vec<F> {
    let n = offset + 1;
    let mut hbar = vec![F::zero(); n - 1];
    for (i, hb) in hbar.iter_mut().enumerate() {
        *hb = laurent_h[offset + (i + 1)] + laurent_h[offset - (i + 1)];
    }
    hbar
}

/// T_r(x) = prod_k ((1-r_k) + r_k x^{2^k}). O(mu) field ops.
fn eval_tensor<F: Field>(mu: usize, r: &[F], x: F) -> F {
    let mut result = F::one();
    let mut x_pow = x;
    for (k, &rk) in r.iter().enumerate().take(mu) {
        result *= (F::one() - rk) + rk * x_pow;
        if k + 1 < mu {
            x_pow = x_pow.square();
        }
    }
    result
}

/// T_r(x^{-1}); x must be nonzero.
fn eval_tensor_inv<F: Field>(mu: usize, r: &[F], x: F) -> Result<F, PCSError> {
    let x_inv = x
        .inverse()
        .ok_or_else(|| PCSError::InvalidProof("z is zero in eval_tensor_inv".to_string()))?;
    Ok(eval_tensor(mu, r, x_inv))
}

/// Reject z = 0 and z^2 = 1 (so that z, z^{-1} are distinct nonzero points).
fn validate_z<F: Field>(z: F) -> Result<F, PCSError> {
    if z.is_zero() {
        return Err(PCSError::InvalidProof(
            "z = 0: two-point divisor degenerates".to_string(),
        ));
    }
    if z * z == F::one() {
        return Err(PCSError::InvalidProof(
            "z^2 = 1: two-point divisor degenerates (z = z^{-1})".to_string(),
        ));
    }
    z.inverse()
        .ok_or_else(|| PCSError::InvalidProof("z inverse failed after guards".to_string()))
}

/// Univariate Horner evaluation.
fn horner<F: Field>(coeffs: &[F], x: F) -> F {
    let mut acc = F::zero();
    for c in coeffs.iter().rev() {
        acc = acc * x + *c;
    }
    acc
}

/// Degree-1 interpolant [b0, b1] through (x, y) and (x_inv, y_inv).
fn two_point_remainder<F: Field>(x: F, y: F, x_inv: F, y_inv: F) -> Result<[F; 2], PCSError> {
    let denom = x_inv - x;
    let inv = denom
        .inverse()
        .ok_or_else(|| PCSError::InvalidProof("duplicate KZG opening points".to_string()))?;
    let b1 = (y_inv - y) * inv;
    let b0 = y - b1 * x;
    Ok([b0, b1])
}

/// q(X) = (C(X) - R(X)) / Z(X) where Z(X) = X^2 - s X + 1 (monic).
fn build_quotient<F: Field>(
    f_coeffs: &[F],
    hbar_coeffs: &[F],
    rf: &[F; 2],
    rh: &[F; 2],
    alpha: F,
    s: F,
) -> Vec<F> {
    let max_deg = f_coeffs.len().max(hbar_coeffs.len());
    let mut combined = vec![F::zero(); max_deg];
    for (i, &c) in f_coeffs.iter().enumerate() {
        combined[i] += c;
    }
    for (i, &c) in hbar_coeffs.iter().enumerate() {
        combined[i] += alpha * c;
    }
    combined[0] -= rf[0] + alpha * rh[0];
    if combined.len() > 1 {
        combined[1] -= rf[1] + alpha * rh[1];
    }
    // divide by monic Z(X) = X^2 - s X + 1
    let z = [F::one(), -s, F::one()];
    monic_div(&combined, &z)
}

/// Polynomial long division a / b, b monic.
fn monic_div<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    let db = b.len() - 1;
    let da = a.len();
    if da < db + 1 {
        return vec![];
    }
    let mut q = vec![F::zero(); da - db];
    let mut rem = a.to_vec();
    for i in (db..da).rev() {
        if rem[i].is_zero() {
            continue;
        }
        let c = rem[i]; // b[db] = 1
        q[i - db] = c;
        for (j, &bj) in b.iter().enumerate() {
            rem[i - db + j] -= c * bj;
        }
    }
    q
}

// ════════════════════════════════════════════════════════════════════
// Prover
// ════════════════════════════════════════════════════════════════════

fn recipcs_open<E: Pairing>(
    pp: &ReciProverParam<E>,
    poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<(ReciProof<E>, E::ScalarField), PCSError> {
    let mu = poly.num_vars();
    if mu == 0 {
        return Err(PCSError::InvalidParameters(
            "mu = 0 unsupported".to_string(),
        ));
    }
    if point.len() != mu {
        return Err(PCSError::InvalidParameters(format!(
            "point length {} != mu {}",
            point.len(),
            mu
        )));
    }
    let n = 1usize << mu;
    let coeffs = poly.to_evaluations();

    let laurent_h = compute_laurent_h(&coeffs, mu, point);
    let offset = n - 1;
    let y = laurent_h[offset]; // = f(r) by the constant-coefficient lemma
                               // Defence in depth: the structured constant coefficient must equal the value
                               // bound into the transcript. These are provably equal for correct inputs;
                               // a mismatch (e.g. a caller passing an inconsistent `value`) is rejected
                               // rather than silently producing an unverifiable proof.
    if &y != value {
        return Err(PCSError::InvalidParameters(
            "claimed value inconsistent with committed polynomial".to_string(),
        ));
    }

    let hbar_coeffs = compute_hbar(&laurent_h, offset);
    let cm_hbar = pp.commit(&hbar_coeffs)?;

    transcript.append_serializable_element(LABEL_CM_HBAR, &cm_hbar)?;
    let z = transcript.get_and_append_challenge_vectors(LABEL_Z, 1)?[0];
    let z_inv = validate_z(z)?;

    let f_z = horner(&coeffs, z);
    let f_z_inv = horner(&coeffs, z_inv);
    let hbar_z = horner(&hbar_coeffs, z);
    let hbar_z_inv = horner(&hbar_coeffs, z_inv);

    transcript.append_field_element(LABEL_FZ, &f_z)?;
    transcript.append_field_element(LABEL_FZI, &f_z_inv)?;
    transcript.append_field_element(LABEL_HZ, &hbar_z)?;
    transcript.append_field_element(LABEL_HZI, &hbar_z_inv)?;
    let alpha = transcript.get_and_append_challenge_vectors(LABEL_ALPHA, 1)?[0];

    let rf = two_point_remainder(z, f_z, z_inv, f_z_inv)?;
    let rh = two_point_remainder(z, hbar_z, z_inv, hbar_z_inv)?;
    let s = z + z_inv;
    let q = build_quotient(&coeffs, &hbar_coeffs, &rf, &rh, alpha, s);
    let pi = pp.commit(&q)?;

    Ok((
        ReciProof {
            cm_hbar,
            pi,
            f_z,
            f_z_inv,
            hbar_z,
            hbar_z_inv,
            mu,
        },
        y,
    ))
}

// ════════════════════════════════════════════════════════════════════
// Verifier
// ════════════════════════════════════════════════════════════════════

fn recipcs_verify<E: Pairing>(
    vp: &ReciVerifierParam<E>,
    com: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &ReciProof<E>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError> {
    let mu = proof.mu;
    // ── input integrity (before any shift) ──
    if mu == 0 {
        return Err(PCSError::InvalidProof("mu is zero".to_string()));
    }
    if mu >= usize::BITS as usize {
        return Err(PCSError::InvalidProof(format!(
            "mu {} exceeds platform word size",
            mu
        )));
    }
    let n = 1usize
        .checked_shl(mu as u32)
        .ok_or_else(|| PCSError::InvalidProof(format!("mu {} overflow", mu)))?;
    if point.len() != mu {
        return Err(PCSError::InvalidProof(format!(
            "point length {} != proof.mu {}",
            point.len(),
            mu
        )));
    }
    if vp.max_degree < n - 1 {
        return Err(PCSError::InvalidProof(format!(
            "verifier max_degree {} insufficient for N-1 = {}",
            vp.max_degree,
            n - 1
        )));
    }

    // Replay Fiat-Shamir.
    transcript.append_serializable_element(LABEL_CM_HBAR, &proof.cm_hbar)?;
    let z = transcript.get_and_append_challenge_vectors(LABEL_Z, 1)?[0];
    let z_inv = validate_z(z)?;

    transcript.append_field_element(LABEL_FZ, &proof.f_z)?;
    transcript.append_field_element(LABEL_FZI, &proof.f_z_inv)?;
    transcript.append_field_element(LABEL_HZ, &proof.hbar_z)?;
    transcript.append_field_element(LABEL_HZI, &proof.hbar_z_inv)?;
    let alpha = transcript.get_and_append_challenge_vectors(LABEL_ALPHA, 1)?[0];

    // Reconstruct remainders from claimed evaluations only.
    let rf = two_point_remainder(z, proof.f_z, z_inv, proof.f_z_inv)?;
    let rh = two_point_remainder(z, proof.hbar_z, z_inv, proof.hbar_z_inv)?;

    // cm_comb = cm_f + alpha cm_hbar - [R(tau)]_1
    let r0 = rf[0] + alpha * rh[0];
    let r1 = rf[1] + alpha * rh[1];
    let cm_r = vp.g1_one.into_group() * r0 + vp.g1_tau.into_group() * r1;
    let cm_comb = com.0.into_group() + proof.cm_hbar.into_group() * alpha - cm_r;

    // [Z(tau)]_2 = [tau^2]_2 - (z+z^{-1}) [tau]_2 + [1]_2 (one G2 scalar mul)
    let s = z + z_inv;
    let g2_z = vp.g2_tau2.into_group() - vp.g2_tau.into_group() * s + vp.g2_one.into_group();

    let neg_pi = (-proof.pi.into_group()).into_affine();
    let pairing_ok = E::multi_pairing(
        [cm_comb.into_affine(), neg_pi],
        [vp.g2_one, g2_z.into_affine()],
    ) == PairingOutput(E::TargetField::one());
    if !pairing_ok {
        return Ok(false);
    }

    // Symmetric evaluation identity.
    let t_z_inv = eval_tensor_inv(mu, point, z)?;
    let t_z = eval_tensor(mu, point, z);
    let lhs = z * proof.hbar_z + z_inv * proof.hbar_z_inv + value.double();
    let rhs = proof.f_z * t_z_inv + proof.f_z_inv * t_z;
    Ok(lhs == rhs)
}

#[cfg(test)]
mod tests;
