//! Symmetric Mulcs PCS — alternative construction using Laurent polynomial
//! symmetry.
//!
//! Compared with the Claymore-style MulcsPCS, the symmetric alternative:
//! - Reduces hbar length from ~2N to ~N (N-1 coefficients)
//! - Reduces KZG quotient length from ~2N to ~N
//! - Reduces verifier G2 scalar multiplications from 2 to 1
//! - Replaces the Claymore identity with a symmetric Laurent identity
//!
//! The Fiat-Shamir order is critical:
//! 1. z is derived after cm_hbar is absorbed
//! 2. alpha is derived after all 4 claimed evaluations are absorbed
//! 3. The batched KZG proof is computed with alpha-randomized combination

use crate::{
    pcs::{
        multilinear_kzg::batching::BatchProof,
        prelude::{Commitment, PCSError},
        profile, PolynomialCommitmentScheme, StructuredReferenceString,
    },
    poly_iop::{prelude::SumCheck, PolyIOP},
};
use arithmetic::{build_eq_x_r_vec, DenseMultilinearExtension, VPAuxInfo, VirtualPolynomial};
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    scalar_mul::variable_base::VariableBaseMSM,
    AffineRepr, CurveGroup,
};
use ark_ff::Field;
use ark_poly::MultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::Borrow, format, marker::PhantomData, rand::Rng, string::ToString, sync::Arc, vec,
    vec::Vec, One, Zero,
};
use std::{collections::BTreeMap, iter, ops::Deref};
use transcript::IOPTranscript;

use super::{
    srs::{MulcsProverParam, MulcsUniversalParams, MulcsVerifierParam},
    util::UnivarPoly,
};

pub struct MulcsSymmetricPCS<E: Pairing> {
    phantom: PhantomData<E>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct MulcsSymmetricProof<E: Pairing> {
    pub cm_hbar: E::G1Affine,
    pub f_z: E::ScalarField,
    pub f_z_inv: E::ScalarField,
    pub hbar_z: E::ScalarField,
    pub hbar_z_inv: E::ScalarField,
    pub z: E::ScalarField,
    pub pi: E::G1Affine,
    pub mu: usize,
}

const BACKEND: &str = "MulcsSym";

// ═══════════════════════════════════════════════════════════════════
// PolynomialCommitmentScheme trait impl
// ═══════════════════════════════════════════════════════════════════

impl<E: Pairing> PolynomialCommitmentScheme<E> for MulcsSymmetricPCS<E> {
    type ProverParam = MulcsProverParam<E>;
    type VerifierParam = MulcsVerifierParam<E>;
    type SRS = MulcsUniversalParams<E>;
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    type Commitment = Commitment<E>;
    type Proof = MulcsSymmetricProof<E>;
    type BatchProof = BatchProof<E, Self>;

    fn gen_srs_for_testing<R: Rng>(rng: &mut R, s: usize) -> Result<Self::SRS, PCSError> {
        MulcsUniversalParams::<E>::gen_srs_for_testing(rng, s)
    }

    fn trim(
        srs: impl Borrow<Self::SRS>,
        _d: Option<usize>,
        nv: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        let nv = nv.ok_or_else(|| PCSError::InvalidParameters("need num_var".to_string()))?;
        srs.borrow().trim(1 << nv)
    }

    fn commit(
        pp: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let pp = pp.borrow();
        let nv = poly.num_vars;
        let n = 1 << nv;
        if pp.max_degree < n - 1 {
            return Err(PCSError::InvalidParameters(format!(
                "degree {} > max {}",
                n - 1,
                pp.max_degree
            )));
        }
        let _t = profile::ScopedTimer::new(BACKEND, nv, n, "commit_to_evals", n, "to_evaluations");
        let scalars = poly.to_evaluations();
        drop(_t);
        let _t = profile::ScopedTimer::new(BACKEND, nv, n, "commit_msm", scalars.len(), "KZG-MSM");
        let cm = pp.commit(&scalars);
        drop(_t);
        Ok(Commitment(cm))
    }

    fn open(
        pp: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let mut t = IOPTranscript::new(b"mulcs-sym-open");
        t.append_field_element(b"mu", &E::ScalarField::from(poly.num_vars as u64))?;
        symmetric_open_with_transcript(pp.borrow(), poly, point, &mut t)
    }

    fn multi_open(
        pp: impl Borrow<Self::ProverParam>,
        polys: &[Self::Polynomial],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Self::BatchProof, PCSError> {
        symmetric_sumcheck_multi_open(pp.borrow(), polys, points, evals, transcript)
    }

    fn verify(
        vp: &Self::VerifierParam,
        com: &Self::Commitment,
        point: &Self::Point,
        val: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        let mut t = IOPTranscript::new(b"mulcs-sym-open");
        t.append_field_element(b"mu", &E::ScalarField::from(proof.mu as u64))?;
        symmetric_verify_with_transcript(vp, com, point, val, proof, &mut t)
    }

    fn batch_verify(
        vp: &Self::VerifierParam,
        coms: &[Self::Commitment],
        points: &[Self::Point],
        bp: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        symmetric_sumcheck_batch_verify(vp, coms, points, bp, transcript)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Laurent polynomial helpers
// ═══════════════════════════════════════════════════════════════════

/// Compute h(X) = f_v(X) * Q_r(X^{-1}) as Laurent coefficients.
///
/// Q_r(X) = Π_{k=0}^{μ-1} ((1-r_k) + r_k·X^{2^k})
/// Q_r(X^{-1}) = Π_{k=0}^{μ-1} ((1-r_k) + r_k·X^{-2^k})
///
/// The constant term h_0 equals the multilinear evaluation f(r).
///
/// Returns Vec of length 2N-1 where index i + (N-1) stores coefficient of X^i,
/// for i = -(N-1) .. (N-1).
///
/// Uses structured right-shift multiplication: O(N * μ) time.
fn compute_laurent_h<F: Field>(f_v: &[F], mu: usize, r: &[F]) -> Vec<F> {
    let n = 1 << mu;
    let offset = n - 1;
    let len = 2 * n - 1;
    let mut h = vec![F::zero(); len];
    for i in 0..n {
        h[offset + i] = f_v[i];
    }

    let _t = profile::ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "sym_open_compute_laurent_h",
        1,
        "structured-right-shift",
    );

    for k in 0..mu {
        let s = 1 << k;
        let rk = r[k];
        let omrk = F::one() - rk;
        let mut new_h = vec![F::zero(); len];
        for i in 0..len {
            let val = h[i];
            if val.is_zero() {
                continue;
            }
            new_h[i] += omrk * val;
            if i >= s {
                new_h[i - s] += rk * val;
            }
        }
        h = new_h;
    }
    drop(_t);
    h
}

/// Compute hbar from the Laurent coefficients of h.
///
/// L(X) = h(X) + h(X^{-1}) - 2y
/// hbar_i = L_{i+1} for i = 0..(N-2)
///
/// Returns hbar coefficients of length N-1.
fn compute_symmetric_hbar<F: Field>(laurent_h: &[F], offset: usize, _y: F) -> Vec<F> {
    let n = offset + 1;
    let _t = profile::ScopedTimer::new(
        BACKEND,
        n.trailing_zeros() as usize,
        n,
        "sym_open_compute_hbar",
        1,
        "symmetric-hbar",
    );
    let mut hbar = vec![F::zero(); n - 1];
    for i in 1..n {
        let idx_pos = offset + i;
        let idx_neg = offset - i;
        hbar[i - 1] = laurent_h[idx_pos] + laurent_h[idx_neg];
    }
    drop(_t);
    hbar
}

/// Evaluate Q_r(X) = Π_{k=0}^{μ-1} ((1-r_k) + r_k·X^{2^k}) at x.
/// O(μ) field operations.
fn eval_q_r<F: Field>(mu: usize, r: &[F], x: F) -> F {
    let mut result = F::one();
    let mut x_pow = x;
    for k in 0..mu {
        result *= (F::one() - r[k]) + r[k] * x_pow;
        if k + 1 < mu {
            x_pow = x_pow.square();
        }
    }
    result
}

/// Evaluate Q_r(X^{-1}) = Π_{k=0}^{μ-1} ((1-r_k) + r_k·X^{-2^k}) at x.
/// x must be nonzero.
fn eval_q_r_inv<F: Field>(mu: usize, r: &[F], x: F) -> Result<F, PCSError> {
    let x_inv = x
        .inverse()
        .ok_or_else(|| PCSError::InvalidProof("z is zero in eval_q_r_inv".to_string()))?;
    let mut result = F::one();
    let mut x_pow = x_inv;
    for k in 0..mu {
        result *= (F::one() - r[k]) + r[k] * x_pow;
        if k + 1 < mu {
            x_pow = x_pow.square();
        }
    }
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════
// z validation helper (shared by prover and verifier)
// ═══════════════════════════════════════════════════════════════════

/// Validate the Fiat-Shamir challenge z and return its inverse.
/// Rejects z = 0 (no inverse) and z^2 = 1 (divisor (X−z)(X−z⁻¹) degenerates).
fn validate_symmetric_z<F: Field>(z: F) -> Result<F, PCSError> {
    if z.is_zero() {
        return Err(PCSError::InvalidProof(
            "z is zero: two-point divisor degenerates".to_string(),
        ));
    }
    if z * z == F::one() {
        return Err(PCSError::InvalidProof(
            "z^2=1: two-point divisor degenerates (z = z^{-1})".to_string(),
        ));
    }
    let z_inv = z.inverse().ok_or_else(|| {
        PCSError::InvalidProof("z inverse failed — should not happen after guards".to_string())
    })?;
    Ok(z_inv)
}

// ═══════════════════════════════════════════════════════════════════
// Single opening (prover)
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn symmetric_open_with_transcript<E: Pairing>(
    pp: &MulcsProverParam<E>,
    polynomial: &Arc<DenseMultilinearExtension<E::ScalarField>>,
    point: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<(MulcsSymmetricProof<E>, E::ScalarField), PCSError> {
    let mu = polynomial.num_vars();
    let n = 1 << mu;

    let _t_total = profile::ScopedTimer::new(BACKEND, mu, n, "sym_open_total", 1, "total");

    // Convert f_v to evaluations
    let _t_evals =
        profile::ScopedTimer::new(BACKEND, mu, n, "sym_open_to_evals", n, "to_evaluations");
    let coeffs = polynomial.to_evaluations();
    let f_v = UnivarPoly::new(coeffs.clone());
    let y = polynomial
        .evaluate(point)
        .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;
    drop(_t_evals);

    // Compute h(X) = f_v(X) * T_r(X^{-1}) as Laurent polynomial
    let laurent_h = compute_laurent_h(&coeffs, mu, point);
    let offset = n - 1;

    // Compute hbar: L(X) = h(X) + h(X^{-1}) - 2y, hbar from positive L
    let _t_hbar = profile::ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "sym_open_compute_hbar_from_L",
        1,
        "hbar-coeffs",
    );
    let hbar_coeffs = compute_symmetric_hbar(&laurent_h, offset, y);
    drop(_t_hbar);

    let hbar = UnivarPoly::new(hbar_coeffs);

    // Commit hbar
    let _t_cm_hbar = profile::ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "sym_open_commit_hbar",
        hbar.coeffs.len(),
        "KZG-commit-hbar",
    );
    let cm_hbar = pp.commit(&hbar.coeffs);
    drop(_t_cm_hbar);

    transcript.append_serializable_element(b"cm_hbar", &cm_hbar)?;

    // Fiat-Shamir: derive z after cm_hbar
    let _t_z = profile::ScopedTimer::new(BACKEND, mu, n, "sym_open_derive_z", 1, "FS-challenge");
    let z_buf = transcript.get_and_append_challenge_vectors(b"mulcs_sym_z", 1)?;
    let z = z_buf[0];
    drop(_t_z);

    let z_inv = validate_symmetric_z(z).map_err(|e| PCSError::InvalidParameters(e.to_string()))?;

    // Evaluate at z and z^{-1}
    let _t_evals_z =
        profile::ScopedTimer::new(BACKEND, mu, n, "sym_open_eval_at_zs", 4, "Horner-evals");
    let f_z = f_v.evaluate(z);
    let f_z_inv = f_v.evaluate(z_inv);
    let hbar_z = hbar.evaluate(z);
    let hbar_z_inv = hbar.evaluate(z_inv);
    drop(_t_evals_z);

    // Absorb claimed evaluations, then derive alpha
    transcript.append_field_element(b"f_z", &f_z)?;
    transcript.append_field_element(b"f_z_inv", &f_z_inv)?;
    transcript.append_field_element(b"hbar_z", &hbar_z)?;
    transcript.append_field_element(b"hbar_z_inv", &hbar_z_inv)?;

    let alpha_buf = transcript.get_and_append_challenge_vectors(b"mulcs_sym_alpha", 1)?;
    let alpha = alpha_buf[0];

    // Compute remainder polynomials
    let rf = two_point_remainder_sym(z, f_z, z_inv, f_z_inv)?;
    let rh = two_point_remainder_sym(z, hbar_z, z_inv, hbar_z_inv)?;

    // Compute the combined quotient:
    // Z(X) = (X-z)(X-z^{-1}) = X^2 - (z+z^{-1})X + 1
    // q(X) = (f_v(X) + alpha * hbar(X) - r_f(X) - alpha * r_h(X)) / Z(X)
    let _t_quot =
        profile::ScopedTimer::new(BACKEND, mu, n, "sym_open_build_quotient", 1, "poly-div");
    let s = z + z_inv;
    let max_deg = coeffs.len().max(hbar.coeffs.len());
    let q = build_combined_quotient(&coeffs, &hbar.coeffs, &rf, &rh, alpha, s, max_deg);
    drop(_t_quot);

    let _t_pi = profile::ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "sym_open_commit_pi",
        q.len(),
        "KZG-commit-pi",
    );
    let pi = pp.commit(&q);
    drop(_t_pi);

    let proof = MulcsSymmetricProof {
        cm_hbar,
        f_z,
        f_z_inv,
        hbar_z,
        hbar_z_inv,
        z,
        pi,
        mu,
    };
    drop(_t_total);
    Ok((proof, y))
}

// ═══════════════════════════════════════════════════════════════════
// Single verification
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn symmetric_verify_with_transcript<E: Pairing>(
    vp: &MulcsVerifierParam<E>,
    commitment: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &MulcsSymmetricProof<E>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError> {
    let mu = proof.mu;

    // --- input integrity checks (before any shift) ---
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
        .ok_or_else(|| PCSError::InvalidProof(format!("mu {} overflow in shift", mu)))?;
    if point.len() != mu {
        return Err(PCSError::InvalidProof(format!(
            "point length {} != proof.mu {}",
            point.len(),
            mu
        )));
    }
    if vp.max_degree < n - 1 {
        return Err(PCSError::InvalidProof(format!(
            "verifier param max_degree {} insufficient for n={}",
            vp.max_degree, n
        )));
    }

    let _t_total = profile::ScopedTimer::new(BACKEND, mu, n, "sym_verify_total", 1, "total");

    // Replay transcript: cm_hbar
    transcript.append_serializable_element(b"cm_hbar", &proof.cm_hbar)?;

    // Replay: derive z
    let z_buf = transcript.get_and_append_challenge_vectors(b"mulcs_sym_z", 1)?;
    let z = z_buf[0];
    if z != proof.z {
        return Ok(false);
    }

    let z_inv = validate_symmetric_z(z)?;

    // Replay: absorb evaluations, derive alpha
    transcript.append_field_element(b"f_z", &proof.f_z)?;
    transcript.append_field_element(b"f_z_inv", &proof.f_z_inv)?;
    transcript.append_field_element(b"hbar_z", &proof.hbar_z)?;
    transcript.append_field_element(b"hbar_z_inv", &proof.hbar_z_inv)?;

    let alpha_buf = transcript.get_and_append_challenge_vectors(b"mulcs_sym_alpha", 1)?;
    let alpha = alpha_buf[0];

    // --- Reconstruct remainders from claimed evaluations (do NOT trust proof) ---
    let rf = two_point_remainder_sym(z, proof.f_z, z_inv, proof.f_z_inv)?;
    let rh = two_point_remainder_sym(z, proof.hbar_z, z_inv, proof.hbar_z_inv)?;

    // Pairing check: e(cm_f + alpha * cm_hbar - cm_r, g2_one) == e(pi, Z(tau)_g2)
    let _t_pair = profile::ScopedTimer::new(BACKEND, mu, n, "sym_verify_pairing", 1, "1-pairing");

    let cm_f = commitment.0.into_group();
    let cm_hbar = proof.cm_hbar.into_group();

    let r0 = rf[0] + alpha * rh[0];
    let r1 = rf[1] + alpha * rh[1];
    let cm_r = vp.g1_one.into_group() * r0 + vp.g1_x.into_group() * r1;

    let cm_comb = cm_f + cm_hbar * alpha - cm_r;

    // Z(X) = X^2 - (z+z^{-1})X + 1
    // [Z(τ)]_2 = g2_x2 - (z+z^{-1}) * g2_x + g2_one
    let s = z + z_inv;
    let g2_zx = vp.g2_x2.into_group() - vp.g2_x.into_group() * s + vp.g2_one.into_group();

    let neg_pi = (-proof.pi.into_group()).into_affine();
    let ok = E::multi_pairing(
        [cm_comb.into_affine(), neg_pi],
        [vp.g2_one, g2_zx.into_affine()],
    ) == PairingOutput(E::TargetField::one());
    drop(_t_pair);
    if !ok {
        return Ok(false);
    }

    // Symmetric identity check
    let _t_id = profile::ScopedTimer::new(BACKEND, mu, n, "sym_verify_identity", 1, "symmetric-id");
    let t_z_inv = eval_q_r_inv(mu, point, z)?;
    let t_z = eval_q_r(mu, point, z);
    let lhs = z * proof.hbar_z + z_inv * proof.hbar_z_inv + value.double();
    let rhs = proof.f_z * t_z_inv + proof.f_z_inv * t_z;
    let result = Ok(lhs == rhs);
    drop(_t_id);
    drop(_t_total);
    result
}

// ═══════════════════════════════════════════════════════════════════
// Polynomial helpers
// ═══════════════════════════════════════════════════════════════════

/// Compute the remainder polynomial r(X) of degree 1 interpolating
/// (x, y) and (x^{-1}, y_inv).
fn two_point_remainder_sym<F: Field>(x: F, y: F, x_inv: F, y_inv: F) -> Result<[F; 2], PCSError> {
    let denom = x_inv - x;
    let inv = denom
        .inverse()
        .ok_or_else(|| PCSError::InvalidProof("duplicate KZG opening points".to_string()))?;
    let slope = (y_inv - y) * inv;
    let intercept = y - slope * x;
    Ok([intercept, slope])
}

/// Build the combined quotient polynomial:
/// q(X) = (f_v(X) + alpha * hbar(X) - (r_f(X) + alpha * r_h(X))) / Z(X)
fn build_combined_quotient<F: Field>(
    f_coeffs: &[F],
    hbar_coeffs: &[F],
    rf: &[F; 2],
    rh: &[F; 2],
    alpha: F,
    s: F,
    max_deg: usize,
) -> Vec<F> {
    let mut combined = vec![F::zero(); max_deg];
    for i in 0..f_coeffs.len() {
        combined[i] += f_coeffs[i];
    }
    for i in 0..hbar_coeffs.len() {
        combined[i] += alpha * hbar_coeffs[i];
    }

    // Subtract combined remainder
    combined[0] -= rf[0] + alpha * rh[0];
    if combined.len() > 1 {
        combined[1] -= rf[1] + alpha * rh[1];
    }

    // Z(X) = X^2 - s*X + 1, monic leading coefficient = 1
    // Long division: q[i] = combined[i+2] + s*q[i+1] - q[i+2]?
    // Actually: q = combined / Z. Since leading coeff of Z is 1:
    // For deg(combined) = d, deg(q) = d-2
    // q_i = combined_{i+2}... Let me use the existing poly_div which works.
    let z_coeffs = [F::one(), -s, F::one()];
    poly_div_sym(&combined, &z_coeffs)
}

/// Polynomial long division a/b where b is monic.
fn poly_div_sym<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    let db = b.len() - 1;
    let da = a.len();
    if da < db + 1 {
        return vec![];
    }
    let mut q = vec![F::zero(); da - db];
    let mut rem = a.to_vec();
    let inv_b = b[db].inverse().unwrap();
    for i in (db..da).rev() {
        if rem[i].is_zero() {
            continue;
        }
        let c = rem[i] * inv_b;
        q[i - db] = c;
        for j in 0..=db {
            rem[i - db + j] -= c * b[j];
        }
    }
    q
}

// ═══════════════════════════════════════════════════════════════════
// Sumcheck batch open
// ═══════════════════════════════════════════════════════════════════

fn symmetric_sumcheck_multi_open<E: Pairing>(
    pp: &MulcsProverParam<E>,
    polynomials: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    points: &[Vec<E::ScalarField>],
    evals: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<BatchProof<E, MulcsSymmetricPCS<E>>, PCSError> {
    if polynomials.is_empty() {
        return Err(PCSError::InvalidParameters(
            "empty polynomial list".to_string(),
        ));
    }
    if polynomials.len() != points.len() || polynomials.len() != evals.len() {
        return Err(PCSError::InvalidParameters(format!(
            "length mismatch: p={}, pt={}, e={}",
            polynomials.len(),
            points.len(),
            evals.len()
        )));
    }
    let num_var = polynomials[0].num_vars;
    let n = 1 << num_var;
    let k = polynomials.len();
    let _t_total =
        profile::ScopedTimer::new(BACKEND, num_var, n, "sym_multi_open_total", k, "total");

    for poly in polynomials {
        if poly.num_vars != num_var {
            return Err(PCSError::InvalidParameters(format!(
                "inconsistent num_vars: {} vs {}",
                poly.num_vars, num_var
            )));
        }
    }
    for point in points {
        if point.len() != num_var {
            return Err(PCSError::InvalidParameters(format!(
                "point len {} != nv {}",
                point.len(),
                num_var
            )));
        }
    }

    let _t_abs = profile::ScopedTimer::new(
        BACKEND,
        num_var,
        n,
        "sym_multi_open_transcript_absorb",
        k,
        "eval_points+evals",
    );
    for eval_point in points.iter() {
        transcript.append_serializable_element(b"eval_point", eval_point)?;
    }
    for eval in evals.iter() {
        transcript.append_field_element(b"eval", eval)?;
    }
    drop(_t_abs);

    let ell = k.next_power_of_two().ilog2() as usize;
    let _t_eq = profile::ScopedTimer::new(
        BACKEND,
        num_var,
        n,
        "sym_multi_open_build_eq_t",
        k,
        "eq(t;i)",
    );
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;
    let eq_t_i_list = if ell == 0 {
        vec![E::ScalarField::one()]
    } else {
        build_eq_x_r_vec(t.as_ref())?
    };
    drop(_t_eq);

    let _t_groups = profile::ScopedTimer::new(
        BACKEND,
        num_var,
        n,
        "sym_multi_open_group_points",
        k,
        "dedup-points",
    );
    let point_indices = points.iter().fold(BTreeMap::<_, _>::new(), |mut m, pt| {
        let i = m.len();
        m.entry(pt).or_insert(i);
        m
    });
    let deduped_points = BTreeMap::from_iter(point_indices.iter().map(|(pt, idx)| (*idx, *pt)))
        .into_values()
        .collect::<Vec<_>>();
    drop(_t_groups);

    let _t_merge = profile::ScopedTimer::new(
        BACKEND,
        num_var,
        n,
        "sym_multi_open_merge_polys",
        deduped_points.len(),
        "merge-by-point",
    );
    let merged_tilde_gs = polynomials
        .iter()
        .zip(points.iter())
        .zip(eq_t_i_list.iter())
        .fold(
            iter::repeat_with(DenseMultilinearExtension::zero)
                .map(Arc::new)
                .take(point_indices.len())
                .collect::<Vec<_>>(),
            |mut merged, ((poly, point), coeff)| {
                *Arc::make_mut(&mut merged[point_indices[point]]) += (*coeff, poly.deref());
                merged
            },
        );
    drop(_t_merge);

    let _t_tilde = profile::ScopedTimer::new(
        BACKEND,
        num_var,
        n,
        "sym_multi_open_build_tilde_eqs",
        deduped_points.len(),
        "eq(b;zi)",
    );
    let tilde_eqs: Vec<_> = deduped_points
        .iter()
        .map(|point| {
            let e = build_eq_x_r_vec(point).unwrap();
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(num_var, e))
        })
        .collect();
    drop(_t_tilde);

    let _t_sc = profile::ScopedTimer::new(
        BACKEND,
        num_var,
        n,
        "sym_multi_open_sumcheck_prove",
        num_var,
        "sumcheck",
    );
    let mut sum_check_vp = VirtualPolynomial::new(num_var);
    for (g, eq) in merged_tilde_gs.iter().zip(tilde_eqs.into_iter()) {
        sum_check_vp.add_mle_list([g.clone(), eq], E::ScalarField::one())?;
    }
    let sc_proof = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::prove(
        &sum_check_vp,
        transcript,
    ) {
        Ok(p) => p,
        Err(_) => return Err(PCSError::InvalidProver("Sumcheck failed".to_string())),
    };
    drop(_t_sc);

    let a2 = &sc_proof.point[..num_var];

    let _t_g = profile::ScopedTimer::new(
        BACKEND,
        num_var,
        n,
        "sym_multi_open_build_g_prime",
        1,
        "g'=sum",
    );
    let mut g_prime = Arc::new(DenseMultilinearExtension::zero());
    for (g, point) in merged_tilde_gs.iter().zip(deduped_points.iter()) {
        let eq = eq_eval(a2, point)?;
        *Arc::make_mut(&mut g_prime) += (eq, g.deref());
    }
    drop(_t_g);

    let mut open_t = IOPTranscript::new(b"mulcs-sym-gprime-open");
    open_t.append_serializable_element(b"point_a2", &a2.to_vec())?;
    open_t.append_field_element(b"mu", &E::ScalarField::from(num_var as u64))?;
    let (g_prime_proof, _g_prime_eval) =
        symmetric_open_with_transcript(pp, &g_prime, a2, &mut open_t)?;

    drop(_t_total);
    Ok(BatchProof {
        sum_check_proof: sc_proof,
        f_i_eval_at_point_i: evals.to_vec(),
        g_prime_proof,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Sumcheck batch verify
// ═══════════════════════════════════════════════════════════════════

fn symmetric_sumcheck_batch_verify<E: Pairing>(
    vp: &MulcsVerifierParam<E>,
    f_i_commitments: &[Commitment<E>],
    points: &[Vec<E::ScalarField>],
    proof: &BatchProof<E, MulcsSymmetricPCS<E>>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError> {
    if f_i_commitments.is_empty() {
        return Err(PCSError::InvalidProof("empty commitments".to_string()));
    }
    if f_i_commitments.len() != points.len()
        || f_i_commitments.len() != proof.f_i_eval_at_point_i.len()
    {
        return Err(PCSError::InvalidProof("length mismatch".to_string()));
    }
    let k = f_i_commitments.len();
    let num_var = proof.sum_check_proof.point.len();
    let n = 1 << num_var;
    let _t_total = profile::ScopedTimer::new(
        BACKEND,
        num_var,
        n,
        "sym_batch_verify_total",
        k,
        "sumcheck-batch",
    );

    let _t_abs = profile::ScopedTimer::new(
        BACKEND,
        num_var,
        n,
        "sym_batch_verify_transcript_absorb",
        k,
        "pts+evals",
    );
    for eval_point in points.iter() {
        transcript.append_serializable_element(b"eval_point", eval_point)?;
    }
    for eval in proof.f_i_eval_at_point_i.iter() {
        transcript.append_field_element(b"eval", eval)?;
    }
    drop(_t_abs);

    for point in points {
        if point.len() != num_var {
            return Err(PCSError::InvalidProof(format!(
                "point len {} != nv {}",
                point.len(),
                num_var
            )));
        }
    }

    let ell = k.next_power_of_two().ilog2() as usize;
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;
    let a2 = &proof.sum_check_proof.point[..num_var];
    let eq_t_list = if ell == 0 {
        vec![E::ScalarField::one()]
    } else {
        build_eq_x_r_vec(t.as_ref())?
    };

    let _t_gc = profile::ScopedTimer::new(
        BACKEND,
        num_var,
        n,
        "sym_batch_verify_build_g_prime_commit",
        k,
        "MSM-g'-commit",
    );
    let mut scalars = vec![];
    let mut bases = vec![];
    for (i, point) in points.iter().enumerate() {
        scalars.push(eq_eval(a2, point)? * eq_t_list[i]);
        bases.push(f_i_commitments[i].0);
    }
    let g_prime_commit = E::G1::msm_unchecked(&bases, &scalars);
    drop(_t_gc);

    let _t_sc = profile::ScopedTimer::new(
        BACKEND,
        num_var,
        n,
        "sym_batch_verify_sumcheck_verify",
        num_var,
        "sumcheck-verify",
    );
    let mut sum = E::ScalarField::zero();
    for (i, &e) in eq_t_list.iter().enumerate().take(k) {
        sum += e * proof.f_i_eval_at_point_i[i];
    }
    let aux_info = VPAuxInfo {
        max_degree: 2,
        num_variables: num_var,
        phantom: PhantomData,
    };
    let subclaim = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::verify(
        sum,
        &proof.sum_check_proof,
        &aux_info,
        transcript,
    ) {
        Ok(p) => p,
        Err(_) => {
            return Err(PCSError::InvalidProver(
                "Sumcheck verify failed".to_string(),
            ))
        },
    };
    drop(_t_sc);

    let _t_open = profile::ScopedTimer::new(
        BACKEND,
        num_var,
        n,
        "sym_batch_verify_final_open",
        1,
        "final-sym-open",
    );
    let mut verify_t = IOPTranscript::new(b"mulcs-sym-gprime-open");
    verify_t.append_serializable_element(b"point_a2", &a2.to_vec())?;
    verify_t.append_field_element(b"mu", &E::ScalarField::from(num_var as u64))?;
    let res = symmetric_verify_with_transcript(
        vp,
        &Commitment(g_prime_commit.into_affine()),
        a2,
        &subclaim.expected_evaluation,
        &proof.g_prime_proof,
        &mut verify_t,
    )?;
    drop(_t_open);
    drop(_t_total);
    Ok(res)
}

// ═══════════════════════════════════════════════════════════════════
// eq_eval helper
// ═══════════════════════════════════════════════════════════════════

fn eq_eval<F: Field>(x: &[F], y: &[F]) -> Result<F, PCSError> {
    if x.len() != y.len() {
        return Err(PCSError::InvalidParameters("len mismatch".to_string()));
    }
    let mut res = F::one();
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        res *= xi * yi + (F::one() - xi) * (F::one() - yi);
    }
    Ok(res)
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_std::{test_rng, UniformRand};

    type E = Bls12_381;

    fn setup(nv: usize) -> (MulcsProverParam<E>, MulcsVerifierParam<E>) {
        let mut rng = test_rng();
        MulcsSymmetricPCS::<E>::trim(
            &MulcsSymmetricPCS::<E>::gen_srs_for_testing(&mut rng, nv).unwrap(),
            None,
            Some(nv),
        )
        .unwrap()
    }

    fn rpt(nv: usize, rng: &mut impl ark_std::rand::Rng) -> Vec<Fr> {
        (0..nv).map(|_| Fr::rand(rng)).collect()
    }
    fn rpoly(nv: usize, rng: &mut impl ark_std::rand::Rng) -> Arc<DenseMultilinearExtension<Fr>> {
        Arc::new(DenseMultilinearExtension::rand(nv, rng))
    }

    // ── Single open positive ──

    #[test]
    fn test_sym_single_commit_open_verify() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [2usize, 4, 6] {
            let (ck, vk) = setup(nv);
            let poly = rpoly(nv, &mut rng);
            let point = rpt(nv, &mut rng);
            let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
            let (proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
            assert!(MulcsSymmetricPCS::<E>::verify(
                &vk, &com, &point, &value, &proof
            )?);
            let fake_val = Fr::rand(&mut rng);
            if fake_val != value {
                assert!(!MulcsSymmetricPCS::<E>::verify(
                    &vk, &com, &point, &fake_val, &proof
                )?);
            }
        }
        Ok(())
    }

    // ── Wrong claimed value reject ──

    #[test]
    fn test_sym_single_open_rejects_wrong_value() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
        let (proof, _) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        assert!(!MulcsSymmetricPCS::<E>::verify(
            &vk,
            &com,
            &point,
            &Fr::rand(&mut rng),
            &proof
        )?);
        Ok(())
    }

    // ── Wrong point reject ──

    #[test]
    fn test_sym_single_open_rejects_wrong_point() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
        let (proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        let wp = rpt(4, &mut rng);
        if wp != point {
            assert!(!MulcsSymmetricPCS::<E>::verify(
                &vk, &com, &wp, &value, &proof
            )?);
        }
        Ok(())
    }

    // ── Wrong commitment reject ──

    #[test]
    fn test_sym_single_open_rejects_wrong_commitment() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let poly2 = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com2 = MulcsSymmetricPCS::<E>::commit(&ck, &poly2)?;
        let (proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        assert!(!MulcsSymmetricPCS::<E>::verify(
            &vk, &com2, &point, &value, &proof
        )?);
        Ok(())
    }

    // ── Tampered cm_hbar reject ──

    #[test]
    fn test_sym_rejects_tampered_cm_hbar() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
        let (mut proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        proof.cm_hbar = (proof.cm_hbar.into_group() * Fr::from(2u64)).into_affine();
        assert!(!MulcsSymmetricPCS::<E>::verify(
            &vk, &com, &point, &value, &proof
        )?);
        Ok(())
    }

    // ── Tampered f_z reject ──

    #[test]
    fn test_sym_rejects_tampered_f_z() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
        let (mut proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        proof.f_z += Fr::ONE;
        assert!(!MulcsSymmetricPCS::<E>::verify(
            &vk, &com, &point, &value, &proof
        )?);
        Ok(())
    }

    // ── Tampered f_z_inv reject ──

    #[test]
    fn test_sym_rejects_tampered_f_z_inv() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
        let (mut proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        proof.f_z_inv += Fr::ONE;
        assert!(!MulcsSymmetricPCS::<E>::verify(
            &vk, &com, &point, &value, &proof
        )?);
        Ok(())
    }

    // ── Tampered hbar_z reject ──

    #[test]
    fn test_sym_rejects_tampered_hbar_z() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
        let (mut proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        proof.hbar_z += Fr::ONE;
        assert!(!MulcsSymmetricPCS::<E>::verify(
            &vk, &com, &point, &value, &proof
        )?);
        Ok(())
    }

    // ── Tampered hbar_z_inv reject ──

    #[test]
    fn test_sym_rejects_tampered_hbar_z_inv() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
        let (mut proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        proof.hbar_z_inv += Fr::ONE;
        assert!(!MulcsSymmetricPCS::<E>::verify(
            &vk, &com, &point, &value, &proof
        )?);
        Ok(())
    }

    // ── Tampered KZG proof reject ──

    #[test]
    fn test_sym_rejects_tampered_pi() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
        let (mut proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        proof.pi = (proof.pi.into_group() * Fr::from(3u64)).into_affine();
        assert!(!MulcsSymmetricPCS::<E>::verify(
            &vk, &com, &point, &value, &proof
        )?);
        Ok(())
    }

    // ── Tampered z (diverges from transcript) reject ──

    #[test]
    fn test_sym_rejects_tampered_z() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
        let (mut proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        proof.z += Fr::ONE;
        assert!(!MulcsSymmetricPCS::<E>::verify(
            &vk, &com, &point, &value, &proof
        )?);
        Ok(())
    }

    // ── Tampered claimed eval rejects (verifier recomputes remainders) ──
    // The rf/rh fields have been removed from MulcsSymmetricProof. The verifier
    // recomputes both remainders from the claimed f_z/f_z_inv/hbar_z/hbar_z_inv
    // alone. Tampering a claimed evaluation therefore breaks both the pairing
    // check and the identity check. These tests verify that behaviour.

    #[test]
    fn test_sym_rejects_tampered_claimed_f_z_after_remainder_recompute() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
        let (mut proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        proof.f_z += Fr::ONE;
        assert!(!MulcsSymmetricPCS::<E>::verify(
            &vk, &com, &point, &value, &proof
        )?);
        Ok(())
    }

    #[test]
    fn test_sym_rejects_tampered_claimed_f_z_inv_after_remainder_recompute() -> Result<(), PCSError>
    {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
        let (mut proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        proof.f_z_inv += Fr::ONE;
        assert!(!MulcsSymmetricPCS::<E>::verify(
            &vk, &com, &point, &value, &proof
        )?);
        Ok(())
    }

    // ── Wrong point length rejects (no panic) ──

    #[test]
    fn test_sym_verify_rejects_wrong_point_len() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let poly = rpoly(nv, &mut rng);
        let point = rpt(nv, &mut rng);
        let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
        let (proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        let short_pt = rpt(2, &mut rng);
        let r = MulcsSymmetricPCS::<E>::verify(&vk, &com, &short_pt, &value, &proof);
        assert!(r.is_err(), "short point should return Error, not panic");
        let long_pt = rpt(8, &mut rng);
        let r2 = MulcsSymmetricPCS::<E>::verify(&vk, &com, &long_pt, &value, &proof);
        assert!(r2.is_err(), "long point should return Error, not panic");
        Ok(())
    }

    // ── validate_symmetric_z helper: reject z=0, z=1, z=-1 ──

    #[test]
    fn test_validate_symmetric_z_rejects_zero() {
        let r = validate_symmetric_z(Fr::zero());
        assert!(r.is_err(), "z=0 must be rejected");
    }

    #[test]
    fn test_validate_symmetric_z_rejects_one() {
        let r = validate_symmetric_z(Fr::one());
        assert!(r.is_err(), "z=1 must be rejected (z^2=1)");
    }

    #[test]
    fn test_validate_symmetric_z_rejects_neg_one() {
        let r = validate_symmetric_z(-Fr::one());
        assert!(r.is_err(), "z=-1 must be rejected (z^2=1)");
    }

    #[test]
    fn test_validate_symmetric_z_accepts_regular() {
        let mut rng = test_rng();
        let z = Fr::rand(&mut rng);
        let r = validate_symmetric_z(z);
        if z.is_zero() || z.square() == Fr::one() {
            assert!(r.is_err());
        } else {
            let z_inv = r.unwrap();
            assert_eq!(z * z_inv, Fr::one(), "returned value should be inverse");
        }
    }

    // ── Huge mu does not panic ──

    #[test]
    fn test_sym_verify_rejects_huge_mu_without_panic() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 2;
        let (ck, vk) = setup(nv);
        let poly = rpoly(nv, &mut rng);
        let point = rpt(nv, &mut rng);
        let com = MulcsSymmetricPCS::<E>::commit(&ck, &poly)?;
        let (mut proof, value) = MulcsSymmetricPCS::<E>::open(&ck, &poly, &point)?;
        proof.mu = usize::BITS as usize;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            MulcsSymmetricPCS::<E>::verify(&vk, &com, &point, &value, &proof)
        }));
        match r {
            Ok(verdict) => assert!(
                verdict.is_err() || !verdict.unwrap(),
                "huge mu ({}) should fail without panic",
                proof.mu
            ),
            Err(_) => panic!("caught panic on huge mu — should not panic"),
        }
        Ok(())
    }

    // ── Sumcheck batch k=1 ──

    #[test]
    fn test_sym_sumcheck_batch_k1() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let polys: Vec<_> = (0..1).map(|_| rpoly(4, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| rpt(4, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| MulcsSymmetricPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let proof = MulcsSymmetricPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(MulcsSymmetricPCS::<E>::batch_verify(
            &vk, &comms, &points, &proof, &mut tv
        )?);
        Ok(())
    }

    // ── Sumcheck batch distinct points ──

    #[test]
    fn test_sym_sumcheck_batch_distinct() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let polys: Vec<_> = (0..3).map(|_| rpoly(4, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| rpt(4, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| MulcsSymmetricPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let proof = MulcsSymmetricPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(MulcsSymmetricPCS::<E>::batch_verify(
            &vk, &comms, &points, &proof, &mut tv
        )?);
        Ok(())
    }

    // ── Sumcheck batch repeated points ──

    #[test]
    fn test_sym_sumcheck_batch_repeated() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let pt = rpt(4, &mut rng);
        let polys: Vec<_> = (0..3).map(|_| rpoly(4, &mut rng)).collect();
        let pts: Vec<_> = vec![pt.clone(), pt.clone(), pt.clone()];
        let evals: Vec<Fr> = polys
            .iter()
            .zip(pts.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| MulcsSymmetricPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let proof = MulcsSymmetricPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(MulcsSymmetricPCS::<E>::batch_verify(
            &vk, &comms, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    // ── Sumcheck batch non-power-of-2 k ──

    #[test]
    fn test_sym_sumcheck_batch_non_power_of_two() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let k = 5;
        let polys: Vec<_> = (0..k).map(|_| rpoly(4, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| rpt(4, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| MulcsSymmetricPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let proof = MulcsSymmetricPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(MulcsSymmetricPCS::<E>::batch_verify(
            &vk, &comms, &points, &proof, &mut tv
        )?);
        Ok(())
    }

    // ── Batch reject wrong eval ──

    #[test]
    fn test_sym_batch_rejects_wrong_eval() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let polys: Vec<_> = (0..3).map(|_| rpoly(4, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| rpt(4, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| MulcsSymmetricPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut we = evals.clone();
        we[0] += Fr::ONE;
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let p = MulcsSymmetricPCS::<E>::multi_open(&ck, &polys, &points, &we, &mut tp)?;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert_sym_rejects(MulcsSymmetricPCS::<E>::batch_verify(
            &vk, &comms, &points, &p, &mut tv,
        ));
        Ok(())
    }

    // ── Batch reject wrong point ──

    #[test]
    fn test_sym_batch_rejects_wrong_point() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let polys: Vec<_> = (0..3).map(|_| rpoly(4, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| rpt(4, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| MulcsSymmetricPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let p = MulcsSymmetricPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        let mut wp = points.clone();
        wp[0] = rpt(4, &mut rng);
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert_sym_rejects(MulcsSymmetricPCS::<E>::batch_verify(
            &vk, &comms, &wp, &p, &mut tv,
        ));
        Ok(())
    }

    // ── Batch reject wrong commitment ──

    #[test]
    fn test_sym_batch_rejects_wrong_commitment() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let polys: Vec<_> = (0..3).map(|_| rpoly(4, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| rpt(4, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| MulcsSymmetricPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let p = MulcsSymmetricPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        let extra = MulcsSymmetricPCS::<E>::commit(&ck, &rpoly(4, &mut rng))?;
        let mut wc = comms.clone();
        wc[0] = extra;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(!MulcsSymmetricPCS::<E>::batch_verify(
            &vk, &wc, &points, &p, &mut tv
        )?);
        Ok(())
    }

    // ── Batch reject malformed lengths ──

    #[test]
    fn test_sym_batch_rejects_malformed_lengths() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let polys: Vec<_> = (0..3).map(|_| rpoly(4, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| rpt(4, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| MulcsSymmetricPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let mut proof = MulcsSymmetricPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        let r = MulcsSymmetricPCS::<E>::batch_verify(&vk, &comms[..2], &points, &proof, &mut tv);
        assert!(r.is_err() || !r.unwrap());
        proof.f_i_eval_at_point_i.pop();
        let mut tv2 = IOPTranscript::new(b"test");
        tv2.append_field_element(b"init", &Fr::ZERO)?;
        let r2 = MulcsSymmetricPCS::<E>::batch_verify(&vk, &comms, &points, &proof, &mut tv2);
        assert!(r2.is_err() || !r2.unwrap());
        Ok(())
    }

    // ── Multi open reject empty ──

    #[test]
    fn test_sym_multi_open_rejects_empty_input() {
        let (ck, _) = setup(4);
        let mut tp = IOPTranscript::new(b"test");
        assert!(MulcsSymmetricPCS::<E>::multi_open(
            &ck,
            &[] as &[Arc<_>],
            &[] as &[Vec<_>],
            &[] as &[Fr],
            &mut tp
        )
        .is_err());
    }

    // ── Multi open reject mismatched lengths ──

    #[test]
    fn test_sym_multi_open_rejects_mismatched_lengths() {
        let (ck, _) = setup(4);
        let mut rng = test_rng();
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let mut tp = IOPTranscript::new(b"test");
        assert!(MulcsSymmetricPCS::<E>::multi_open(
            &ck,
            &[poly.clone()],
            &[point.clone(), point.clone()],
            &[Fr::one()],
            &mut tp
        )
        .is_err());
        let mut tp = IOPTranscript::new(b"test");
        assert!(MulcsSymmetricPCS::<E>::multi_open(
            &ck,
            &[poly.clone()],
            &[point.clone()],
            &[Fr::one(), Fr::one()],
            &mut tp
        )
        .is_err());
    }

    // ── Multi open reject inconsistent num_vars ──

    #[test]
    fn test_sym_multi_open_rejects_inconsistent_num_vars() {
        let (ck, _) = setup(4);
        let mut rng = test_rng();
        let point = rpt(4, &mut rng);
        let mut tp = IOPTranscript::new(b"test");
        assert!(MulcsSymmetricPCS::<E>::multi_open(
            &ck,
            &[rpoly(4, &mut rng), rpoly(3, &mut rng)],
            &[point.clone(), point.clone()],
            &[Fr::one(), Fr::one()],
            &mut tp
        )
        .is_err());
    }

    // ── Multi open reject wrong point len ──

    #[test]
    fn test_sym_multi_open_rejects_wrong_point_len() {
        let (ck, _) = setup(4);
        let mut rng = test_rng();
        let mut tp = IOPTranscript::new(b"test");
        assert!(MulcsSymmetricPCS::<E>::multi_open(
            &ck,
            &[rpoly(4, &mut rng)],
            &[rpt(3, &mut rng)],
            &[Fr::one()],
            &mut tp
        )
        .is_err());
    }

    // ── hbar length check ──

    #[test]
    fn test_sym_hbar_length() {
        let mut rng = test_rng();
        for nv in [2, 4, 6] {
            let n = 1 << nv;
            let coeffs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let r: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();
            let offset = n - 1;
            let laurent_h = compute_laurent_h(&coeffs, nv, &r);
            assert_eq!(laurent_h.len(), 2 * n - 1);
            let hbar = compute_symmetric_hbar(&laurent_h, offset, Fr::rand(&mut rng));
            assert_eq!(hbar.len(), n - 1, "hbar should have N-1 coefficients");
        }
    }

    fn assert_sym_rejects(r: Result<bool, PCSError>) {
        match r {
            Ok(true) => panic!("expected reject"),
            Ok(false) => {},
            Err(_) => {},
        }
    }

    // ── Direct identity/pairing diagnostics ──

    #[test]
    fn test_sym_debug_diagnostics() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 2;
        let n = 1 << nv;
        let poly = rpoly(nv, &mut rng);
        let point = rpt(nv, &mut rng);

        let coeffs = poly.to_evaluations();
        let y = poly.evaluate(&point).unwrap();

        let offset = n - 1;
        let laurent_h = compute_laurent_h(&coeffs, nv, &point);
        let hbar_coeffs = compute_symmetric_hbar(&laurent_h, offset, y);

        let h_0 = laurent_h[offset];
        assert_eq!(h_0, y, "h_0 should equal y");

        for i in 0..(n - 1) {
            let l_pos = laurent_h[offset + i + 1] + laurent_h[offset - (i + 1)];
            assert_eq!(
                l_pos,
                hbar_coeffs[i],
                "hbar[{}] should equal L_{}",
                i,
                i + 1
            );
        }

        let hbar = UnivarPoly::new(hbar_coeffs);
        let f_v = UnivarPoly::new(coeffs.clone());

        let test_z = Fr::rand(&mut rng);
        let test_z_inv = test_z.inverse().unwrap();

        let fz = f_v.evaluate(test_z);
        let fz_inv = f_v.evaluate(test_z_inv);
        let hbar_z = hbar.evaluate(test_z);
        let hbar_z_inv = hbar.evaluate(test_z_inv);

        let lhs = test_z * hbar_z + test_z_inv * hbar_z_inv + y.double();
        let rhs = fz * eval_q_r_inv(nv, &point, test_z)? + fz_inv * eval_q_r(nv, &point, test_z);

        assert_eq!(
            lhs, rhs,
            "Identity should hold for random z\nlhs={:?}, rhs={:?}",
            lhs, rhs
        );

        Ok(())
    }
}
