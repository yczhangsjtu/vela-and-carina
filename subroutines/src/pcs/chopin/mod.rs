//! CHOPIN — optimal pairing-based multilinear PCS from bivariate KZG.
//!
//! Implements the *optimized* instantiation of Chopin (Figure 5) together with
//! the BDFG20 batch protocol (Figure 6). Two separate pairing-product checks
//! (3-term bivariate + 2-term BDFG20) are never merged into one.
//!
//! Proof = 7 G1 + 7 scalars + `mu` metadata (560 + 4 bytes on BLS12-381).
//!
//! See `docs/hyperplonk_chopin_design.md` for the full mapping to the paper.

use crate::pcs::{
    bdfg::{
        bdfg_first_round, bdfg_second_round, bdfg_verifier_combination,
        poly_eval, BdfgClaim,
        BdfgVerifierCombination,
    },
    laurent::{laurent_offset, mul_by_reciprocal_tensor},
    multilinear_kzg::batching::{batch_verify_internal, multi_open_internal, BatchProof},
    prelude::{Commitment, PCSError},
    profile::ScopedTimer,
    PolynomialCommitmentScheme, StructuredReferenceString,
};
use arithmetic::DenseMultilinearExtension;
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    scalar_mul::variable_base::VariableBaseMSM,
    AffineRepr, CurveGroup,
};
use ark_ff::{Field, PrimeField};
use ark_poly::MultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::Borrow, format, marker::PhantomData, rand::Rng, string::ToString, sync::Arc, vec,
    vec::Vec, One, Zero,
};
use transcript::IOPTranscript;

pub mod srs;

use srs::{
    split_exponents, ChopinProverParam, ChopinUniversalParams, ChopinVerifierParam,
};

const BACKEND: &str = "Chopin";
const DOMAIN: &[u8] = b"chopin-mlpcs-v1";
const MAX_CHALLENGE_RETRY: usize = 64;

// Transcript labels.
const L_VERSION: &[u8] = b"ver";
const L_MU: &[u8] = b"mu";
const L_ML: &[u8] = b"ml";
const L_MR: &[u8] = b"mr";
const L_CF: &[u8] = b"cf";
const L_POINT: &[u8] = b"pt";
const L_ETA: &[u8] = b"eta";
const L_C0: &[u8] = b"c0";
const L_ALPHA: &[u8] = b"alpha";
const L_C1: &[u8] = b"c1";
const L_A: &[u8] = b"a";
const L_GAMMA: &[u8] = b"gamma";
const L_CS: &[u8] = b"cs";
const L_BETA: &[u8] = b"beta";
const L_A1: &[u8] = b"a1";
const L_A2: &[u8] = b"a2";
const L_B1: &[u8] = b"b1";
const L_B2: &[u8] = b"b2";
const L_S1: &[u8] = b"s1";
const L_S2: &[u8] = b"s2";
const L_PI_X: &[u8] = b"px";
const L_PI_Y: &[u8] = b"py";
const L_RHO: &[u8] = b"rho";
const L_W: &[u8] = b"w";
const L_Z_BDFG: &[u8] = b"z";
const L_WP: &[u8] = b"wp";

/// CHOPIN scheme handle.
pub struct ChopinPCS<E: Pairing> {
    phantom: PhantomData<E>,
}

/// CHOPIN opening proof: 7 G1 + 7 scalars + `mu` metadata.
/// Cryptographic payload = 7·48 + 7·32 = 560 bytes (BLS12-381).
/// Canonical serialized including `mu: u32` = 564 bytes.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct ChopinProof<E: Pairing> {
    pub comm_f_zr: E::G1Affine,
    pub comm_f_alpha: E::G1Affine,
    pub comm_s: E::G1Affine,
    pub pi_biv_x: E::G1Affine,
    pub pi_biv_y: E::G1Affine,
    pub batch_w: E::G1Affine,
    pub batch_w_prime: E::G1Affine,
    pub a: E::ScalarField,
    pub a1: E::ScalarField,
    pub a2: E::ScalarField,
    pub b1: E::ScalarField,
    pub b2: E::ScalarField,
    pub s1: E::ScalarField,
    pub s2: E::ScalarField,
    pub mu: u32,
}

impl<E: Pairing> ChopinProof<E> {
    /// Cryptographic payload size in bytes.
    pub fn cryptographic_payload_bytes(&self) -> usize {
        7 * self.comm_f_zr.compressed_size() + 7 * self.a.compressed_size()
    }
}

// ════════════════════════════════════════════════════════════════════
// PolynomialCommitmentScheme trait
// ════════════════════════════════════════════════════════════════════

impl<E: Pairing> PolynomialCommitmentScheme<E> for ChopinPCS<E> {
    type ProverParam = ChopinProverParam<E>;
    type VerifierParam = ChopinVerifierParam<E>;
    type SRS = ChopinUniversalParams<E>;
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    type Commitment = Commitment<E>;
    type Proof = ChopinProof<E>;
    type BatchProof = BatchProof<E, Self>;

    fn gen_srs_for_testing<R: Rng>(rng: &mut R, nv: usize) -> Result<Self::SRS, PCSError> {
        ChopinUniversalParams::<E>::gen_srs_for_testing(rng, nv)
    }

    fn trim(
        srs: impl Borrow<Self::SRS>,
        _supported_degree: Option<usize>,
        supported_num_vars: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        let nv = supported_num_vars
            .ok_or_else(|| PCSError::InvalidParameters("need num_var".to_string()))?;
        srs.borrow().trim(nv)
    }

    fn commit(
        pp: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let pp = pp.borrow();
        check_mu(poly.num_vars)?;
        if pp.num_vars < poly.num_vars {
            return Err(PCSError::InvalidParameters(format!(
                "prover param for {} vars < poly {} vars",
                pp.num_vars, poly.num_vars
            )));
        }
        // If the key supports more variables than the polynomial, pad
        // evaluations with zeros to the key's N.
        let poly_n = 1usize << poly.num_vars;
        let pp_n = pp.n();
        let mut evals = poly.to_evaluations();
        if pp.num_vars > poly.num_vars {
            evals.resize(pp_n, E::ScalarField::zero());
        }
        let _t_total = ScopedTimer::new(BACKEND, poly.num_vars, poly_n, "chopin_commit_total", 1, "total");
        let _t_reorder =
            ScopedTimer::new(BACKEND, poly.num_vars, poly_n, "chopin_commit_reorder", poly_n, "reorder");
        let cm = pp.msm_full_reordered(&evals)?;
        drop(_t_reorder);
        let _t_msm = ScopedTimer::new(BACKEND, poly.num_vars, poly_n, "chopin_commit_msm", poly_n, "N-MSM");
        drop(_t_msm);
        Ok(Commitment(cm))
    }

    fn open(
        pp: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let pp = pp.borrow();
        let mu = poly.num_vars();
        let n = 1usize.checked_shl(mu as u32).unwrap_or(0);
        check_mu(mu)?;
        if point.len() != mu {
            return Err(PCSError::InvalidParameters(format!(
                "point length {} != mu {}",
                point.len(),
                mu
            )));
        }
        // trait `open` has no commitment argument; re-commit C_f (an extra
        // N-MSM). Callers holding C_f should use `open_with_commitment`.
        let _t_recommit = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "chopin_open_statement_recommit",
            n,
            "recommit-cf",
        );
        let value = poly
            .evaluate(point)
            .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;
        let cm_f = ChopinPCS::<E>::commit(pp, poly)?;
        drop(_t_recommit);
        let mut t = new_transcript::<E>(pp.num_vars, pp.m_left, pp.m_right, &cm_f, point, &value)?;
        let proof = chopin_core_open(pp, &poly.evaluations, mu, point, &value, &mut t)?;
        Ok((proof, value))
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

    fn verify(
        vp: &Self::VerifierParam,
        com: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        chopin_core_verify(vp, com, point, value, proof)
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

impl<E: Pairing> ChopinPCS<E> {
    /// Open a polynomial at a point when the caller already holds `C_f`,
    /// avoiding the extra N-MSM recommit that the trait `open` performs.
    pub fn open_with_commitment(
        pp: &ChopinProverParam<E>,
        poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
        point: &[E::ScalarField],
        commitment: &Commitment<E>,
    ) -> Result<(ChopinProof<E>, E::ScalarField), PCSError> {
        let mu = poly.num_vars();
        check_mu(mu)?;
        let value = poly
            .evaluate(point)
            .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;
        if point.len() != mu {
            return Err(PCSError::InvalidParameters(format!(
                "point length {} != mu {}",
                point.len(),
                mu
            )));
        }
        let mut t = new_transcript::<E>(pp.num_vars, pp.m_left, pp.m_right, commitment, point, &value)?;
        let proof = chopin_core_open(pp, &poly.evaluations, mu, point, &value, &mut t)?;
        Ok((proof, value))
    }
}

// ════════════════════════════════════════════════════════════════════
// Dimensions / input validation
// ════════════════════════════════════════════════════════════════════

#[inline]
fn check_mu(mu: usize) -> Result<(), PCSError> {
    if mu < 2 {
        return Err(PCSError::InvalidParameters(format!(
            "chopin requires mu >= 2, got {mu}"
        )));
    }
    if mu >= usize::BITS as usize {
        return Err(PCSError::InvalidParameters(format!(
            "mu {mu} exceeds platform word size"
        )));
    }
    Ok(())
}

fn check_mu_verify(mu: usize) -> Result<(), PCSError> {
    check_mu(mu).map_err(|e| PCSError::InvalidProof(e.to_string()))
}

// ════════════════════════════════════════════════════════════════════
// Coefficient-level helpers
// ════════════════════════════════════════════════════════════════════

/// Build the eq vector `eq(i, u)` of length `b` (little-endian). Requires
/// `b >= 2^{u.len()}`. Works for empty `u` (returns `[1]`).
fn build_eq_vec<F: Field>(u: &[F], b: usize) -> Vec<F> {
    if u.is_empty() {
        let mut v = vec![F::zero(); b];
        v[0] = F::one();
        return v;
    }
    let mut eq = vec![F::zero(); b];
    eq[0] = F::one();
    for (k, &uk) in u.iter().enumerate() {
        let half = 1usize << k;
        for i in (0..half).rev() {
            let v = eq[i];
            let hi = v * uk;
            eq[i + half] = hi;
            eq[i] = v - hi;
        }
    }
    eq
}

/// Determine if eq points are at valid indices for parallel partitioning in the numeric domain
fn compute_restriction<F: Field>(
    evals: &[F],
    psi_r: &[F],
    big_ml: usize,
    big_mr: usize,
) -> Vec<F> {
    let mut g = vec![F::zero(); big_ml];
    for j in 0..big_mr {
        let pj = psi_r[j];
        if pj.is_zero() {
            continue;
        }
        let base = big_ml * j;
        for i in 0..big_ml {
            g[i] += evals[base + i] * pj;
        }
    }
    g
}

/// Synthetic division in X: for each column `j`, divide
/// `f_j(X) = sum_i F[i,j] X^i` by `(X - alpha)`, yielding
/// `q1_j[i]` for `i < M_L-1` and `f_alpha[j] = f_j(alpha)`.
/// Returns `(q1, f_alpha)` where `q1` is in `j*(M_L-1)+i` order.
fn divide_x_at_alpha<F: Field>(
    evals: &[F],
    big_ml: usize,
    big_mr: usize,
    alpha: F,
) -> (Vec<F>, Vec<F>) {
    let q1_len = (big_ml - 1) * big_mr;
    let mut q1 = vec![F::zero(); q1_len];
    let mut f_alpha = vec![F::zero(); big_mr];
    for j in 0..big_mr {
        let col = &evals[big_ml * j..big_ml * (j + 1)];
        // synthetic division of col by (X-alpha)
        let mut carry = F::zero();
        for i in (0..big_ml - 1).rev() {
            let c = col[i + 1] + alpha * carry;
            let pos = j * (big_ml - 1) + i;
            q1[pos] = c;
            carry = c;
        }
        f_alpha[j] = col[0] + alpha * carry;
    }
    (q1, f_alpha)
}

/// Divide the univariate `f_alpha(Y)` (degree < M_R) by `(Y - beta)`, returning
/// `(q2, remainder = f_alpha(beta) = b1)` with `q2` of length `M_R-1`.
fn divide_y_at_beta<F: Field>(
    f_alpha: &[F],
    beta: F,
) -> Result<(Vec<F>, F), PCSError> {
    if f_alpha.is_empty() {
        return Err(PCSError::InvalidProver("empty f_alpha".to_string()));
    }
    if f_alpha.len() == 1 {
        return Ok((Vec::new(), f_alpha[0]));
    }
    let mut q2 = vec![F::zero(); f_alpha.len() - 1];
    let mut carry = F::zero();
    for i in (0..f_alpha.len() - 1).rev() {
        let c = f_alpha[i + 1] + beta * carry;
        q2[i] = c;
        carry = c;
    }
    let b1 = f_alpha[0] + beta * carry;
    Ok((q2, b1))
}

/// Symmetric Lagrange witness: given `coeffs` of length `M = 2^m` and the tensor
/// point `u` (length `m`), returns `S` of length `M-1` satisfying
/// `coeffs(X)·ψ_u(1/X) + coeffs(1/X)·ψ_u(X) = 2<coeffs,ψ_u> + X·S(X) + X^{-1}·S(X^{-1})`.
/// Uses the shared Laurent kernel (`mul_by_reciprocal_tensor`).
fn symmetric_lagrange_witness<F: Field>(
    coeffs: &[F],
    u: &[F],
    m: usize,
) -> Vec<F> {
    let big_m = 1usize << m;
    debug_assert_eq!(coeffs.len(), big_m, "coeffs length mismatch");
    let buf = mul_by_reciprocal_tensor(coeffs, m, u);
    let offset = laurent_offset(m);
    let mut s = vec![F::zero(); big_m - 1];
    for (i, si) in s.iter_mut().enumerate() {
        *si = buf[offset + (i + 1)] + buf[offset - (i + 1)];
    }
    s
}

/// Tensor polynomial `prod_k ((1-u_k) + u_k x^{2^k})` evaluated at `x`.
/// `O(|u|)` field ops.
fn eval_tensor<F: Field>(u: &[F], x: F) -> F {
    let mut res = F::one();
    let mut xp = x;
    for (k, &uk) in u.iter().enumerate() {
        res *= (F::one() - uk) + uk * xp;
        if k + 1 < u.len() {
            xp = xp.square();
        }
    }
    res
}

/// Bivariate evaluation `f(α,β) = sum_{i,j} F[i,j] α^i β^j` via nested Horner.
#[allow(dead_code)]
pub(crate) fn evaluate_bivariate<F: Field>(
    evals: &[F],
    big_ml: usize,
    big_mr: usize,
    alpha: F,
    beta: F,
) -> F {
    // Evaluate each column at β, then the resulting row vector at α.
    let mut acc = F::zero();
    let mut alpha_pow = F::one();
    for i in 0..big_ml {
        let mut col_val = F::zero();
        let mut beta_pow = F::one();
        for j in 0..big_mr {
            col_val += evals[i + big_ml * j] * beta_pow;
            beta_pow *= beta;
        }
        acc += alpha_pow * col_val;
        alpha_pow *= alpha;
    }
    acc
}

// ════════════════════════════════════════════════════════════════════
// Transcript
// ════════════════════════════════════════════════════════════════════

fn new_transcript<E: Pairing>(
    mu: usize,
    m_left: usize,
    m_right: usize,
    cm_f: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
) -> Result<IOPTranscript<E::ScalarField>, PCSError> {
    let mut t = IOPTranscript::new(DOMAIN);
    t.append_field_element(L_VERSION, &E::ScalarField::from(1u64))?;
    t.append_field_element(L_MU, &E::ScalarField::from(mu as u64))?;
    t.append_field_element(L_ML, &E::ScalarField::from(m_left as u64))?;
    t.append_field_element(L_MR, &E::ScalarField::from(m_right as u64))?;
    t.append_serializable_element(L_CF, &cm_f.0)?;
    t.append_serializable_element(L_POINT, &point.to_vec())?;
    t.append_field_element(L_ETA, value)?;
    Ok(t)
}

/// Draw a nonzero challenge with counter-based resampling.
fn draw_nonzero<F: PrimeField>(
    t: &mut IOPTranscript<F>,
    label: &'static [u8],
) -> Result<F, PCSError> {
    for _ in 0..MAX_CHALLENGE_RETRY {
        let c = t.get_and_append_challenge(label)?;
        if !c.is_zero() {
            return Ok(c);
        }
    }
    Err(PCSError::InvalidParameters(
        "exhausted retries drawing a nonzero challenge".to_string(),
    ))
}

/// Draw a challenge `c` with `c != 0` and `c^2 != 1`, returning `(c, c^{-1})`.
fn draw_reciprocal<F: PrimeField>(
    t: &mut IOPTranscript<F>,
    label: &'static [u8],
) -> Result<(F, F), PCSError> {
    for _ in 0..MAX_CHALLENGE_RETRY {
        let c = t.get_and_append_challenge(label)?;
        if !c.is_zero() && c.square() != F::one() {
            let c_inv = c
                .inverse()
                .ok_or_else(|| PCSError::InvalidParameters("inverse failed".to_string()))?;
            return Ok((c, c_inv));
        }
    }
    Err(PCSError::InvalidParameters(
        "exhausted retries drawing a reciprocal challenge".to_string(),
    ))
}

/// Draw `beta` with the extra constraints `beta != alpha`, `beta^{-1} != alpha`.
fn draw_beta<F: PrimeField>(
    t: &mut IOPTranscript<F>,
    label: &'static [u8],
    alpha: F,
) -> Result<(F, F), PCSError> {
    for _ in 0..MAX_CHALLENGE_RETRY {
        let c = t.get_and_append_challenge(label)?;
        if !c.is_zero() && c.square() != F::one() && c != alpha {
            let c_inv = c
                .inverse()
                .ok_or_else(|| PCSError::InvalidParameters("beta inverse failed".to_string()))?;
            if c_inv != alpha {
                return Ok((c, c_inv));
            }
        }
    }
    Err(PCSError::InvalidParameters(
        "exhausted retries drawing beta".to_string(),
    ))
}

/// Draw `z` with the constraint `z not in {alpha, beta, beta^{-1}}`.
fn draw_z<F: PrimeField>(
    t: &mut IOPTranscript<F>,
    label: &'static [u8],
    alpha: F,
    beta: F,
    beta_inv: F,
) -> Result<F, PCSError> {
    for _ in 0..MAX_CHALLENGE_RETRY {
        let c = t.get_and_append_challenge(label)?;
        if c != alpha && c != beta && c != beta_inv {
            return Ok(c);
        }
    }
    Err(PCSError::InvalidParameters(
        "exhausted retries drawing BDFG20 z".to_string(),
    ))
}

// ════════════════════════════════════════════════════════════════════
// Prover core
// ════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_lines)]
fn chopin_core_open<E: Pairing>(
    pp: &ChopinProverParam<E>,
    evals: &[E::ScalarField],
    mu: usize,
    point: &[E::ScalarField],
    _value: &E::ScalarField,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<ChopinProof<E>, PCSError> {
    check_mu(mu)?;
    if pp.num_vars < mu {
        return Err(PCSError::InvalidParameters(format!(
            "prover param for {} vars < opening mu {mu}",
            pp.num_vars
        )));
    }
    // Use the key's dimensions for the protocol. When the key is larger than
    // the polynomial, pad evaluations and point with zeros.
    let (m_left, m_right) = split_exponents(pp.num_vars);
    let big_ml = pp.big_ml();
    let big_mr = pp.big_mr();
    let n = big_ml * big_mr;
    let mu_key = pp.num_vars;

    // When key is larger than the polynomial, pad evals and point with zeros.
    let mut evals_padded;
    let mut point_padded;
    let (evals, point) = if pp.num_vars > mu {
        evals_padded = evals.to_vec();
        evals_padded.resize(n, E::ScalarField::zero());
        point_padded = point.to_vec();
        point_padded.resize(mu_key, E::ScalarField::zero());
        (&evals_padded[..], &point_padded[..])
    } else {
        (evals, point)
    };
    if evals.len() != n {
        return Err(PCSError::InvalidParameters(format!(
            "polynomial has {} evaluations, expected N={}",
            evals.len(),
            n
        )));
    }
    if pp.g1_powers.len() < n {
        return Err(PCSError::InvalidParameters(format!(
            "SRS G1 length {} insufficient for N={}",
            pp.g1_powers.len(),
            n
        )));
    }

    let _t_total = ScopedTimer::new(BACKEND, mu_key, n, "chopin_open_total", 1, "core-open");

    let z_l = &point[..m_left];
    let z_r = &point[m_left..];

    // ── Psi_R (eq vector for right variables) ──
    let psi_r = {
        let _t =
            ScopedTimer::new(BACKEND, mu, n, "chopin_open_build_psi_r", big_mr, "eq-vec-r");
        build_eq_vec::<E::ScalarField>(z_r, big_mr)
    };

    // ── Step 1: f_zR[i] = sum_j F[i,j] psi_R[j] (column restriction, O(N)) ──
    let f_zr = {
        let _t =
            ScopedTimer::new(BACKEND, mu, n, "chopin_open_restrict_right", n, "restrict");
        compute_restriction(evals, &psi_r, big_ml, big_mr)
    };

    // ── C0 = [f_zR(tau)]_1 ──
    let comm_f_zr = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "chopin_open_commit_f_zr",
            f_zr.len(),
            "tau-slice",
        );
        pp.msm_tau_slice(&f_zr)?
    };
    transcript.append_serializable_element(L_C0, &comm_f_zr)?;

    // ── derive alpha ──
    let alpha = {
        let _t =
            ScopedTimer::new(BACKEND, mu, n, "chopin_open_derive_alpha", 1, "fs");
        transcript.get_and_append_challenge(L_ALPHA)?
    };

    // ── Step 3: synthetic division in X → (q1, f_alpha) (O(N), one pass) ──
    let (q1_coeffs, f_alpha) = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "chopin_open_divide_x_and_fold",
            n,
            "syndiv-X",
        );
        divide_x_at_alpha(evals, big_ml, big_mr, alpha)
    };

    // ── pi_biv_x = [q1(tau,sigma)]_1 (the single N-scale MSM) ──
    let pi_biv_x = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "chopin_open_commit_q1",
            q1_coeffs.len(),
            "prefix-MSM",
        );
        pp.msm_q1_prefix(&q1_coeffs)?
    };

    // ── C1 = [f_alpha(tau)]_1 ──
    let comm_f_alpha = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "chopin_open_commit_f_alpha",
            f_alpha.len(),
            "tau-slice",
        );
        pp.msm_tau_slice(&f_alpha)?
    };

    // ── a = f_zR(alpha) ──
    let a = {
        let _t = ScopedTimer::new(BACKEND, mu, n, "chopin_open_eval_a", 1, "horner");
        poly_eval(&f_zr, alpha)
    };

    transcript.append_serializable_element(L_C1, &comm_f_alpha)?;
    transcript.append_field_element(L_A, &a)?;

    // ── derive gamma (batched IPA challenge) ──
    let gamma = {
        let _t =
            ScopedTimer::new(BACKEND, mu, n, "chopin_open_derive_gamma", 1, "fs");
        draw_nonzero(transcript, L_GAMMA)?
    };

    // ── S witness: S = S0 + gamma·S1 ──
    let s_coeffs = {
        let _t =
            ScopedTimer::new(BACKEND, mu, n, "chopin_open_build_s", big_ml, "reciprocal");
        let s0 = symmetric_lagrange_witness(&f_zr, z_l, m_left);
        let s1 = symmetric_lagrange_witness(&f_alpha, z_r, m_right);
        let mut s = vec![E::ScalarField::zero(); big_ml - 1];
        for (i, &c0) in s0.iter().enumerate() {
            s[i] = c0;
        }
        for (i, &c1) in s1.iter().enumerate() {
            s[i] += gamma * c1;
        }
        s
    };

    // ── CS = [S(tau)]_1 ──
    let comm_s = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "chopin_open_commit_s",
            s_coeffs.len(),
            "tau-slice",
        );
        pp.msm_tau_slice(&s_coeffs)?
    };
    transcript.append_serializable_element(L_CS, &comm_s)?;

    // ── derive beta ──
    let (beta, beta_inv) = {
        let _t =
            ScopedTimer::new(BACKEND, mu, n, "chopin_open_derive_beta", 1, "fs");
        draw_beta(transcript, L_BETA, alpha)?
    };

    // ── evaluations at beta, beta^{-1} ──
    let (a1, a2, b1, b2, s1, s2) = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "chopin_open_eval_claims",
            6,
            "horner-6",
        );
        let a1 = poly_eval(&f_zr, beta);
        let a2 = poly_eval(&f_zr, beta_inv);
        let b1 = poly_eval(&f_alpha, beta);
        let b2 = poly_eval(&f_alpha, beta_inv);
        let s1 = poly_eval(&s_coeffs, beta);
        let s2 = poly_eval(&s_coeffs, beta_inv);
        (a1, a2, b1, b2, s1, s2)
    };

    // ── q2(Y) = (f_alpha(Y) - b1) / (Y - beta) ──
    let (q2_coeffs, _q2_rem) = {
        let _t =
            ScopedTimer::new(BACKEND, mu, n, "chopin_open_divide_y", big_mr, "syndiv-Y");
        divide_y_at_beta(&f_alpha, beta)?
    };

    // ── pi_biv_y = [q2(sigma)]_1 ──
    let pi_biv_y = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "chopin_open_commit_q2",
            q2_coeffs.len(),
            "sigma-slice",
        );
        pp.msm_sigma_slice(&q2_coeffs)?
    };

    transcript.append_field_element(L_A1, &a1)?;
    transcript.append_field_element(L_A2, &a2)?;
    transcript.append_field_element(L_B1, &b1)?;
    transcript.append_field_element(L_B2, &b2)?;
    transcript.append_field_element(L_S1, &s1)?;
    transcript.append_field_element(L_S2, &s2)?;
    transcript.append_serializable_element(L_PI_X, &pi_biv_x)?;
    transcript.append_serializable_element(L_PI_Y, &pi_biv_y)?;

    // ── BDFG20 batch opening of {f_zR, f_alpha, S} at {α,β,β^{-1}} ──
    let (batch_w, batch_w_prime) = bdfg_prove_chopin(
        pp, mu, n, transcript, &f_zr, &f_alpha, &s_coeffs,
        alpha, beta, beta_inv, a, a1, a2, b1, b2, s1, s2,
    )?;

    Ok(ChopinProof {
        comm_f_zr,
        comm_f_alpha,
        comm_s,
        pi_biv_x,
        pi_biv_y,
        batch_w,
        batch_w_prime,
        a,
        a1,
        a2,
        b1,
        b2,
        s1,
        s2,
        mu: pp.num_vars as u32,
    })
}

#[allow(clippy::too_many_arguments)]
fn bdfg_prove_chopin<E: Pairing>(
    pp: &ChopinProverParam<E>,
    mu: usize,
    n: usize,
    transcript: &mut IOPTranscript<E::ScalarField>,
    f_zr: &[E::ScalarField],
    f_alpha: &[E::ScalarField],
    s: &[E::ScalarField],
    alpha: E::ScalarField,
    beta: E::ScalarField,
    beta_inv: E::ScalarField,
    a: E::ScalarField,
    a1: E::ScalarField,
    a2: E::ScalarField,
    b1: E::ScalarField,
    b2: E::ScalarField,
    s1: E::ScalarField,
    s2: E::ScalarField,
) -> Result<(E::G1Affine, E::G1Affine), PCSError> {
    // rho (BDFG batching challenge)
    let rho = draw_nonzero(transcript, L_RHO)?;

    let claims = [
        BdfgClaim {
            poly: f_zr,
            points: &[alpha, beta, beta_inv],
            values: &[a, a1, a2],
        },
        BdfgClaim {
            poly: f_alpha,
            points: &[beta, beta_inv],
            values: &[b1, b2],
        },
        BdfgClaim {
            poly: s,
            points: &[beta, beta_inv],
            values: &[s1, s2],
        },
    ];

    // Round 1
    let first = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "chopin_open_bdfg_build_w",
            1,
            "bdfg-1",
        );
        bdfg_first_round(&claims, rho)?
    };

    // commit W
    let batch_w = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "chopin_open_bdfg_commit_w",
            first.quot_m.len(),
            "tau-slice",
        );
        pp.msm_tau_slice(&first.quot_m)?
    };
    transcript.append_serializable_element(L_W, &batch_w)?;

    // derive z
    let z = draw_z(transcript, L_Z_BDFG, alpha, beta, beta_inv)?;

    // Round 2
    let second = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "chopin_open_bdfg_build_w_prime",
            1,
            "bdfg-2",
        );
        bdfg_second_round(&claims, &first, rho, z)?
    };

    // commit W'
    let batch_w_prime = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "chopin_open_bdfg_commit_w_prime",
            second.quot_l.len(),
            "tau-slice",
        );
        pp.msm_tau_slice(&second.quot_l)?
    };
    transcript.append_serializable_element(L_WP, &batch_w_prime)?;

    Ok((batch_w, batch_w_prime))
}

// ════════════════════════════════════════════════════════════════════
// Verifier core
// ════════════════════════════════════════════════════════════════════

fn chopin_core_verify<E: Pairing>(
    vp: &ChopinVerifierParam<E>,
    com: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &ChopinProof<E>,
) -> Result<bool, PCSError> {
    let mu = proof.mu as usize;
    check_mu_verify(mu)?;
    if mu > vp.max_num_vars {
        return Err(PCSError::InvalidProof(format!(
            "proof.mu {} exceeds verifier key capacity {}",
            mu, vp.max_num_vars
        )));
    }
    let (m_left, m_right) = split_exponents(mu);
    let big_ml = 1usize
        .checked_shl(m_left as u32)
        .ok_or_else(|| PCSError::InvalidProof("M_L overflow".to_string()))?;
    let big_mr = 1usize
        .checked_shl(m_right as u32)
        .ok_or_else(|| PCSError::InvalidProof("M_R overflow".to_string()))?;
    let n = big_ml
        .checked_mul(big_mr)
        .ok_or_else(|| PCSError::InvalidProof("N overflow".to_string()))?;
    if point.len() != mu {
        // When the verifier key supports more variables than the polynomial,
        // pad the point with zeros (the extra variables are implicitly 0).
        if point.len() > mu || mu - point.len() > usize::BITS as usize {
            return Err(PCSError::InvalidProof(format!(
                "point length {} vs proof.mu {}",
                point.len(),
                mu
            )));
        }
        // pad with zeros
    }
    let point: Vec<E::ScalarField> = {
        let mut p = point.to_vec();
        p.resize(mu, E::ScalarField::zero());
        p
    };

    let _t_total = ScopedTimer::new(BACKEND, mu, n, "chopin_verify_total", 1, "verify");

    // ── replay transcript ──
    let _t_tr = ScopedTimer::new(BACKEND, mu, n, "chopin_verify_transcript", 1, "replay");
    let mut transcript = new_transcript::<E>(mu, m_left, m_right, com, &point, value)?;
    transcript.append_serializable_element(L_C0, &proof.comm_f_zr)?;
    let alpha = transcript.get_and_append_challenge(L_ALPHA)?;
    transcript.append_serializable_element(L_C1, &proof.comm_f_alpha)?;
    transcript.append_field_element(L_A, &proof.a)?;
    let gamma = draw_nonzero(&mut transcript, L_GAMMA)?;
    transcript.append_serializable_element(L_CS, &proof.comm_s)?;
    let (beta, beta_inv) = draw_beta(&mut transcript, L_BETA, alpha)?;
    transcript.append_field_element(L_A1, &proof.a1)?;
    transcript.append_field_element(L_A2, &proof.a2)?;
    transcript.append_field_element(L_B1, &proof.b1)?;
    transcript.append_field_element(L_B2, &proof.b2)?;
    transcript.append_field_element(L_S1, &proof.s1)?;
    transcript.append_field_element(L_S2, &proof.s2)?;
    transcript.append_serializable_element(L_PI_X, &proof.pi_biv_x)?;
    transcript.append_serializable_element(L_PI_Y, &proof.pi_biv_y)?;

    let z_l = &point[..m_left];
    let z_r = &point[m_left..];

    // ── IPA identity check ──
    let _t_ipa = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "chopin_verify_ipa_identity",
        1,
        "ipa-field",
    );
    let psi_l_beta = eval_tensor(z_l, beta);
    let psi_l_beta_inv = eval_tensor(z_l, beta_inv);
    let psi_r_beta = eval_tensor(z_r, beta);
    let psi_r_beta_inv = eval_tensor(z_r, beta_inv);
    let lhs = proof.a1 * psi_l_beta_inv
        + proof.a2 * psi_l_beta
        + gamma * (proof.b1 * psi_r_beta_inv + proof.b2 * psi_r_beta);
    let rhs = (*value + gamma * proof.a).double() + beta * proof.s1 + beta_inv * proof.s2;
    drop(_t_ipa);
    if lhs != rhs {
        return Ok(false);
    }

    // ── bivariate KZG check (3-term multi_pairing) ──
    // dynamic G2 elements
    let _t_g2 = ScopedTimer::new(BACKEND, mu, n, "chopin_verify_g2_scalars", 2, "G2-mul");
    let tau_minus_alpha =
        (vp.g2_tau.into_group() - vp.g2_one.into_group() * alpha).into_affine();
    let sigma_minus_beta =
        (vp.g2_sigma.into_group() - vp.g2_one.into_group() * beta).into_affine();
    drop(_t_g2);

    // C_F - b1·[1]_1
    let cf_minus_b1 = (com.0.into_group() - vp.g1_one.into_group() * proof.b1).into_affine();
    let neg_pi_x = (-proof.pi_biv_x.into_group()).into_affine();
    let neg_pi_y = (-proof.pi_biv_y.into_group()).into_affine();

    let _t_pair_biv = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "chopin_verify_bivariate_multi_pairing",
        3,
        "3-term",
    );
    let ok_biv = E::multi_pairing(
        [cf_minus_b1, neg_pi_x, neg_pi_y],
        [vp.g2_one, tau_minus_alpha, sigma_minus_beta],
    ) == PairingOutput(E::TargetField::one());
    drop(_t_pair_biv);
    if !ok_biv {
        return Ok(false);
    }

    // ── BDFG20 batch check (2-term multi_pairing) ──
    let rho = draw_nonzero(&mut transcript, L_RHO)?;
    transcript.append_serializable_element(L_W, &proof.batch_w)?;
    let z = draw_z(&mut transcript, L_Z_BDFG, alpha, beta, beta_inv)?;
    transcript.append_serializable_element(L_WP, &proof.batch_w_prime)?;

    let bdfg_comb = bdfg_chopin_verify_lhs(vp, proof, alpha, beta, beta_inv, rho, z)?;

    // C_s = Σ scalars[i]*C_i - const_scalar·[1]_1 - Z_T(z)·W
    let _t_msm = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "chopin_verify_bdfg_msm",
        6,
        "6-base-MSM",
    );
    let bases = [
        proof.comm_f_zr,
        proof.comm_f_alpha,
        proof.comm_s,
        vp.g1_one,
        proof.batch_w,
        proof.batch_w_prime,
    ];
    let scalars = [
        bdfg_comb.commit_scalars[0],
        bdfg_comb.commit_scalars[1],
        bdfg_comb.commit_scalars[2],
        -bdfg_comb.const_scalar,
        -bdfg_comb.z_t_z,
        z,
    ];
    let cs_plus_z_wp = E::G1::msm_unchecked(&bases, &scalars).into_affine();
    let neg_wp = (-proof.batch_w_prime.into_group()).into_affine();
    drop(_t_msm);

    let _t_pair_bdfg = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "chopin_verify_bdfg_multi_pairing",
        2,
        "2-term",
    );
    let ok_bdfg = E::multi_pairing(
        [cs_plus_z_wp, neg_wp],
        [vp.g2_one, vp.g2_tau],
    ) == PairingOutput(E::TargetField::one());
    drop(_t_pair_bdfg);

    Ok(ok_bdfg)
}

fn bdfg_chopin_verify_lhs<E: Pairing>(
    _vp: &ChopinVerifierParam<E>,
    proof: &ChopinProof<E>,
    alpha: E::ScalarField,
    beta: E::ScalarField,
    beta_inv: E::ScalarField,
    rho: E::ScalarField,
    z: E::ScalarField,
) -> Result<BdfgVerifierCombination<E::ScalarField>, PCSError> {
    bdfg_verifier_combination(
        &[&[alpha, beta, beta_inv], &[beta, beta_inv], &[beta, beta_inv]],
        &[
            &[proof.a, proof.a1, proof.a2],
            &[proof.b1, proof.b2],
            &[proof.s1, proof.s2],
        ],
        rho,
        z,
    )
}

#[cfg(test)]
mod tests;
