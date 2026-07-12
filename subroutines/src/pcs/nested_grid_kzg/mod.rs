//! Nested Reciprocal Grid-KZG (NRG-KZG) multilinear polynomial commitment.
//!
//! This implements the candidate construction analysed in
//! `research/pcs-field-map/proof-notes/nested-reciprocal-grid-kzg-mlpcs.md`
//! and `...-soundness.md`. For a multilinear polynomial in `mu` variables it
//! views the evaluation table as an `M_L x M_R` matrix `F[i,j]` and its
//! bivariate twin `f(X,Y) = sum F[i,j] X^i Y^j`, then closes two nested
//! reciprocal inner-product arguments and one aggregated bivariate KZG opening
//! on the reciprocal Cartesian grid `{r,r^{-1}} x {s,s^{-1}}`.
//!
//! Cryptographic proof payload: `4 G1 + 8 F` (448 bytes on BLS12-381 with
//! compressed points), plus non-cryptographic `mu` metadata.
//!
//! # Security caveats (see the design/soundness notes)
//! - This implementation provides NO hiding and NO zero knowledge.
//! - `gen_srs_for_testing` samples the trapdoors locally and is NOT a
//!   production trusted setup.
//! - Statistical soundness is established only in the online-extraction /
//!   ideal-polynomial model. An AGM instantiation still needs an adaptive
//!   two-trapdoor ideal-check lemma; a standard-model proof still needs the
//!   OnlineHomGridExt / ARSDH(2) extractor.
//! - The Fiat-Shamir transcript below fully binds the statement, but ROM
//!   knowledge soundness of the compiled protocol is NOT formally proved.

use crate::pcs::{
    multilinear_kzg::batching::{batch_verify_internal, multi_open_internal, BatchProof},
    prelude::{Commitment, PCSError},
    profile::ScopedTimer,
    PolynomialCommitmentScheme, StructuredReferenceString,
};
use arithmetic::{build_eq_x_r_vec, DenseMultilinearExtension};
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
    split_exponents, NestedGridKzgProverParam, NestedGridKzgUniversalParams,
    NestedGridKzgVerifierParam,
};

const BACKEND: &str = "NestedGridKZG";

/// Maximum number of Fiat-Shamir resampling attempts for a rejected challenge.
const MAX_CHALLENGE_RETRY: usize = 64;

// Transcript labels.
const DS: &[u8] = b"nested-grid-kzg-v1";
const LABEL_VERSION: &[u8] = b"nrg::version";
const LABEL_MU: &[u8] = b"nrg::mu";
const LABEL_ML: &[u8] = b"nrg::m_left";
const LABEL_MR: &[u8] = b"nrg::m_right";
const LABEL_CF: &[u8] = b"nrg::cm_f";
const LABEL_POINT: &[u8] = b"nrg::point";
const LABEL_VALUE: &[u8] = b"nrg::value";
const LABEL_CM_S0: &[u8] = b"nrg::cm_s0";
const LABEL_R: &[u8] = b"nrg::r";
const LABEL_A_PLUS: &[u8] = b"nrg::a_plus";
const LABEL_A_MINUS: &[u8] = b"nrg::a_minus";
const LABEL_T0_PLUS: &[u8] = b"nrg::t0_plus";
const LABEL_LAMBDA: &[u8] = b"nrg::lambda";
const LABEL_CM_S1: &[u8] = b"nrg::cm_s1";
const LABEL_S: &[u8] = b"nrg::s";
const LABEL_V_PP: &[u8] = b"nrg::v_pp";
const LABEL_V_PN: &[u8] = b"nrg::v_pn";
const LABEL_V_NP: &[u8] = b"nrg::v_np";
const LABEL_V_NN: &[u8] = b"nrg::v_nn";
const LABEL_T1_PLUS: &[u8] = b"nrg::t1_plus";
const LABEL_ETA: &[u8] = b"nrg::eta";

/// NRG-KZG scheme handle.
pub struct NestedGridKzgPCS<E: Pairing> {
    phantom: PhantomData<E>,
}

/// NRG-KZG opening proof: 4 G1 elements + 8 field elements (+ `mu` metadata).
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct NestedGridKzgProof<E: Pairing> {
    /// `C_0 = [S_0(tau)]_1`.
    pub cm_s0: E::G1Affine,
    /// `C_1 = [S_1(sigma)]_1`.
    pub cm_s1: E::G1Affine,
    /// `Pi_X = [Q_X + eta W_0]_1`.
    pub pi_x: E::G1Affine,
    /// `Pi_Y = [Q_Y + eta^2 W_1]_1`.
    pub pi_y: E::G1Affine,
    /// `a_+ = g(r)`.
    pub a_plus: E::ScalarField,
    /// `a_- = g(r^{-1})`.
    pub a_minus: E::ScalarField,
    /// `t_{0,+} = S_0(r)`.
    pub t0_plus: E::ScalarField,
    /// `v_{++} = f(r,s)`.
    pub v_pp: E::ScalarField,
    /// `v_{+-} = f(r,s^{-1})`.
    pub v_pn: E::ScalarField,
    /// `v_{-+} = f(r^{-1},s)`.
    pub v_np: E::ScalarField,
    /// `v_{--} = f(r^{-1},s^{-1})`.
    pub v_nn: E::ScalarField,
    /// `t_{1,+} = S_1(s)`.
    pub t1_plus: E::ScalarField,
    /// Number of variables (non-cryptographic metadata, bound into transcript).
    pub mu: u32,
}

impl<E: Pairing> NestedGridKzgProof<E> {
    /// Cryptographic payload size in bytes: `4 * |G1_compressed| + 8 * |F|`.
    pub fn cryptographic_payload_bytes(&self) -> usize {
        4 * self.cm_s0.compressed_size() + 8 * self.a_plus.compressed_size()
    }
}

// ════════════════════════════════════════════════════════════════════
// PolynomialCommitmentScheme trait
// ════════════════════════════════════════════════════════════════════

impl<E: Pairing> PolynomialCommitmentScheme<E> for NestedGridKzgPCS<E> {
    type ProverParam = NestedGridKzgProverParam<E>;
    type VerifierParam = NestedGridKzgVerifierParam<E>;
    type SRS = NestedGridKzgUniversalParams<E>;
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    type Commitment = Commitment<E>;
    type Proof = NestedGridKzgProof<E>;
    type BatchProof = BatchProof<E, Self>;

    fn gen_srs_for_testing<R: Rng>(rng: &mut R, nv: usize) -> Result<Self::SRS, PCSError> {
        NestedGridKzgUniversalParams::<E>::gen_srs_for_testing(rng, nv)
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
        let mu = poly.num_vars;
        check_mu(mu)?;
        if pp.num_vars != mu {
            return Err(PCSError::InvalidParameters(format!(
                "prover param is for {} vars but polynomial has {}",
                pp.num_vars, mu
            )));
        }
        let n = pp.n();
        if poly.evaluations.len() != n {
            return Err(PCSError::InvalidParameters(format!(
                "polynomial has {} evaluations, expected N={}",
                poly.evaluations.len(),
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

        let scalars = {
            let _t = ScopedTimer::new(BACKEND, mu, n, "nrg_commit_reorder", n, "layout-reorder");
            layout_scalars(pp, &poly.evaluations)
        };
        let _t = ScopedTimer::new(BACKEND, mu, n, "nrg_commit_msm", n, "N-MSM");
        let cm = pp.msm_prefix(&scalars)?;
        Ok(Commitment(cm))
    }

    fn open(
        pp: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let pp = pp.borrow();
        let mu = poly.num_vars;
        let n = 1usize.checked_shl(mu as u32).unwrap_or(0);
        let _t = ScopedTimer::new(BACKEND, mu, n, "nrg_open_trait_total", 1, "trait-open");
        // The trait `open` does not receive a commitment, so we must recompute
        // C_f here to bind it into the transcript. This recommitment is an
        // N-MSM that is NOT part of the four theoretical proof MSMs.
        let commitment = {
            let _t = ScopedTimer::new(
                BACKEND,
                mu,
                n,
                "nrg_open_statement_recommit",
                n,
                "recommit-C_f",
            );
            Self::commit(pp, poly)?
        };
        Self::open_with_commitment(pp, poly, point, &commitment)
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
        nested_grid_verify(vp, com, point, value, proof)
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

impl<E: Pairing> NestedGridKzgPCS<E> {
    /// Core opening using an already-computed commitment. This avoids the
    /// statement recommitment cost incurred by the `PolynomialCommitmentScheme`
    /// trait `open`, and is what benchmarks label `core_open`.
    pub fn open_with_commitment(
        pp: &NestedGridKzgProverParam<E>,
        poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
        point: &[E::ScalarField],
        commitment: &Commitment<E>,
    ) -> Result<(NestedGridKzgProof<E>, E::ScalarField), PCSError> {
        nested_grid_open(pp, poly, point, commitment)
    }
}

// ════════════════════════════════════════════════════════════════════
// Shared helpers
// ════════════════════════════════════════════════════════════════════

#[inline]
fn check_mu(mu: usize) -> Result<(), PCSError> {
    if mu < 4 {
        return Err(PCSError::InvalidParameters(format!(
            "nested-grid-kzg requires mu >= 4, got {}",
            mu
        )));
    }
    if mu >= usize::BITS as usize {
        return Err(PCSError::InvalidParameters(format!(
            "mu {} exceeds platform word size",
            mu
        )));
    }
    Ok(())
}

/// Reorder canonical evaluations `F[i + M_L*j]` into the dominant-QX-first
/// layout so a single prefix MSM commits the bivariate twin.
fn layout_scalars<E: Pairing>(
    pp: &NestedGridKzgProverParam<E>,
    evals: &[E::ScalarField],
) -> Vec<E::ScalarField> {
    let big_ml = pp.big_ml();
    let big_mr = pp.big_mr();
    let n = big_ml * big_mr;
    let mut scalars = vec![E::ScalarField::zero(); n];
    for j in 0..big_mr {
        for i in 0..big_ml {
            scalars[pp.base_index(i, j)] = evals[i + big_ml * j];
        }
    }
    scalars
}

/// Univariate Horner evaluation.
fn horner<F: Field>(coeffs: &[F], x: F) -> F {
    let mut acc = F::zero();
    for c in coeffs.iter().rev() {
        acc = acc * x + *c;
    }
    acc
}

/// Tensor polynomial `prod_k ((1-p_k) + p_k x^{2^k})` at scalar `x`. O(len).
fn eval_tensor<F: Field>(point: &[F], x: F) -> F {
    let mut res = F::one();
    let mut xp = x;
    for (k, &pk) in point.iter().enumerate() {
        res *= (F::one() - pk) + pk * xp;
        if k + 1 < point.len() {
            xp = xp.square();
        }
    }
    res
}

/// Structured reciprocal witness. Given `a` of length `M = 2^m` and the tensor
/// point `p` (length `m`) whose polynomial is
/// `psi(X) = prod_k ((1-p_k) + p_k X^{2^k})`, returns `S` of length `M-1`
/// (degree `<= M-2`) with
///   `a(X)psi(X^{-1}) + a(X^{-1})psi(X) = 2<a,psi> + X S(X) + X^{-1}
/// S(X^{-1})`.
///
/// Complexity `O(M log M)` via `m` structured shift-add passes over two
/// preallocated ping-pong buffers (no per-pass allocation, no dense O(M^2)
/// convolution).
fn reciprocal_witness<F: Field>(a: &[F], p: &[F], m: usize) -> Vec<F> {
    let big_m = a.len();
    debug_assert_eq!(big_m, 1usize << m);
    let offset = big_m - 1;
    let len = 2 * big_m - 1;
    let mut cur = vec![F::zero(); len];
    let mut next = vec![F::zero(); len];
    // W starts as a(X): coefficient of X^i at position offset+i.
    for (i, &ai) in a.iter().enumerate() {
        cur[offset + i] = ai;
    }
    // Multiply by psi(X^{-1}) = prod_k ((1-p_k) + p_k X^{-2^k}) one factor at a
    // time: new[i] = (1-p_k) cur[i] + p_k cur[i+s], with s = 2^k.
    for (k, &pk) in p.iter().take(m).enumerate() {
        let s = 1usize << k;
        let ompk = F::one() - pk;
        for i in 0..len {
            let mut v = ompk * cur[i];
            if i + s < len {
                v += pk * cur[i + s];
            }
            next[i] = v;
        }
        core::mem::swap(&mut cur, &mut next);
    }
    // S[i] = W_{i+1} + W_{-(i+1)}.
    let mut s = vec![F::zero(); big_m - 1];
    for (i, si) in s.iter_mut().enumerate() {
        *si = cur[offset + (i + 1)] + cur[offset - (i + 1)];
    }
    s
}

/// Degree-1 interpolant `[b0, b1]` (constant, linear) through `(x0,y0)` and
/// `(x1,y1)`. Fails on duplicate abscissae.
fn two_point_remainder<F: Field>(x0: F, y0: F, x1: F, y1: F) -> Result<[F; 2], PCSError> {
    let denom = x1 - x0;
    let inv = denom.inverse().ok_or_else(|| {
        PCSError::InvalidProof("duplicate interpolation points (degenerate divisor)".to_string())
    })?;
    let b1 = (y1 - y0) * inv;
    let b0 = y0 - b1 * x0;
    Ok([b0, b1])
}

/// Divide the univariate coefficient slice `c` by the monic quadratic
/// `Z(X) = X^2 - p X + 1` via synthetic division. Returns `(quotient, [rem0,
/// rem1])`. The quotient has length `c.len().saturating_sub(2)`.
fn div_by_monic_quadratic<F: Field>(c: &[F], p: F) -> (Vec<F>, [F; 2]) {
    let d = c.len();
    if d <= 2 {
        let mut rem = [F::zero(); 2];
        rem[..d].copy_from_slice(&c[..d]);
        return (Vec::new(), rem);
    }
    let mut rem = c.to_vec();
    let mut q = vec![F::zero(); d - 2];
    // Z(X)*coeff*X^{i-2} = coeff X^i - p coeff X^{i-1} + coeff X^{i-2}.
    for i in (2..d).rev() {
        let coeff = rem[i];
        if coeff.is_zero() {
            continue;
        }
        q[i - 2] = coeff;
        rem[i - 1] += p * coeff;
        rem[i - 2] -= coeff;
        // rem[i] is now logically zero and no longer read.
    }
    (q, [rem[0], rem[1]])
}

// ════════════════════════════════════════════════════════════════════
// Transcript
// ════════════════════════════════════════════════════════════════════

fn new_transcript<E: Pairing>(
    mu: usize,
    m_left: usize,
    m_right: usize,
    cm_f: &E::G1Affine,
    point: &[E::ScalarField],
    value: &E::ScalarField,
) -> Result<IOPTranscript<E::ScalarField>, PCSError> {
    let mut t = IOPTranscript::new(DS);
    t.append_message(LABEL_VERSION, DS)?;
    t.append_field_element(LABEL_MU, &E::ScalarField::from(mu as u64))?;
    t.append_field_element(LABEL_ML, &E::ScalarField::from(m_left as u64))?;
    t.append_field_element(LABEL_MR, &E::ScalarField::from(m_right as u64))?;
    t.append_serializable_element(LABEL_CF, cm_f)?;
    t.append_serializable_element(LABEL_POINT, &point.to_vec())?;
    t.append_field_element(LABEL_VALUE, value)?;
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

/// Draw a challenge `c` with `c != 0` and `c^2 != 1` (so `c, c^{-1}` are
/// distinct nonzero points), returning `(c, c^{-1})`. Counter-based resampling.
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

// ════════════════════════════════════════════════════════════════════
// Prover
// ════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_lines)]
fn nested_grid_open<E: Pairing>(
    pp: &NestedGridKzgProverParam<E>,
    poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
    point: &[E::ScalarField],
    commitment: &Commitment<E>,
) -> Result<(NestedGridKzgProof<E>, E::ScalarField), PCSError> {
    let mu = poly.num_vars;
    check_mu(mu)?;
    let (m_left, m_right) = split_exponents(mu);
    if pp.num_vars != mu {
        return Err(PCSError::InvalidParameters(format!(
            "prover param is for {} vars but polynomial has {}",
            pp.num_vars, mu
        )));
    }
    if point.len() != mu {
        return Err(PCSError::InvalidParameters(format!(
            "point length {} != mu {}",
            point.len(),
            mu
        )));
    }
    let big_ml = pp.big_ml();
    let big_mr = pp.big_mr();
    let n = big_ml * big_mr;
    if poly.evaluations.len() != n {
        return Err(PCSError::InvalidParameters(format!(
            "polynomial has {} evaluations, expected N={}",
            poly.evaluations.len(),
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

    let _t_total = ScopedTimer::new(BACKEND, mu, n, "nrg_open_core_total", 1, "core-open");

    let evals = &poly.evaluations;
    let u_l = &point[..m_left];
    let u_r = &point[m_left..];

    // value = f~(u); bind it.
    let value = poly
        .evaluate(point)
        .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;
    let mut transcript = new_transcript::<E>(mu, m_left, m_right, &commitment.0, point, &value)?;

    // ── psi_L, psi_R coefficients ──
    let (psi_l, psi_r) = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "nrg_open_build_psi",
            big_ml + big_mr,
            "eq-vecs",
        );
        let psi_l = build_eq_x_r_vec(u_l)?;
        let psi_r = build_eq_x_r_vec(u_r)?;
        (psi_l, psi_r)
    };

    // ── g[i] = sum_j F[i,j] psi_R[j] (matrix-vector, O(N)) ──
    let g = {
        let _t = ScopedTimer::new(BACKEND, mu, n, "nrg_open_compute_g", n, "matvec");
        compute_g::<E>(evals, &psi_r, big_ml, big_mr)
    };
    // y = <g, psi_L>; must equal the claimed value.
    let mut y = E::ScalarField::zero();
    for (gi, pli) in g.iter().zip(psi_l.iter()) {
        y += *gi * *pli;
    }
    if y != value {
        return Err(PCSError::InvalidProver(
            "claimed value inconsistent with committed polynomial".to_string(),
        ));
    }

    // ── S0 (reciprocal witness, length M_L-1) ──
    let s0 = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "nrg_open_compute_s0",
            big_ml,
            "reciprocal-S0",
        );
        reciprocal_witness(&g, u_l, m_left)
    };

    // ── commit S0: bases [tau^i] (Y^0 slice), i in 0..M_L-1 ──
    let cm_s0 = {
        let _t = ScopedTimer::new(BACKEND, mu, n, "nrg_open_commit_s0", big_ml - 1, "MSM-S0");
        let indices: Vec<usize> = (0..big_ml - 1).map(|i| pp.base_index(i, 0)).collect();
        pp.msm_collected(&indices, &s0)?
    };
    transcript.append_serializable_element(LABEL_CM_S0, &cm_s0)?;
    let (r, r_inv) = draw_reciprocal(&mut transcript, LABEL_R)?;

    // ── outer values a_+, a_-, t0_+ ──
    let (a_plus, a_minus, t0_plus) = {
        let _t = ScopedTimer::new(BACKEND, mu, n, "nrg_open_eval_outer", 3, "outer-evals");
        (horner(&g, r), horner(&g, r_inv), horner(&s0, r))
    };
    transcript.append_field_element(LABEL_A_PLUS, &a_plus)?;
    transcript.append_field_element(LABEL_A_MINUS, &a_minus)?;
    transcript.append_field_element(LABEL_T0_PLUS, &t0_plus)?;
    let lambda = draw_nonzero(&mut transcript, LABEL_LAMBDA)?;

    // ── Y-restrictions h_+(Y)=f(r,Y), h_-(Y)=f(r^{-1},Y) (O(2N)) ──
    let (h_plus, h_minus) = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "nrg_open_compute_restrictions",
            2 * n,
            "columns",
        );
        compute_restrictions::<E>(evals, r, r_inv, big_ml, big_mr)
    };
    // H_lambda(Y) = h_+ + lambda h_-, A_lambda = a_+ + lambda a_-.
    let mut h_comb = vec![E::ScalarField::zero(); big_mr];
    for j in 0..big_mr {
        h_comb[j] = h_plus[j] + lambda * h_minus[j];
    }
    let a_lambda = a_plus + lambda * a_minus;

    // ── S1 (reciprocal witness in Y, length M_R-1) ──
    let s1 = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "nrg_open_compute_s1",
            big_mr,
            "reciprocal-S1",
        );
        reciprocal_witness(&h_comb, u_r, m_right)
    };

    // ── commit S1: bases [sigma^j] (X^0 slice), j in 0..M_R-1 ──
    let cm_s1 = {
        let _t = ScopedTimer::new(BACKEND, mu, n, "nrg_open_commit_s1", big_mr - 1, "MSM-S1");
        let indices: Vec<usize> = (0..big_mr - 1).map(|j| pp.base_index(0, j)).collect();
        pp.msm_collected(&indices, &s1)?
    };
    transcript.append_serializable_element(LABEL_CM_S1, &cm_s1)?;
    let (s, s_inv) = draw_reciprocal(&mut transcript, LABEL_S)?;

    // ── grid values and t1_+ ──
    let (v_pp, v_pn, v_np, v_nn, t1_plus) = {
        let _t = ScopedTimer::new(BACKEND, mu, n, "nrg_open_eval_grid", 5, "grid-evals");
        (
            horner(&h_plus, s),
            horner(&h_plus, s_inv),
            horner(&h_minus, s),
            horner(&h_minus, s_inv),
            horner(&s1, s),
        )
    };
    transcript.append_field_element(LABEL_V_PP, &v_pp)?;
    transcript.append_field_element(LABEL_V_PN, &v_pn)?;
    transcript.append_field_element(LABEL_V_NP, &v_np)?;
    transcript.append_field_element(LABEL_V_NN, &v_nn)?;
    transcript.append_field_element(LABEL_T1_PLUS, &t1_plus)?;
    let eta = draw_nonzero(&mut transcript, LABEL_ETA)?;
    let eta2 = eta.square();

    // ── recover t0_-, t1_- (verifier computes the same) ──
    let psi_l_r = eval_tensor(u_l, r);
    let psi_l_rinv = eval_tensor(u_l, r_inv);
    let psi_r_s = eval_tensor(u_r, s);
    let psi_r_sinv = eval_tensor(u_r, s_inv);
    let t0_minus = r * (a_plus * psi_l_rinv + a_minus * psi_l_r - y.double() - r * t0_plus);
    let h_s = v_pp + lambda * v_np;
    let h_s_inv = v_pn + lambda * v_nn;
    let t1_minus = s * (h_s * psi_r_sinv + h_s_inv * psi_r_s - a_lambda.double() - s * t1_plus);

    // ── interpolation ──
    let (i_coeffs, l0, l1) = {
        let _t = ScopedTimer::new(BACKEND, mu, n, "nrg_open_interpolate", 1, "grid-interp");
        // I(X, s), I(X, s^{-1}) as linear-in-X interpolants.
        let px_s = two_point_remainder(r, v_pp, r_inv, v_np)?;
        let px_sinv = two_point_remainder(r, v_pn, r_inv, v_nn)?;
        // I_{k,·}(Y): interpolate each X-coefficient across Y.
        let i0 = two_point_remainder(s, px_s[0], s_inv, px_sinv[0])?; // [I00, I01]
        let i1 = two_point_remainder(s, px_s[1], s_inv, px_sinv[1])?; // [I10, I11]
        let l0 = two_point_remainder(r, t0_plus, r_inv, t0_minus)?; // [L0_0, L0_1]
        let l1 = two_point_remainder(s, t1_plus, s_inv, t1_minus)?; // [L1_0, L1_1]
        ([i0, i1], l0, l1)
    };
    let i0 = i_coeffs[0];
    let i1 = i_coeffs[1];
    let p_r = r + r_inv;
    let q_s = s + s_inv;

    // ── bivariate quotient: (f - I) = Z_A Q_X + Z_B Q_Y ──
    let qx_len = pp.qx_len();
    let mut pi_x_scalars = vec![E::ScalarField::zero(); qx_len];
    let (q_y0, q_y1) = {
        let _t = ScopedTimer::new(BACKEND, mu, n, "nrg_open_divide_grid", n, "quad-syndiv-X");
        let mut r0_col = vec![E::ScalarField::zero(); big_mr];
        let mut r1_col = vec![E::ScalarField::zero(); big_mr];
        let stride = big_ml - 2;
        for j in 0..big_mr {
            let mut col = evals[big_ml * j..big_ml * (j + 1)].to_vec();
            // Subtract the I(X,Y) contribution to the Y^j coefficient.
            if j == 0 {
                col[0] -= i0[0];
                col[1] -= i1[0];
            } else if j == 1 {
                col[0] -= i0[1];
                col[1] -= i1[1];
            }
            let (q, rem) = div_by_monic_quadratic(&col, p_r);
            let block = j * stride;
            pi_x_scalars[block..block + q.len()].copy_from_slice(&q);
            r0_col[j] = rem[0];
            r1_col[j] = rem[1];
        }
        // Divide the deg_X<2 remainder by Z_B(Y); remainders must vanish.
        let (q_y0, rem_y0) = div_by_monic_quadratic(&r0_col, q_s);
        let (q_y1, rem_y1) = div_by_monic_quadratic(&r1_col, q_s);
        if !rem_y0[0].is_zero()
            || !rem_y0[1].is_zero()
            || !rem_y1[0].is_zero()
            || !rem_y1[1].is_zero()
        {
            return Err(PCSError::InvalidProver(
                "grid quotient remainder is nonzero (f-I not in <Z_A,Z_B>)".to_string(),
            ));
        }
        (q_y0, q_y1)
    };

    // ── witness quotients: S0-L0 = Z_A W0, S1-L1 = Z_B W1 ──
    let (w0, w1) = {
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "nrg_open_divide_witnesses",
            2,
            "quad-syndiv-W",
        );
        let mut s0m = s0.clone();
        s0m[0] -= l0[0];
        s0m[1] -= l0[1];
        let (w0, rem_w0) = div_by_monic_quadratic(&s0m, p_r);
        if !rem_w0[0].is_zero() || !rem_w0[1].is_zero() {
            return Err(PCSError::InvalidProver(
                "S0-L0 not divisible by Z_A".to_string(),
            ));
        }
        let mut s1m = s1.clone();
        s1m[0] -= l1[0];
        s1m[1] -= l1[1];
        let (w1, rem_w1) = div_by_monic_quadratic(&s1m, q_s);
        if !rem_w1[0].is_zero() || !rem_w1[1].is_zero() {
            return Err(PCSError::InvalidProver(
                "S1-L1 not divisible by Z_B".to_string(),
            ));
        }
        (w0, w1)
    };

    // ── Pi_X = [Q_X + eta W0]_1 (W0 in the Y^0 slice): prefix MSM ──
    let pi_x = {
        let _t = ScopedTimer::new(BACKEND, mu, n, "nrg_open_commit_pi_x", qx_len, "prefix-MSM");
        for (i, w) in w0.iter().enumerate() {
            pi_x_scalars[i] += eta * *w;
        }
        pp.msm_prefix(&pi_x_scalars)?
    };

    // ── Pi_Y = [Q_Y + eta^2 W1]_1 (W1 in the X^0 slice): small collected MSM ──
    let pi_y = {
        let pi_y_len = 2 * (big_mr - 2);
        let _t = ScopedTimer::new(
            BACKEND,
            mu,
            n,
            "nrg_open_commit_pi_y",
            pi_y_len,
            "small-MSM",
        );
        let mut indices = Vec::with_capacity(pi_y_len);
        let mut scalars = Vec::with_capacity(pi_y_len);
        for j in 0..(big_mr - 2) {
            let mut c0 = q_y0[j];
            if j < w1.len() {
                c0 += eta2 * w1[j];
            }
            indices.push(pp.base_index(0, j));
            scalars.push(c0);
            indices.push(pp.base_index(1, j));
            scalars.push(q_y1[j]);
        }
        pp.msm_collected(&indices, &scalars)?
    };

    let proof = NestedGridKzgProof {
        cm_s0,
        cm_s1,
        pi_x,
        pi_y,
        a_plus,
        a_minus,
        t0_plus,
        v_pp,
        v_pn,
        v_np,
        v_nn,
        t1_plus,
        mu: mu as u32,
    };
    Ok((proof, value))
}

/// `g[i] = sum_j F[i,j] psi_R[j]` where `F[i,j] = evals[i + M_L*j]`.
fn compute_g<E: Pairing>(
    evals: &[E::ScalarField],
    psi_r: &[E::ScalarField],
    big_ml: usize,
    big_mr: usize,
) -> Vec<E::ScalarField> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..big_ml)
            .into_par_iter()
            .map(|i| {
                let mut acc = E::ScalarField::zero();
                for j in 0..big_mr {
                    acc += evals[i + big_ml * j] * psi_r[j];
                }
                acc
            })
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut g = vec![E::ScalarField::zero(); big_ml];
        for j in 0..big_mr {
            let pj = psi_r[j];
            let base = big_ml * j;
            for i in 0..big_ml {
                g[i] += evals[base + i] * pj;
            }
        }
        g
    }
}

/// `h_+(Y) = f(r,Y)`, `h_-(Y) = f(r^{-1},Y)`: for each contiguous column
/// `F[0..M_L, j]`, Horner-evaluate at `r` and `r^{-1}` simultaneously.
fn compute_restrictions<E: Pairing>(
    evals: &[E::ScalarField],
    r: E::ScalarField,
    r_inv: E::ScalarField,
    big_ml: usize,
    big_mr: usize,
) -> (Vec<E::ScalarField>, Vec<E::ScalarField>) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let pairs: Vec<(E::ScalarField, E::ScalarField)> = (0..big_mr)
            .into_par_iter()
            .map(|j| {
                let col = &evals[big_ml * j..big_ml * (j + 1)];
                (horner(col, r), horner(col, r_inv))
            })
            .collect();
        let mut h_plus = Vec::with_capacity(big_mr);
        let mut h_minus = Vec::with_capacity(big_mr);
        for (hp, hm) in pairs {
            h_plus.push(hp);
            h_minus.push(hm);
        }
        (h_plus, h_minus)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut h_plus = vec![E::ScalarField::zero(); big_mr];
        let mut h_minus = vec![E::ScalarField::zero(); big_mr];
        for j in 0..big_mr {
            let col = &evals[big_ml * j..big_ml * (j + 1)];
            h_plus[j] = horner(col, r);
            h_minus[j] = horner(col, r_inv);
        }
        (h_plus, h_minus)
    }
}

// ════════════════════════════════════════════════════════════════════
// Verifier
// ════════════════════════════════════════════════════════════════════

fn nested_grid_verify<E: Pairing>(
    vp: &NestedGridKzgVerifierParam<E>,
    com: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &NestedGridKzgProof<E>,
) -> Result<bool, PCSError> {
    // ── integrity checks before any shift / allocation ──
    let mu = proof.mu as usize;
    if mu < 4 {
        return Err(PCSError::InvalidProof(format!(
            "proof.mu {} < 4 unsupported",
            mu
        )));
    }
    if mu >= usize::BITS as usize {
        return Err(PCSError::InvalidProof(format!(
            "proof.mu {} exceeds platform word size",
            mu
        )));
    }
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
        return Err(PCSError::InvalidProof(format!(
            "point length {} != proof.mu {}",
            point.len(),
            mu
        )));
    }

    let _t_total = ScopedTimer::new(BACKEND, mu, n, "nrg_verify_total", 1, "verify");

    // ── replay the transcript ──
    let _t_ts = ScopedTimer::new(BACKEND, mu, n, "nrg_verify_transcript", 1, "replay");
    let mut transcript = new_transcript::<E>(mu, m_left, m_right, &com.0, point, value)?;
    transcript.append_serializable_element(LABEL_CM_S0, &proof.cm_s0)?;
    let (r, r_inv) = draw_reciprocal(&mut transcript, LABEL_R)?;
    transcript.append_field_element(LABEL_A_PLUS, &proof.a_plus)?;
    transcript.append_field_element(LABEL_A_MINUS, &proof.a_minus)?;
    transcript.append_field_element(LABEL_T0_PLUS, &proof.t0_plus)?;
    let lambda = draw_nonzero(&mut transcript, LABEL_LAMBDA)?;
    transcript.append_serializable_element(LABEL_CM_S1, &proof.cm_s1)?;
    let (s, s_inv) = draw_reciprocal(&mut transcript, LABEL_S)?;
    transcript.append_field_element(LABEL_V_PP, &proof.v_pp)?;
    transcript.append_field_element(LABEL_V_PN, &proof.v_pn)?;
    transcript.append_field_element(LABEL_V_NP, &proof.v_np)?;
    transcript.append_field_element(LABEL_V_NN, &proof.v_nn)?;
    transcript.append_field_element(LABEL_T1_PLUS, &proof.t1_plus)?;
    let eta = draw_nonzero(&mut transcript, LABEL_ETA)?;
    let eta2 = eta.square();
    drop(_t_ts);

    // ── tensor evaluations (product form, O(mu)) and recovered witnesses ──
    let _t_interp = ScopedTimer::new(BACKEND, mu, n, "nrg_verify_interpolate", 1, "interp");
    let u_l = &point[..m_left];
    let u_r = &point[m_left..];
    let psi_l_r = eval_tensor(u_l, r);
    let psi_l_rinv = eval_tensor(u_l, r_inv);
    let psi_r_s = eval_tensor(u_r, s);
    let psi_r_sinv = eval_tensor(u_r, s_inv);

    let t0_minus = r
        * (proof.a_plus * psi_l_rinv + proof.a_minus * psi_l_r
            - value.double()
            - r * proof.t0_plus);
    let a_lambda = proof.a_plus + lambda * proof.a_minus;
    let h_s = proof.v_pp + lambda * proof.v_np;
    let h_s_inv = proof.v_pn + lambda * proof.v_nn;
    let t1_minus =
        s * (h_s * psi_r_sinv + h_s_inv * psi_r_s - a_lambda.double() - s * proof.t1_plus);

    // I(X,Y), L0(X), L1(Y).
    let px_s = two_point_remainder(r, proof.v_pp, r_inv, proof.v_np)?;
    let px_sinv = two_point_remainder(r, proof.v_pn, r_inv, proof.v_nn)?;
    let i0 = two_point_remainder(s, px_s[0], s_inv, px_sinv[0])?;
    let i1 = two_point_remainder(s, px_s[1], s_inv, px_sinv[1])?;
    let l0 = two_point_remainder(r, proof.t0_plus, r_inv, t0_minus)?;
    let l1 = two_point_remainder(s, proof.t1_plus, s_inv, t1_minus)?;

    // J_eta(X,Y) = I + eta L0(X) + eta^2 L1(Y). Only 4 monomials.
    let j00 = i0[0] + eta * l0[0] + eta2 * l1[0];
    let j10 = i1[0] + eta * l0[1];
    let j01 = i0[1] + eta2 * l1[1];
    let j11 = i1[1];
    drop(_t_interp);

    // ── C_E via a single <=7-base G1 MSM ──
    let _t_msm = ScopedTimer::new(BACKEND, mu, n, "nrg_verify_g1_msm", 7, "7-base-MSM");
    let bases = [
        com.0,
        proof.cm_s0,
        proof.cm_s1,
        vp.g1_one,
        vp.g1_tau,
        vp.g1_sigma,
        vp.g1_tau_sigma,
    ];
    let scalars = [E::ScalarField::one(), eta, eta2, -j00, -j10, -j01, -j11];
    let c_e = E::G1::msm_unchecked(&bases, &scalars).into_affine();
    drop(_t_msm);

    // ── divisor commitments: 2 dynamic G2 scalar multiplications ──
    let _t_g2 = ScopedTimer::new(BACKEND, mu, n, "nrg_verify_g2_divisors", 2, "G2-scalar-mul");
    let p_r = r + r_inv;
    let q_s = s + s_inv;
    let z_a_g2 = (vp.g2_tau2.into_group() - vp.g2_tau.into_group() * p_r + vp.g2_one.into_group())
        .into_affine();
    let z_b_g2 = (vp.g2_sigma2.into_group() - vp.g2_sigma.into_group() * q_s
        + vp.g2_one.into_group())
    .into_affine();
    drop(_t_g2);

    // ── single three-term multi-pairing ──
    let _t_pair = ScopedTimer::new(BACKEND, mu, n, "nrg_verify_pairing", 3, "3-term-pairing");
    let neg_pi_x = (-proof.pi_x.into_group()).into_affine();
    let neg_pi_y = (-proof.pi_y.into_group()).into_affine();
    let lhs = E::multi_pairing([c_e, neg_pi_x, neg_pi_y], [vp.g2_one, z_a_g2, z_b_g2]);
    Ok(lhs == PairingOutput(E::TargetField::one()))
}

#[cfg(test)]
mod tests;
