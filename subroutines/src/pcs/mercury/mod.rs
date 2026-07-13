//! Mercury — a pairing-based multilinear polynomial commitment scheme with
//! constant proof size and no prover FFTs (ePrint 2025/385, Section 6).
//!
//! For an `mu`-variate multilinear `fhat` with `N = 2^mu` evaluations, Mercury
//! interprets the little-endian evaluation vector as a univariate twin
//!   f(X) = sum_{i<b} X^i f_i(X^b),   b = 2^{ceil(mu/2)} = sqrt(N) (square)
//! laid out as a `b_row x b` matrix (`b_row = 2^{floor(mu/2)}`, columns = low
//! `t = ceil(mu/2)` variables `u1`, rows = high variables `u2`). For an opening
//! point `u = (u1, u2)` and value `v = fhat(u)`, it proves the evaluation with:
//!   1. a restriction polynomial `h(X) = sum_i eq(i,u1) f_i(X)`
//!      (`h(alpha)=ghat(u1)`, `hhat(u2)=v`);
//!   2. a univariate division `f(X) = (X^b - alpha) q(X) + g(X)` with `g` of
//!      degree `< b` (`g(X) = sum_i f_i(alpha) X^i`);
//!   3. two batched symmetric-Laurent Lagrange inner-product claims (witness
//!      `S(X)`), plus a degree check `D(X) = X^{b-1} g(1/X)` for `g`;
//!   4. a single KZG folding proof for step 2 at a challenge `zeta`, and a
//!      BDFG20 batched multi-point KZG opening of `{g,h,S,D}` at `{zeta,
//!      1/zeta, alpha}`.
//!
//! The verifier does `O(log N)` field work, three small G1 MSMs, and a single
//! `multi_pairing` product check with **2 pairing terms** (one Miller loop +
//! one final exponentiation). Proof = 8 G1 + 6 field elements. See
//! `docs/hyperplonk_mercury_design.md` for the full mapping to the paper and to
//! Microsoft Nova's `src/provider/mercury.rs` (MIT), whose Fiat-Shamir schedule
//! and BDFG20 batching this clean-room arkworks rewrite follows. Nova's
//! FFT-based `make_s_polynomial` is replaced by an FFT-free structured
//! computation (see [`make_s_polynomial_structured`]) so the scheme stays
//! generic over any `Pairing` with no `FftField` bound.

use crate::pcs::{
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
use ark_ff::Field;
use ark_poly::MultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::Borrow, format, marker::PhantomData, rand::Rng, string::ToString, sync::Arc, vec,
    vec::Vec, One, Zero,
};
use transcript::IOPTranscript;

pub mod srs;

use srs::{MercuryProverParam, MercuryUniversalParams, MercuryVerifierParam};

const BACKEND: &str = "Mercury";
const DOMAIN: &[u8] = b"mercury-mlpcs-v1";

// Transcript labels (shared by prover and verifier).
const L_VER: &[u8] = b"ver";
const L_MU: &[u8] = b"mu";
const L_T: &[u8] = b"t";
const L_B: &[u8] = b"b";
const L_CF: &[u8] = b"cf";
const L_U: &[u8] = b"u";
const L_E: &[u8] = b"e";
const L_H: &[u8] = b"h";
const L_ALPHA: &[u8] = b"a";
const L_Q: &[u8] = b"q";
const L_G: &[u8] = b"g";
const L_GAMMA: &[u8] = b"gm";
const L_S: &[u8] = b"s";
const L_D: &[u8] = b"d";
const L_ZETA: &[u8] = b"zt";
const L_GZ: &[u8] = b"gz";
const L_GZI: &[u8] = b"gzi";
const L_HZ: &[u8] = b"hz";
const L_HZI: &[u8] = b"hzi";
const L_SZ: &[u8] = b"sz";
const L_SZI: &[u8] = b"szi";
const L_QUOTF: &[u8] = b"tf";
const L_BETA: &[u8] = b"bt";
const L_W: &[u8] = b"w";
const L_ZBDFG: &[u8] = b"z";
const L_WP: &[u8] = b"wp";
const L_DPAIR: &[u8] = b"pd";

const PROTOCOL_VERSION: u64 = 1;

/// Mercury scheme handle.
pub struct MercuryPCS<E: Pairing> {
    phantom: PhantomData<E>,
}

/// Mercury opening proof: 8 G1 commitments + 6 field evaluations + `mu`.
///
/// Every field is used by the verifier. `h(alpha)` and `d(zeta)` are NOT sent;
/// the verifier reconstructs them from the batched Lagrange-IPA identity and
/// the degree-check identity respectively. The six FS challenges are never
/// sent.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct MercuryProof<E: Pairing> {
    /// Commitment to the restriction polynomial `h`.
    pub comm_h: E::G1Affine,
    /// Commitment to the folded polynomial `g` (degree `< b`).
    pub comm_g: E::G1Affine,
    /// Commitment to the quotient `q` of `f / (X^b - alpha)`.
    pub comm_q: E::G1Affine,
    /// Commitment to the batched Lagrange-IPA witness `S`.
    pub comm_s: E::G1Affine,
    /// Commitment to the degree-check polynomial `D = X^{b-1} g(1/X)`.
    pub comm_d: E::G1Affine,
    /// KZG folding proof `pi_z` for `f = (X^b-alpha)q + g` at `zeta`.
    pub comm_quot_f: E::G1Affine,
    /// First BDFG20 witness `W = m / Z_T`.
    pub comm_w: E::G1Affine,
    /// Second BDFG20 witness `W' = L / (X - z)`.
    pub comm_w_prime: E::G1Affine,
    /// `g(zeta)`.
    pub g_zeta: E::ScalarField,
    /// `g(1/zeta)`.
    pub g_zeta_inv: E::ScalarField,
    /// `h(zeta)`.
    pub h_zeta: E::ScalarField,
    /// `h(1/zeta)`.
    pub h_zeta_inv: E::ScalarField,
    /// `S(zeta)`.
    pub s_zeta: E::ScalarField,
    /// `S(1/zeta)`.
    pub s_zeta_inv: E::ScalarField,
    /// Number of variables `mu` (bound into the transcript, checked by
    /// verifier).
    pub mu: usize,
}

// ════════════════════════════════════════════════════════════════════
// PolynomialCommitmentScheme trait
// ════════════════════════════════════════════════════════════════════

impl<E: Pairing> PolynomialCommitmentScheme<E> for MercuryPCS<E> {
    type ProverParam = MercuryProverParam<E>;
    type VerifierParam = MercuryVerifierParam<E>;
    type SRS = MercuryUniversalParams<E>;
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    type Commitment = Commitment<E>;
    type Proof = MercuryProof<E>;
    type BatchProof = BatchProof<E, Self>;

    fn gen_srs_for_testing<R: Rng>(rng: &mut R, nv: usize) -> Result<Self::SRS, PCSError> {
        MercuryUniversalParams::<E>::gen_srs_for_testing(rng, nv)
    }

    fn trim(
        srs: impl Borrow<Self::SRS>,
        _supported_degree: Option<usize>,
        supported_num_vars: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        let nv = supported_num_vars
            .ok_or_else(|| PCSError::InvalidParameters("need num_var".to_string()))?;
        let (_t, _b, _b_row, n) = mercury_dims(nv)?;
        srs.borrow().trim(n - 1)
    }

    fn commit(
        pp: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let pp = pp.borrow();
        let nv = poly.num_vars;
        let (_t, _b, _b_row, n) = mercury_dims(nv)?;
        if pp.max_degree < n - 1 {
            return Err(PCSError::InvalidParameters(format!(
                "SRS max degree {} insufficient for N-1 = {}",
                pp.max_degree,
                n - 1
            )));
        }
        let _t_total = ScopedTimer::new(BACKEND, nv, n, "mercury_commit_total", 1, "total");
        let coeffs = poly.to_evaluations();
        let _t_msm = ScopedTimer::new(
            BACKEND,
            nv,
            n,
            "mercury_commit_msm",
            coeffs.len(),
            "KZG-MSM",
        );
        let cm = pp.commit(&coeffs)?;
        drop(_t_msm);
        Ok(Commitment(cm))
    }

    fn open(
        pp: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let pp = pp.borrow();
        let nv = poly.num_vars();
        let (_t, _b, _b_row, n) = mercury_dims(nv)?;
        if point.len() != nv {
            return Err(PCSError::InvalidParameters(format!(
                "point length {} != mu {}",
                point.len(),
                nv
            )));
        }
        let value = poly
            .evaluate(point)
            .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;
        // trait `open` has no commitment argument, so re-commit C_f (an extra
        // N-MSM). Callers holding C_f should use `open_with_commitment`.
        let _t_recommit = ScopedTimer::new(
            BACKEND,
            nv,
            n,
            "mercury_open_statement_recommit",
            n,
            "recommit-cf",
        );
        let cm_f = MercuryPCS::<E>::commit(pp, poly)?;
        drop(_t_recommit);
        let mut t = new_transcript::<E>(nv, &cm_f, point, &value)?;
        let coeffs = poly.to_evaluations();
        let proof = mercury_core_open(pp, &coeffs, nv, point, &value, &mut t)?;
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
        mercury_core_verify(vp, com, point, value, proof)
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

impl<E: Pairing> MercuryPCS<E> {
    /// Open a polynomial at a point when the caller already holds the
    /// commitment `C_f`, avoiding the extra N-MSM that the trait `open`
    /// performs to re-derive it. `commitment` MUST be `commit(pp, poly)`.
    pub fn open_with_commitment(
        pp: &MercuryProverParam<E>,
        poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
        point: &[E::ScalarField],
        commitment: &Commitment<E>,
    ) -> Result<(MercuryProof<E>, E::ScalarField), PCSError> {
        let nv = poly.num_vars();
        let (_t, _b, _b_row, _n) = mercury_dims(nv)?;
        if point.len() != nv {
            return Err(PCSError::InvalidParameters(format!(
                "point length {} != mu {}",
                point.len(),
                nv
            )));
        }
        let value = poly
            .evaluate(point)
            .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;
        let mut t = new_transcript::<E>(nv, commitment, point, &value)?;
        let coeffs = poly.to_evaluations();
        let proof = mercury_core_open(pp, &coeffs, nv, point, &value, &mut t)?;
        Ok((proof, value))
    }
}

// ════════════════════════════════════════════════════════════════════
// Dimensions / transcript
// ════════════════════════════════════════════════════════════════════

/// Rectangular (non-padding) split: columns = low `t = ceil(mu/2)` variables
/// (`b = 2^t`), rows = high `mu - t` variables (`b_row = 2^{mu-t}`). For even
/// `mu` this is the square `b x b` split; for odd `mu` it is `b x (b/2)`, which
/// keeps the committed `f/q/quot_f` at their original `N = 2^mu` size (only the
/// `O(sqrt N)` helper polynomials grow). Returns `(t, b, b_row, N)`.
fn mercury_dims(mu: usize) -> Result<(usize, usize, usize, usize), PCSError> {
    if mu == 0 {
        return Err(PCSError::InvalidParameters(
            "mu = 0 unsupported (constant polynomial)".to_string(),
        ));
    }
    if mu >= usize::BITS as usize {
        return Err(PCSError::InvalidParameters(format!(
            "mu {} exceeds platform word size",
            mu
        )));
    }
    let t = mu.div_ceil(2);
    let b = 1usize
        .checked_shl(t as u32)
        .ok_or_else(|| PCSError::InvalidParameters(format!("b overflow for t={t}")))?;
    let n = 1usize
        .checked_shl(mu as u32)
        .ok_or_else(|| PCSError::InvalidParameters(format!("N overflow for mu={mu}")))?;
    let b_row = 1usize
        .checked_shl((mu - t) as u32)
        .ok_or_else(|| PCSError::InvalidParameters("b_row overflow".to_string()))?;
    debug_assert_eq!(b * b_row, n);
    Ok((t, b, b_row, n))
}

/// Same as [`mercury_dims`] but with untrusted-input error variant for the
/// verifier path.
fn mercury_dims_verify(mu: usize) -> Result<(usize, usize, usize, usize), PCSError> {
    mercury_dims(mu).map_err(|e| PCSError::InvalidProof(e.to_string()))
}

fn new_transcript<E: Pairing>(
    mu: usize,
    cm_f: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
) -> Result<IOPTranscript<E::ScalarField>, PCSError> {
    let (t, b, _b_row, _n) = mercury_dims(mu)?;
    let mut tr = IOPTranscript::new(DOMAIN);
    tr.append_field_element(L_VER, &E::ScalarField::from(PROTOCOL_VERSION))?;
    tr.append_field_element(L_MU, &E::ScalarField::from(mu as u64))?;
    tr.append_field_element(L_T, &E::ScalarField::from(t as u64))?;
    tr.append_field_element(L_B, &E::ScalarField::from(b as u64))?;
    tr.append_serializable_element(L_CF, &cm_f.0)?;
    tr.append_serializable_element(L_U, &point.to_vec())?;
    tr.append_field_element(L_E, value)?;
    Ok(tr)
}

// ════════════════════════════════════════════════════════════════════
// Univariate helpers (FFT-free) — thin re-exports from the shared bdfg module
// ════════════════════════════════════════════════════════════════════

use crate::pcs::bdfg::{
    add_scaled, divide_by_linear, lagrange_interpolate, mul_by_linear, poly_eval, poly_sub,
    subtract_const,
};

/// `P_u(x) = prod_{k<u.len()} (u_k x^{2^k} + (1-u_k))`, so `[X^i] P_u =
/// eq(i,u)` (little-endian). `O(|u|)` field ops.
fn pu_eval<F: Field>(u: &[F], x: F) -> F {
    let mut acc = F::one();
    let mut x_pow = x;
    for (k, &uk) in u.iter().enumerate() {
        acc *= uk * x_pow + (F::one() - uk);
        if k + 1 < u.len() {
            x_pow = x_pow.square();
        }
    }
    acc
}

/// Divide `p(X)` by `(X - root)`, returning `(quotient, remainder = p(root))`.
/// re-exported from bdfg

/// `p(X) <- p(X) * (X - root)` (returns a new vector one longer).
/// re-exported from bdfg

/// `dst[i] += scale * src[i]` (grows `dst` as needed).
/// re-exported from bdfg

/// Lagrange interpolation through `(xs[i], ys[i])`; `xs` must be pairwise
/// distinct. Supports the small (1..=3) point sets used by BDFG20.
/// re-exported from bdfg

/// `h(X)` column: `h_j = sum_{i<b} coeffs[i + j*b] * eq_col[i]`, `j < b_row`.
/// `O(N)` field ops. Result padded to length `b`.
fn compute_h<F: Field>(coeffs: &[F], eq_col: &[F], b_row: usize, b: usize) -> Vec<F> {
    let mut h = vec![F::zero(); b];
    for (j, hj) in h.iter_mut().enumerate().take(b_row) {
        let mut acc = F::zero();
        for (i, &e) in eq_col.iter().enumerate().take(b) {
            acc += coeffs[i + j * b] * e;
        }
        *hj = acc;
    }
    h
}

/// Univariate division `f(X) = (X^b - alpha) q(X) + g(X)` (Mercury §5). Returns
/// `(g, q)` with `g` of length `b` (`g_i = f_i(alpha)`) and `q` of length
/// `b*(b_row-1)` (`q(X) = sum_i X^i q_i(X^b)`). `O(N)` field ops.
fn divide_by_binomial<F: Field>(
    coeffs: &[F],
    b: usize,
    b_row: usize,
    alpha: F,
) -> (Vec<F>, Vec<F>) {
    let mut g = vec![F::zero(); b];
    let q_len = b * b_row.saturating_sub(1);
    let mut q = vec![F::zero(); q_len];
    for i in 0..b {
        // column polynomial f_i(X) = sum_k coeffs[i + k*b] X^k, degree < b_row
        let mut buf: Vec<F> = (0..b_row).map(|k| coeffs[i + k * b]).collect();
        for k in (0..b_row.saturating_sub(1)).rev() {
            let hi = buf[k + 1];
            buf[k] += alpha * hi;
        }
        g[i] = buf[0];
        for k in 0..b_row.saturating_sub(1) {
            q[i + k * b] = buf[k + 1];
        }
    }
    (g, q)
}

/// `D(X) = X^{b-1} g(1/X)` = coefficient reversal of `g` (padded to `b`).
fn reverse_coeffs<F: Field>(g: &[F], b: usize) -> Vec<F> {
    let mut d = vec![F::zero(); b];
    for (i, &c) in g.iter().enumerate().take(b) {
        d[b - 1 - i] = c;
    }
    d
}

/// FFT-free structured computation of `S(X)` (Mercury §4.1) from the tensor
/// structure of `P_{u1}, P_{u2}`. `S(X) = sum_{k=1}^{b-1} A_k X^{k-1}` where
/// `A_k` is the coefficient of `X^k` in
///   g(X)P_{u1}(1/X)+g(1/X)P_{u1}(X)+gamma(h(X)P_{u2}(1/X)+h(1/X)P_{u2}(X)).
/// Cost `O(b t) = O(sqrt N log N)`, no FFT.
///
/// `g,h` must have length `b`; `u1,u2` must have length `t` (pad with 0).
fn make_s_polynomial_structured<F: Field>(
    g: &[F],
    h: &[F],
    u1: &[F],
    u2: &[F],
    t: usize,
    b: usize,
    gamma: F,
) -> Vec<F> {
    // C1(X) = g(X) P_{u1}(1/X) as a Laurent buffer (offset = b-1).
    let c1 = mul_by_reciprocal_tensor(g, t, u1);
    let c2 = mul_by_reciprocal_tensor(h, t, u2);
    let off = laurent_offset(t);
    debug_assert_eq!(off, b - 1);
    let mut s = vec![F::zero(); b - 1];
    for (k1, sk) in s.iter_mut().enumerate() {
        let k = k1 + 1;
        let a1 = c1[off + k] + c1[off - k];
        let a2 = c2[off + k] + c2[off - k];
        *sk = a1 + gamma * a2;
    }
    s
}

// ════════════════════════════════════════════════════════════════════
// Prover core
// ════════════════════════════════════════════════════════════════════

/// Reject `zeta = 0` and `zeta^2 = 1`, returning `zeta^{-1}`.
fn validate_zeta<F: Field>(zeta: F) -> Result<F, PCSError> {
    if zeta.is_zero() {
        return Err(PCSError::InvalidProof("zeta = 0".to_string()));
    }
    if zeta * zeta == F::one() {
        return Err(PCSError::InvalidProof(
            "zeta^2 = 1 (zeta = 1/zeta): reciprocal points collide".to_string(),
        ));
    }
    zeta.inverse()
        .ok_or_else(|| PCSError::InvalidProof("zeta inverse failed".to_string()))
}

/// Reject BDFG20 point `z` colliding with any interpolation node.
fn validate_zbdfg<F: Field>(z: F, zeta: F, zeta_inv: F, alpha: F) -> Result<(), PCSError> {
    if z == zeta || z == zeta_inv || z == alpha {
        return Err(PCSError::InvalidProof(
            "BDFG20 point collides with an evaluation node".to_string(),
        ));
    }
    Ok(())
}

/// Commit to two independent polynomials, in parallel under the `parallel`
/// feature (following Nova's `rayon::join` of `(comm_q, comm_g)` and
/// `(comm_s, comm_d)`). The coefficient slices are borrowed, never copied. Each
/// commitment keeps its own profiling phase / real MSM `count`; transcript
/// absorption stays sequential at the call site, so the Fiat-Shamir order is
/// unaffected.
#[allow(clippy::too_many_arguments)]
fn commit_two<E: Pairing>(
    pp: &MercuryProverParam<E>,
    a: &[E::ScalarField],
    phase_a: &'static str,
    note_a: &'static str,
    b: &[E::ScalarField],
    phase_b: &'static str,
    note_b: &'static str,
    mu: usize,
    n: usize,
) -> Result<(E::G1Affine, E::G1Affine), PCSError> {
    let commit_a = || {
        let _t = ScopedTimer::new(BACKEND, mu, n, phase_a, a.len(), note_a);
        pp.commit(a)
    };
    let commit_b = || {
        let _t = ScopedTimer::new(BACKEND, mu, n, phase_b, b.len(), note_b);
        pp.commit(b)
    };
    #[cfg(feature = "parallel")]
    {
        let (ra, rb) = rayon::join(commit_a, commit_b);
        Ok((ra?, rb?))
    }
    #[cfg(not(feature = "parallel"))]
    {
        Ok((commit_a()?, commit_b()?))
    }
}

#[allow(clippy::too_many_arguments)]
fn mercury_core_open<E: Pairing>(
    pp: &MercuryProverParam<E>,
    coeffs: &[E::ScalarField],
    mu: usize,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<MercuryProof<E>, PCSError> {
    let (t, b, b_row, n) = mercury_dims(mu)?;
    if pp.max_degree < n - 1 {
        return Err(PCSError::InvalidParameters(format!(
            "prover key max_degree {} insufficient for N-1 = {}",
            pp.max_degree,
            n - 1
        )));
    }
    let _t_total = ScopedTimer::new(BACKEND, mu, n, "mercury_open_total", 1, "total");

    // point split: u1 = low t vars (columns), u2 = high vars (rows).
    let u1 = &point[..t];
    let u2 = &point[t..];
    // u2 padded to length t (trailing 0) for the tensor S helper.
    let mut u2_full: Vec<E::ScalarField> = Vec::with_capacity(t);
    u2_full.extend_from_slice(u2);
    u2_full.resize(t, E::ScalarField::zero());

    // eq vectors.
    let _t_eq = ScopedTimer::new(BACKEND, mu, n, "mercury_open_build_eq_vectors", b, "eq");
    let eq_col = build_eq_vec::<E::ScalarField>(u1, b);
    drop(_t_eq);

    // ── Step 1: restriction polynomial h ──
    let _t_h = ScopedTimer::new(BACKEND, mu, n, "mercury_open_compute_h", n, "h");
    let h_poly = compute_h(coeffs, &eq_col, b_row, b);
    drop(_t_h);
    // Defence in depth: the restriction IPA value hhat(u2) must equal the value
    // bound into the transcript. These are provably equal for correct inputs; a
    // mismatch (a caller passing an inconsistent `value`) is rejected rather
    // than producing an unverifiable proof.
    {
        let eq_row = build_eq_vec::<E::ScalarField>(u2, b_row);
        let mut v_check = E::ScalarField::zero();
        for (j, &e) in eq_row.iter().enumerate().take(b_row) {
            v_check += e * h_poly[j];
        }
        if &v_check != value {
            return Err(PCSError::InvalidParameters(
                "claimed value inconsistent with committed polynomial".to_string(),
            ));
        }
    }
    let _t_ch = ScopedTimer::new(BACKEND, mu, n, "mercury_open_commit_h", b, "KZG-h");
    let comm_h = pp.commit(&h_poly)?;
    drop(_t_ch);
    transcript.append_serializable_element(L_H, &comm_h)?;

    // ── Step 2: fold via univariate division ──
    let alpha = transcript.get_and_append_challenge(L_ALPHA)?;
    let _t_div = ScopedTimer::new(BACKEND, mu, n, "mercury_open_divide_by_binomial", n, "div");
    let (g_poly, q_poly) = divide_by_binomial(&coeffs[..n], b, b_row, alpha);
    drop(_t_div);
    // comm_q and comm_g are independent MSMs -> committed in parallel.
    let (comm_q, comm_g) = commit_two(
        pp,
        &q_poly,
        "mercury_open_commit_q",
        "KZG-q",
        &g_poly,
        "mercury_open_commit_g",
        "KZG-g",
        mu,
        n,
    )?;
    transcript.append_serializable_element(L_Q, &comm_q)?;
    transcript.append_serializable_element(L_G, &comm_g)?;

    // ── Step 3: Lagrange IPA witness S and degree-check D ──
    let gamma = transcript.get_and_append_challenge(L_GAMMA)?;
    let _t_ipa = ScopedTimer::new(BACKEND, mu, n, "mercury_open_build_lagrange_ipa", b, "ipa");
    let h_alpha = poly_eval(&h_poly, alpha);
    drop(_t_ipa);
    let _t_s = ScopedTimer::new(BACKEND, mu, n, "mercury_open_compute_s", b, "S");
    let s_poly = make_s_polynomial_structured(&g_poly, &h_poly, u1, &u2_full, t, b, gamma);
    drop(_t_s);
    let d_poly = reverse_coeffs(&g_poly, b);
    // comm_s and comm_d are independent MSMs -> committed in parallel.
    let (comm_s, comm_d) = commit_two(
        pp,
        &s_poly,
        "mercury_open_commit_s",
        "KZG-s",
        &d_poly,
        "mercury_open_commit_d",
        "KZG-d",
        mu,
        n,
    )?;
    transcript.append_serializable_element(L_S, &comm_s)?;
    transcript.append_serializable_element(L_D, &comm_d)?;

    // ── Step 4: KZG evaluations ──
    let zeta = transcript.get_and_append_challenge(L_ZETA)?;
    let zeta_inv = validate_zeta(zeta)?;
    let _t_ev = ScopedTimer::new(BACKEND, mu, n, "mercury_open_eval_claims", 8, "evals");
    let g_zeta = poly_eval(&g_poly, zeta);
    let g_zeta_inv = poly_eval(&g_poly, zeta_inv);
    let h_zeta = poly_eval(&h_poly, zeta);
    let h_zeta_inv = poly_eval(&h_poly, zeta_inv);
    let s_zeta = poly_eval(&s_poly, zeta);
    let s_zeta_inv = poly_eval(&s_poly, zeta_inv);
    let d_zeta = poly_eval(&d_poly, zeta);
    drop(_t_ev);

    // folding KZG proof: quot_f = (f - (zeta^b - alpha) q - g_zeta) / (X - zeta)
    let _t_bqf = ScopedTimer::new(BACKEND, mu, n, "mercury_open_build_quot_f", n, "quot_f");
    let zeta_b = zeta.pow([b as u64]);
    let zeta_b_alpha = zeta_b - alpha;
    let mut num_f = coeffs[..n].to_vec();
    add_scaled(&mut num_f, &q_poly, -zeta_b_alpha);
    num_f[0] -= g_zeta;
    let (quot_f, rem_f) = divide_by_linear(&num_f, zeta);
    if !rem_f.is_zero() {
        return Err(PCSError::InvalidProver(
            "folding quotient has nonzero remainder".to_string(),
        ));
    }
    drop(_t_bqf);
    let _t_cqf = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "mercury_open_commit_quot_f",
        quot_f.len(),
        "KZG-quot_f",
    );
    let comm_quot_f = pp.commit(&quot_f)?;
    drop(_t_cqf);

    transcript.append_field_element(L_GZ, &g_zeta)?;
    transcript.append_field_element(L_GZI, &g_zeta_inv)?;
    transcript.append_field_element(L_HZ, &h_zeta)?;
    transcript.append_field_element(L_HZI, &h_zeta_inv)?;
    transcript.append_field_element(L_SZ, &s_zeta)?;
    transcript.append_field_element(L_SZI, &s_zeta_inv)?;
    transcript.append_serializable_element(L_QUOTF, &comm_quot_f)?;

    // ── BDFG20 batched opening of {g, h, S, D} at {zeta, zeta_inv, alpha} ──
    let bdfg = BdfgProverInput {
        g: &g_poly,
        h: &h_poly,
        s: &s_poly,
        d: &d_poly,
        g_zeta,
        g_zeta_inv,
        h_zeta,
        h_zeta_inv,
        h_alpha,
        s_zeta,
        s_zeta_inv,
        d_zeta,
        zeta,
        zeta_inv,
        alpha,
    };
    let (comm_w, comm_w_prime) = bdfg_prove(pp, &bdfg, mu, n, transcript)?;

    // final pairing batch challenge (bound but consumed only for verifier match)
    transcript.append_serializable_element(L_WP, &comm_w_prime)?;
    let _d_pair = transcript.get_and_append_challenge(L_DPAIR)?;

    Ok(MercuryProof {
        comm_h,
        comm_g,
        comm_q,
        comm_s,
        comm_d,
        comm_quot_f,
        comm_w,
        comm_w_prime,
        g_zeta,
        g_zeta_inv,
        h_zeta,
        h_zeta_inv,
        s_zeta,
        s_zeta_inv,
        mu,
    })
}

/// `eq(i, u)` vector of length `b` (little-endian: variable `u[k]` is bit `k`).
/// Requires `b >= 2^{u.len()}`. Avoids `build_eq_x_r_vec` (which errors on
/// empty `u`) so `mu = 1` works.
fn build_eq_vec<F: Field>(u: &[F], b: usize) -> Vec<F> {
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

// ════════════════════════════════════════════════════════════════════
// BDFG20 batched multi-point KZG opening (ePrint 2020/081, §4)
// ════════════════════════════════════════════════════════════════════

struct BdfgProverInput<'a, F: Field> {
    g: &'a [F],
    h: &'a [F],
    s: &'a [F],
    d: &'a [F],
    g_zeta: F,
    g_zeta_inv: F,
    h_zeta: F,
    h_zeta_inv: F,
    h_alpha: F,
    s_zeta: F,
    s_zeta_inv: F,
    d_zeta: F,
    zeta: F,
    zeta_inv: F,
    alpha: F,
}

/// Compute the four interpolants `g*, h*, s*, d*` through their eval sets.
#[allow(clippy::type_complexity)]
fn bdfg_star_polys<F: Field>(
    zeta: F,
    zeta_inv: F,
    alpha: F,
    g_zeta: F,
    g_zeta_inv: F,
    h_zeta: F,
    h_zeta_inv: F,
    h_alpha: F,
    s_zeta: F,
    s_zeta_inv: F,
    d_zeta: F,
) -> Result<(Vec<F>, Vec<F>, Vec<F>, Vec<F>), PCSError> {
    // interpolation nodes must be pairwise distinct.
    if zeta == alpha || zeta_inv == alpha {
        return Err(PCSError::InvalidProof(
            "alpha collides with a reciprocal node".to_string(),
        ));
    }
    let g_star = lagrange_interpolate(&[zeta, zeta_inv], &[g_zeta, g_zeta_inv])?;
    let h_star = lagrange_interpolate(&[zeta, zeta_inv, alpha], &[h_zeta, h_zeta_inv, h_alpha])?;
    let s_star = lagrange_interpolate(&[zeta, zeta_inv], &[s_zeta, s_zeta_inv])?;
    let d_star = vec![d_zeta];
    Ok((g_star, h_star, s_star, d_star))
}

/// Pure BDFG20 first-round polynomials: the star interpolants, the combined
/// numerator `m(X)`, and the first witness `W(X) = m(X) / Z_T(X)` with
/// `Z_T(X) = (X-alpha)(X-zeta)(X-zeta_inv)`. No transcript / no commitment, so
/// this is unit-testable in isolation (see `tests::bdfg_*`).
///
/// `m(X) = Z_{T\S_g}(X)(g-g*) + beta Z_{T\S_h}(X)(h-h*)
///         + beta^2 Z_{T\S_s}(X)(s-s*) + beta^3 Z_{T\S_d}(X)(d-d*)`
/// with `Z_{T\S_g}=(X-alpha)`, `Z_{T\S_h}=1`, `Z_{T\S_s}=(X-alpha)`,
/// `Z_{T\S_d}=(X-alpha)(X-zeta_inv)`.
struct BdfgMPolys<F: Field> {
    g_star: Vec<F>,
    h_star: Vec<F>,
    s_star: Vec<F>,
    d_star: Vec<F>,
    /// The combined numerator `m(X)`. Consumed by the coefficient-level tests
    /// (`m == Z_T * W`); the prover only needs `quot_m`.
    #[allow(dead_code)]
    m: Vec<F>,
    quot_m: Vec<F>,
}

fn bdfg_build_m<F: Field>(
    inp: &BdfgProverInput<'_, F>,
    beta: F,
) -> Result<BdfgMPolys<F>, PCSError> {
    let beta2 = beta * beta;
    let beta3 = beta2 * beta;

    let (g_star, h_star, s_star, d_star) = bdfg_star_polys(
        inp.zeta,
        inp.zeta_inv,
        inp.alpha,
        inp.g_zeta,
        inp.g_zeta_inv,
        inp.h_zeta,
        inp.h_zeta_inv,
        inp.h_alpha,
        inp.s_zeta,
        inp.s_zeta_inv,
        inp.d_zeta,
    )?;

    let diff_g = poly_sub(inp.g, &g_star);
    let diff_h = poly_sub(inp.h, &h_star);
    let diff_s = poly_sub(inp.s, &s_star);
    let diff_d = poly_sub(inp.d, &d_star);

    let term_g = mul_by_linear(&diff_g, inp.alpha);
    let term_h = diff_h;
    let term_s = mul_by_linear(&diff_s, inp.alpha);
    let term_d = {
        let tmp = mul_by_linear(&diff_d, inp.alpha);
        mul_by_linear(&tmp, inp.zeta_inv)
    };

    let mut m: Vec<F> = Vec::new();
    add_scaled(&mut m, &term_g, F::one());
    add_scaled(&mut m, &term_h, beta);
    add_scaled(&mut m, &term_s, beta2);
    add_scaled(&mut m, &term_d, beta3);

    // quot_m = m / (X-alpha)(X-zeta)(X-zeta_inv)
    let (m1, r1) = divide_by_linear(&m, inp.alpha);
    let (m2, r2) = divide_by_linear(&m1, inp.zeta);
    let (quot_m, r3) = divide_by_linear(&m2, inp.zeta_inv);
    if !r1.is_zero() || !r2.is_zero() || !r3.is_zero() {
        return Err(PCSError::InvalidProver(
            "BDFG20 m(X) not divisible by Z_T".to_string(),
        ));
    }
    Ok(BdfgMPolys {
        g_star,
        h_star,
        s_star,
        d_star,
        m,
        quot_m,
    })
}

/// Pure BDFG20 second-round polynomials: `L(X) = m_z(X) - Z_T(z) W(X)` and the
/// second witness `W'(X) = L(X)/(X-z)`. No transcript / no commitment.
struct BdfgLPolys<F: Field> {
    /// `L(X)`. Consumed by the coefficient-level tests (`L == (X-z) * W'`); the
    /// prover only needs `quot_l`.
    #[allow(dead_code)]
    l: Vec<F>,
    quot_l: Vec<F>,
}

fn bdfg_build_l<F: Field>(
    inp: &BdfgProverInput<'_, F>,
    mpolys: &BdfgMPolys<F>,
    beta: F,
    z: F,
) -> Result<BdfgLPolys<F>, PCSError> {
    let beta2 = beta * beta;
    let beta3 = beta2 * beta;
    let zg_z = z - inp.alpha;
    let zh_z = F::one();
    let zs_z = z - inp.alpha;
    let zd_z = (z - inp.zeta_inv) * (z - inp.alpha);
    let zt_z = zd_z * (z - inp.zeta);

    let g_star_z = poly_eval(&mpolys.g_star, z);
    let h_star_z = poly_eval(&mpolys.h_star, z);
    let s_star_z = poly_eval(&mpolys.s_star, z);
    let d_star_z = poly_eval(&mpolys.d_star, z);

    // m_z(X) = Zg(z)(g - g*(z)) + beta Zh(z)(h - h*(z)) + beta^2 Zs(z)(s - s*(z))
    //          + beta^3 Zd(z)(d - d*(z))  (built directly into `l`).
    let mut l: Vec<F> = Vec::new();
    add_scaled(&mut l, inp.g, zg_z);
    subtract_const(&mut l, zg_z * g_star_z);
    add_scaled(&mut l, inp.h, beta * zh_z);
    subtract_const(&mut l, beta * zh_z * h_star_z);
    add_scaled(&mut l, inp.s, beta2 * zs_z);
    subtract_const(&mut l, beta2 * zs_z * s_star_z);
    add_scaled(&mut l, inp.d, beta3 * zd_z);
    subtract_const(&mut l, beta3 * zd_z * d_star_z);

    // L(X) = m_z(X) - Z_T(z) quot_m(X)
    add_scaled(&mut l, &mpolys.quot_m, -zt_z);
    let (quot_l, rl) = divide_by_linear(&l, z);
    if !rl.is_zero() {
        return Err(PCSError::InvalidProver(
            "BDFG20 L(X) not divisible by (X - z)".to_string(),
        ));
    }
    Ok(BdfgLPolys { l, quot_l })
}

fn bdfg_prove<E: Pairing>(
    pp: &MercuryProverParam<E>,
    inp: &BdfgProverInput<'_, E::ScalarField>,
    mu: usize,
    n: usize,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<(E::G1Affine, E::G1Affine), PCSError> {
    let beta = transcript.get_and_append_challenge(L_BETA)?;

    let _t_bq = ScopedTimer::new(BACKEND, mu, n, "mercury_open_build_batch_quotient", 1, "m");
    let mpolys = bdfg_build_m(inp, beta)?;
    drop(_t_bq);

    let _t_cw = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "mercury_open_commit_batch_w",
        mpolys.quot_m.len(),
        "KZG-w",
    );
    let comm_w = pp.commit(&mpolys.quot_m)?;
    drop(_t_cw);
    transcript.append_serializable_element(L_W, &comm_w)?;

    let z = transcript.get_and_append_challenge(L_ZBDFG)?;
    validate_zbdfg(z, inp.zeta, inp.zeta_inv, inp.alpha)?;

    let _t_bf = ScopedTimer::new(BACKEND, mu, n, "mercury_open_build_final_quotient", 1, "L");
    let lpolys = bdfg_build_l(inp, &mpolys, beta, z)?;
    drop(_t_bf);

    let _t_cwp = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "mercury_open_commit_batch_w_prime",
        lpolys.quot_l.len(),
        "KZG-wp",
    );
    let comm_w_prime = pp.commit(&lpolys.quot_l)?;
    drop(_t_cwp);

    Ok((comm_w, comm_w_prime))
}

// poly_sub, subtract_const re-exported from crate::pcs::bdfg

// ════════════════════════════════════════════════════════════════════
// Verifier core
// ════════════════════════════════════════════════════════════════════

fn mercury_core_verify<E: Pairing>(
    vp: &MercuryVerifierParam<E>,
    com: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &MercuryProof<E>,
) -> Result<bool, PCSError> {
    let mu = proof.mu;
    // ── untrusted-input integrity (before any shift / alloc) ──
    let (t, b, _b_row, n) = mercury_dims_verify(mu)?;
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
    let _t_total = ScopedTimer::new(BACKEND, mu, n, "mercury_verify_total", 1, "total");

    // ── replay Fiat-Shamir ──
    let _t_tr = ScopedTimer::new(BACKEND, mu, n, "mercury_verify_transcript", 1, "fs");
    let mut tr = new_transcript::<E>(mu, com, point, value)?;
    tr.append_serializable_element(L_H, &proof.comm_h)?;
    let alpha = tr.get_and_append_challenge(L_ALPHA)?;
    tr.append_serializable_element(L_Q, &proof.comm_q)?;
    tr.append_serializable_element(L_G, &proof.comm_g)?;
    let gamma = tr.get_and_append_challenge(L_GAMMA)?;
    tr.append_serializable_element(L_S, &proof.comm_s)?;
    tr.append_serializable_element(L_D, &proof.comm_d)?;
    let zeta = tr.get_and_append_challenge(L_ZETA)?;
    let zeta_inv = validate_zeta(zeta)?;
    tr.append_field_element(L_GZ, &proof.g_zeta)?;
    tr.append_field_element(L_GZI, &proof.g_zeta_inv)?;
    tr.append_field_element(L_HZ, &proof.h_zeta)?;
    tr.append_field_element(L_HZI, &proof.h_zeta_inv)?;
    tr.append_field_element(L_SZ, &proof.s_zeta)?;
    tr.append_field_element(L_SZI, &proof.s_zeta_inv)?;
    tr.append_serializable_element(L_QUOTF, &proof.comm_quot_f)?;
    drop(_t_tr);

    // ── reconstruct d_zeta (degree check) and h_alpha (Lagrange IPA) ──
    let u1 = &point[..t];
    let u2 = &point[t..];
    let pu1_zeta = pu_eval(u1, zeta);
    let pu1_zeta_inv = pu_eval(u1, zeta_inv);
    let pu2_zeta = pu_eval(u2, zeta);
    let pu2_zeta_inv = pu_eval(u2, zeta_inv);

    let zeta_b_minus_1 = zeta.pow([(b - 1) as u64]);
    let d_zeta = zeta_b_minus_1 * proof.g_zeta_inv;

    let two = E::ScalarField::one() + E::ScalarField::one();
    let two_inv = two
        .inverse()
        .ok_or_else(|| PCSError::InvalidProof("char 2 field unsupported".to_string()))?;
    let h_alpha = {
        let lhs = proof.g_zeta * pu1_zeta_inv
            + proof.g_zeta_inv * pu1_zeta
            + gamma * (proof.h_zeta * pu2_zeta_inv + proof.h_zeta_inv * pu2_zeta - value.double())
            - zeta * proof.s_zeta
            - zeta_inv * proof.s_zeta_inv;
        lhs * two_inv
    };

    // ── folding pairing statement (lhs_1_1, lhs_2_1) ──
    let _t_msm = ScopedTimer::new(BACKEND, mu, n, "mercury_verify_g1_msm", 3, "MSM-fold");
    let zeta_b = zeta_b_minus_1 * zeta;
    let zeta_b_alpha = zeta_b - alpha;
    let fold_scalars = [-zeta_b_alpha, -proof.g_zeta, zeta];
    let fold_bases = [proof.comm_q, vp.g1_one, proof.comm_quot_f];
    let lhs_1_1 = com.0.into_group() + E::G1::msm_unchecked(&fold_bases, &fold_scalars);
    let lhs_2_1 = proof.comm_quot_f.into_group();
    drop(_t_msm);

    // ── BDFG20 verifier reconstruction (lhs_1_2, lhs_2_2) ──
    let (lhs_1_2, lhs_2_2) = bdfg_verify_lhs(
        vp,
        proof,
        BdfgVerifyEvals {
            zeta,
            zeta_inv,
            alpha,
            g_zeta: proof.g_zeta,
            g_zeta_inv: proof.g_zeta_inv,
            h_zeta: proof.h_zeta,
            h_zeta_inv: proof.h_zeta_inv,
            h_alpha,
            s_zeta: proof.s_zeta,
            s_zeta_inv: proof.s_zeta_inv,
            d_zeta,
        },
        mu,
        n,
        &mut tr,
    )?;

    // final pairing batch challenge.
    tr.append_serializable_element(L_WP, &proof.comm_w_prime)?;
    let d_pair = tr.get_and_append_challenge(L_DPAIR)?;

    let ll = (lhs_1_1 + lhs_1_2 * d_pair).into_affine();
    // Negate the right side so the product form
    //   e(ll, [1]_2) * e(-rl, [tau]_2) = 1_{G_T}
    // is equivalent to the two-pairing equality e(ll, [1]_2) == e(rl, [tau]_2).
    // This is the same multi_pairing idiom used by the ReciPCS / Gemini /
    // Zeromorph / Samaritan / NestedGridKZG verifiers in this repo: one Miller
    // loop + one final exponentiation instead of two separate pairings.
    let neg_rl = (-(lhs_2_1 + lhs_2_2 * d_pair)).into_affine();

    let _t_mp = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "mercury_verify_multi_pairing",
        2,
        "2-term-product",
    );
    let ok = E::multi_pairing([ll, neg_rl], [vp.g2_one, vp.g2_tau])
        == PairingOutput(E::TargetField::one());
    drop(_t_mp);
    Ok(ok)
}

struct BdfgVerifyEvals<F: Field> {
    zeta: F,
    zeta_inv: F,
    alpha: F,
    g_zeta: F,
    g_zeta_inv: F,
    h_zeta: F,
    h_zeta_inv: F,
    h_alpha: F,
    s_zeta: F,
    s_zeta_inv: F,
    d_zeta: F,
}

/// Reconstruct the two group elements of the BDFG20 pairing statement.
/// `lhs_1_2 = F + z * comm_w_prime`, `lhs_2_2 = comm_w_prime` (§4.1, item 6).
fn bdfg_verify_lhs<E: Pairing>(
    vp: &MercuryVerifierParam<E>,
    proof: &MercuryProof<E>,
    ev: BdfgVerifyEvals<E::ScalarField>,
    mu: usize,
    n: usize,
    tr: &mut IOPTranscript<E::ScalarField>,
) -> Result<(E::G1, E::G1), PCSError> {
    let beta = tr.get_and_append_challenge(L_BETA)?;
    tr.append_serializable_element(L_W, &proof.comm_w)?;
    let z = tr.get_and_append_challenge(L_ZBDFG)?;
    validate_zbdfg(z, ev.zeta, ev.zeta_inv, ev.alpha)?;
    bdfg_verify_lhs_pure(vp, proof, &ev, beta, z, mu, n)
}

/// Pure homomorphic reconstruction of the BDFG20 pairing group elements with
/// the challenges `beta, z` given explicitly (no transcript). Algebraically
/// `lhs_1_2 = [tau * W'(tau)]_1` and `lhs_2_2 = [W'(tau)]_1`, so the pairing
/// check `e(lhs_1_2,[1]_2) = e(lhs_2_2,[tau]_2)` encodes `L(X) = (X-z)W'(X)`.
/// Unit-tested directly against commitments in `tests::bdfg_*`.
fn bdfg_verify_lhs_pure<E: Pairing>(
    vp: &MercuryVerifierParam<E>,
    proof: &MercuryProof<E>,
    ev: &BdfgVerifyEvals<E::ScalarField>,
    beta: E::ScalarField,
    z: E::ScalarField,
    mu: usize,
    n: usize,
) -> Result<(E::G1, E::G1), PCSError> {
    let (g_star, h_star, s_star, d_star) = bdfg_star_polys(
        ev.zeta,
        ev.zeta_inv,
        ev.alpha,
        ev.g_zeta,
        ev.g_zeta_inv,
        ev.h_zeta,
        ev.h_zeta_inv,
        ev.h_alpha,
        ev.s_zeta,
        ev.s_zeta_inv,
        ev.d_zeta,
    )?;
    let g_star_z = poly_eval(&g_star, z);
    let h_star_z = poly_eval(&h_star, z);
    let s_star_z = poly_eval(&s_star, z);
    let d_star_z = poly_eval(&d_star, z);

    let beta2 = beta * beta;
    let beta3 = beta2 * beta;
    let zg_z = z - ev.alpha;
    let zh_z = E::ScalarField::one();
    let zs_z = z - ev.alpha;
    let zd_z = (z - ev.zeta_inv) * (z - ev.alpha);
    let zt_z = zd_z * (z - ev.zeta);

    let scalar = zg_z * g_star_z
        + beta * zh_z * h_star_z
        + beta2 * zs_z * s_star_z
        + beta3 * zd_z * d_star_z;

    let _t_msm = ScopedTimer::new(BACKEND, mu, n, "mercury_verify_g1_msm", 7, "MSM-batch");
    let scalars = [
        zg_z,
        beta * zh_z,
        beta2 * zs_z,
        beta3 * zd_z,
        -scalar,
        -zt_z,
        z,
    ];
    let bases = [
        proof.comm_g,
        proof.comm_h,
        proof.comm_s,
        proof.comm_d,
        vp.g1_one,
        proof.comm_w,
        proof.comm_w_prime,
    ];
    let lhs_1_2 = E::G1::msm_unchecked(&bases, &scalars);
    drop(_t_msm);
    let lhs_2_2 = proof.comm_w_prime.into_group();
    Ok((lhs_1_2, lhs_2_2))
}

#[cfg(test)]
mod tests;
