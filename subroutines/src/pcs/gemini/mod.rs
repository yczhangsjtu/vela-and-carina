//! Gemini PCS — multilinear PCS based on KZG univariate commitments.
//!
//! Converts multilinear polynomial evaluations to univariate coefficient form,
//! then commits using KZG G1 MSM. The opening protocol uses the Gemini
//! transformation to reduce multilinear openings to univariate KZG proofs.
//!
//! The single-open protocol uses Shplonk batching to achieve O(1) KZG proof
//! elements. Fold claims are accumulated into a single batched quotient whose
//! evaluation at a random point z must be zero; a single KZG opening proof
//! witnesses that condition.

pub(crate) mod srs;

use crate::{
    pcs::{
        multilinear_kzg::batching::BatchProof,
        prelude::{Commitment, PCSError},
        profile::ScopedTimer,
        PolynomialCommitmentScheme, StructuredReferenceString,
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

use srs::{GeminiProverParam, GeminiUniversalParams, GeminiVerifierParam};

const BACKEND: &str = "Gemini";

// ═══════════════════════════════════════════════════════════════════
// Proof / PCS structures
// ═══════════════════════════════════════════════════════════════════

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct GeminiProof<E: Pairing> {
    pub fold_comms: Vec<E::G1Affine>,
    pub fold_evals: Vec<(E::ScalarField, E::ScalarField)>,
    pub final_coeffs: (E::ScalarField, E::ScalarField),
    pub shplonk_q_commit: E::G1Affine,
    pub kzg_witness: E::G1Affine,
    pub mu: usize,
}

pub struct GeminiPCS<E: Pairing> {
    phantom: PhantomData<E>,
}

// ═══════════════════════════════════════════════════════════════════
// PolynomialCommitmentScheme impl
// ═══════════════════════════════════════════════════════════════════

impl<E: Pairing> PolynomialCommitmentScheme<E> for GeminiPCS<E> {
    type ProverParam = GeminiProverParam<E>;
    type VerifierParam = GeminiVerifierParam<E>;
    type SRS = GeminiUniversalParams<E>;
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    type Commitment = Commitment<E>;
    type Proof = GeminiProof<E>;
    type BatchProof = BatchProof<E, Self>;

    fn gen_srs_for_testing<R: Rng>(rng: &mut R, s: usize) -> Result<Self::SRS, PCSError> {
        GeminiUniversalParams::<E>::gen_srs_for_testing(rng, s)
    }

    fn trim(
        srs: impl Borrow<Self::SRS>,
        _d: Option<usize>,
        nv: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        let nv = nv.ok_or_else(|| PCSError::InvalidParameters("need num_var".to_string()))?;
        let n = checked_domain_size_from_mu(nv, "trim")
            .map_err(|e| PCSError::InvalidParameters(e.to_string()))?;
        srs.borrow().trim(n)
    }

    fn commit(
        pp: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let pp = pp.borrow();
        let nv = poly.num_vars;
        let n = checked_domain_size_from_mu(nv, "commit")
            .map_err(|e| PCSError::InvalidParameters(e.to_string()))?;
        if pp.max_degree < n {
            return Err(PCSError::InvalidParameters(format!(
                "degree {} > max {}",
                n, pp.max_degree
            )));
        }
        let _t = ScopedTimer::new(BACKEND, nv, n, "commit_to_evals", n, "to_evaluations");
        let coeffs = poly.to_evaluations();
        drop(_t);
        let _t = ScopedTimer::new(BACKEND, nv, n, "commit_msm", coeffs.len(), "KZG-MSM");
        let cm = pp.try_commit(&coeffs)?;
        drop(_t);
        Ok(Commitment(cm))
    }

    fn open(
        pp: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let mu = poly.num_vars();
        let mut transcript = IOPTranscript::new(b"gemini-open");
        transcript.append_field_element(b"mu", &E::ScalarField::from(mu as u64))?;
        gemini_open_with_transcript(pp.borrow(), poly, point, &mut transcript)
    }

    fn multi_open(
        pp: impl Borrow<Self::ProverParam>,
        polys: &[Self::Polynomial],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Self::BatchProof, PCSError> {
        gemini_sumcheck_multi_open(pp.borrow(), polys, points, evals, transcript)
    }

    fn verify(
        vp: &Self::VerifierParam,
        com: &Self::Commitment,
        point: &Self::Point,
        val: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        let mut transcript = IOPTranscript::new(b"gemini-open");
        transcript.append_field_element(b"mu", &E::ScalarField::from(proof.mu as u64))?;
        gemini_verify_with_transcript(vp, com, point, val, proof, &mut transcript)
    }

    fn batch_verify(
        vp: &Self::VerifierParam,
        coms: &[Self::Commitment],
        points: &[Self::Point],
        bp: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        gemini_sumcheck_batch_verify(vp, coms, points, bp, transcript)
    }
}

impl<E: Pairing> GeminiPCS<E> {
    /// Open a polynomial at a point given a pre-computed commitment `cm_f`.
    ///
    /// This avoids the N-size MSM recommit that the trait `open` performs
    /// (via `gemini_open_with_transcript`).  `commitment` MUST equal
    /// `commit(pp, poly)`.
    pub fn open_with_commitment(
        pp: &GeminiProverParam<E>,
        poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
        point: &[E::ScalarField],
        commitment: &Commitment<E>,
    ) -> Result<(GeminiProof<E>, E::ScalarField), PCSError> {
        let mu = poly.num_vars();
        let _n = checked_domain_size_from_mu(mu, "open_with_commitment")
            .map_err(|e| PCSError::InvalidParameters(e.to_string()))?;
        if point.len() != mu {
            return Err(PCSError::InvalidParameters(
                "point length mismatch".to_string(),
            ));
        }
        let mut transcript = IOPTranscript::new(b"gemini-open");
        transcript.append_field_element(b"mu", &E::ScalarField::from(mu as u64))?;

        let f_hat = poly.to_evaluations();
        let eval = poly
            .evaluate(point)
            .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;

        transcript.append_serializable_element(b"commitment", &commitment.0)?;
        transcript.append_serializable_element(b"point", &point.to_vec())?;
        transcript.append_field_element(b"eval", &eval)?;

        gemini_core_open_prebound(pp, poly, point, f_hat, eval, &mut transcript, commitment.0)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Verifier safety helpers
// ═══════════════════════════════════════════════════════════════════

fn checked_domain_size_from_mu(mu: usize, label: &str) -> Result<usize, PCSError> {
    if mu == 0 {
        return Err(PCSError::InvalidProof(format!("{label}: mu is zero")));
    }
    if mu >= usize::BITS as usize {
        return Err(PCSError::InvalidProof(format!(
            "{label}: mu {mu} exceeds platform word size"
        )));
    }
    1usize
        .checked_shl(mu as u32)
        .ok_or_else(|| PCSError::InvalidProof(format!("{label}: mu {mu} overflow in shift")))
}

// ═══════════════════════════════════════════════════════════════════
// Polynomial helpers
// ═══════════════════════════════════════════════════════════════════

fn poly_eval<F: Field>(coeffs: &[F], x: F) -> F {
    let mut result = F::zero();
    for c in coeffs.iter().rev() {
        result = result * x + *c;
    }
    result
}

fn fold_polynomial<F: Field>(coeffs: &[F], u: F) -> Vec<F> {
    let len = coeffs.len();
    let half = len / 2;
    let mut result = vec![F::zero(); half];
    for i in 0..half {
        result[i] = (F::one() - u) * coeffs[2 * i] + u * coeffs[2 * i + 1];
    }
    result
}

fn kzg_quotient<F: Field>(coeffs: &[F], point: F, value: F) -> Vec<F> {
    if coeffs.len() <= 1 {
        return vec![];
    }
    let n = coeffs.len();
    let mut div = vec![F::zero(); n];
    for (i, &c) in coeffs.iter().enumerate() {
        div[i] = c;
    }
    div[0] -= value;
    let mut q = vec![F::zero(); n - 1];
    let mut carry = F::zero();
    for i in (1..n).rev() {
        let term = div[i] + carry;
        q[i - 1] = term;
        carry = term * point;
    }
    q
}

// ═══════════════════════════════════════════════════════════════════
// Gemini-Shplonk claim list — canonical fixed-order infrastructure.
//
// Prover and verifier MUST use the same claim list to:
//   a) bind all claims in transcript before deriving Shplonk:nu,
//   b) construct the batched quotient Q(X),
//   c) compute G(X) in the prover and [G] in the verifier.
//
// Canonical claim order (claim index j):
//   For i = 0..num_folds-1:
//     [2i]     A_i at +r^{2^i}    (nu power = 2i)
//     [2i+1]   A_i at -r^{2^i}    (nu power = 2i+1)
//   [2*num_folds]     A_last at 0   (c0)
//   [2*num_folds+1]   A_last at 1   (c0+c1)
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub(crate) struct GeminiClaim<F: Field> {
    /// Gemini round index: 0 = original polynomial, >=1 = fold polys
    pub round: usize,
    /// Opening point for this claim
    pub point: F,
    /// Claimed evaluation at the point
    pub evaluation: F,
}

/// Build the canonical fixed-order claim list for Gemini-Shplonk.
pub(crate) fn build_gemini_claims<F: Field>(
    mu: usize,
    r: F,
    fold_evals: &[(F, F)],
    c0: F,
    c1: F,
) -> Vec<GeminiClaim<F>> {
    let num_folds = mu.saturating_sub(1);
    let mut claims = Vec::with_capacity(2 * num_folds + 2);
    let mut r_pow = r;
    for i in 0..num_folds {
        claims.push(GeminiClaim {
            round: i,
            point: r_pow,
            evaluation: fold_evals[i].0,
        });
        claims.push(GeminiClaim {
            round: i,
            point: -r_pow,
            evaluation: fold_evals[i].1,
        });
        r_pow = r_pow * r_pow;
    }
    let last_round = mu.saturating_sub(1);
    claims.push(GeminiClaim {
        round: last_round,
        point: F::zero(),
        evaluation: c0,
    });
    claims.push(GeminiClaim {
        round: last_round,
        point: F::one(),
        evaluation: c0 + c1,
    });
    claims
}

/// Absorb all Shplonk claims into the transcript in canonical order.
/// Binds claim index, commitment, point, and evaluation before deriving
/// Shplonk:nu. Called by both prover and verifier.
pub(crate) fn append_gemini_claims_to_transcript<E: Pairing>(
    claims: &[GeminiClaim<E::ScalarField>],
    original_commitment: &E::G1Affine,
    fold_comms: &[E::G1Affine],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<(), PCSError> {
    for (claim_idx, claim) in claims.iter().enumerate() {
        let commit = if claim.round == 0 {
            *original_commitment
        } else {
            fold_comms[claim.round - 1]
        };
        transcript.append_field_element(
            b"Shplonk:claim_idx",
            &E::ScalarField::from(claim_idx as u64),
        )?;
        transcript.append_serializable_element(b"Shplonk:claim_cm", &commit)?;
        transcript.append_field_element(b"Shplonk:claim_pt", &claim.point)?;
        transcript.append_field_element(b"Shplonk:claim_ev", &claim.evaluation)?;
    }
    Ok(())
}

/// Map a claim's round to the correct G1 commitment.
fn claim_commit<E: Pairing>(
    round: usize,
    original_commitment: &E::G1Affine,
    fold_comms: &[E::G1Affine],
) -> E::G1Affine {
    if round == 0 {
        *original_commitment
    } else {
        fold_comms[round - 1]
    }
}

// ── Production challenge validation helpers ──

pub(crate) fn validate_shplonk_r<F: Field>(r: F, side: &str) -> Result<(), PCSError> {
    if r.is_zero() {
        return Err(PCSError::InvalidProof(format!(
            "Gemini {side}: r=0 degenerates fold relations"
        )));
    }
    Ok(())
}

pub(crate) fn validate_shplonk_nu<F: Field>(nu: F, side: &str) -> Result<(), PCSError> {
    if nu.is_zero() {
        return Err(PCSError::InvalidProof(format!(
            "Shplonk {side}: nu=0 makes batching degenerate"
        )));
    }
    Ok(())
}

pub(crate) fn validate_shplonk_z_against_claims<F: Field>(
    z: F,
    claims: &[GeminiClaim<F>],
    side: &str,
) -> Result<(), PCSError> {
    if z.is_zero() {
        return Err(PCSError::InvalidProof(format!("Shplonk {side}: z is zero")));
    }
    for claim in claims {
        if z == claim.point {
            return Err(PCSError::InvalidProof(format!(
                "Shplonk {side}: z collides with claim point {:?}",
                claim.point
            )));
        }
    }
    Ok(())
}

// ── Production Q / G / [G] helpers ──
//
// Q(X) = sum_j nu^j * (f_j(X) - v_j) / (X - x_j)
// G(X) = Q(X) - sum_j nu^j/(z - x_j) * (f_j(X) - v_j)
// Verifier reconstructs [G] = [Q] - sum_j nu^j/(z-x_j) * ([f_j] - v_j * [1])

pub(crate) fn compute_shplonk_batched_quotient<F: Field>(
    claims: &[GeminiClaim<F>],
    a_polys: &[Vec<F>],
    nu: F,
    n: usize,
) -> Vec<F> {
    let mut batched_q = vec![F::zero(); n];
    let mut nu_pow = F::one();
    for claim in claims {
        let q = kzg_quotient(&a_polys[claim.round], claim.point, claim.evaluation);
        for (j, &qj) in q.iter().enumerate() {
            batched_q[j] += nu_pow * qj;
        }
        nu_pow *= nu;
    }
    batched_q
}

pub(crate) fn compute_shplonk_g_coeffs<F: Field>(
    claims: &[GeminiClaim<F>],
    a_polys: &[Vec<F>],
    batched_q: &[F],
    nu: F,
    z: F,
    n: usize,
) -> Result<Vec<F>, PCSError> {
    let mut g_coeffs = batched_q.to_vec();
    g_coeffs.resize(n, F::zero());
    let mut nu_pow = F::one();
    for claim in claims {
        let denom = z - claim.point;
        let inv = denom.inverse().ok_or_else(|| {
            PCSError::InvalidProof(
                "Shplonk prover: z collides with a claim point during G construction".to_string(),
            )
        })?;
        let scale = nu_pow * inv;
        for (j, &cj) in a_polys[claim.round].iter().enumerate() {
            g_coeffs[j] -= scale * cj;
        }
        g_coeffs[0] += scale * claim.evaluation;
        nu_pow *= nu;
    }
    Ok(g_coeffs)
}

pub(crate) fn compute_shplonk_g_commit<E: Pairing>(
    claims: &[GeminiClaim<E::ScalarField>],
    shplonk_q_commit: &E::G1Affine,
    original_commitment: &E::G1Affine,
    fold_comms: &[E::G1Affine],
    nu: E::ScalarField,
    z: E::ScalarField,
    vp_g: &E::G1Affine,
) -> Result<E::G1Affine, PCSError> {
    let mut bases: Vec<E::G1Affine> = vec![*shplonk_q_commit];
    let mut scalars: Vec<E::ScalarField> = vec![E::ScalarField::one()];
    let mut identity_scalar = E::ScalarField::zero();
    let mut nu_pow = E::ScalarField::one();

    for claim in claims {
        let commit = claim_commit::<E>(claim.round, original_commitment, fold_comms);
        let denom = z - claim.point;
        let inv = denom.inverse().ok_or_else(|| {
            PCSError::InvalidProof(
                "Shplonk verifier: z collides with a claim point during G reduction".to_string(),
            )
        })?;
        let scale = nu_pow * inv;
        bases.push(commit);
        scalars.push(-scale);
        identity_scalar += scale * claim.evaluation;
        nu_pow *= nu;
    }

    bases.push(*vp_g);
    scalars.push(identity_scalar);

    Ok(E::G1::msm_unchecked(&bases, &scalars).into_affine())
}

// ═══════════════════════════════════════════════════════════════════
// Transcript-aware single opening (Shplonk-batched)
// ═══════════════════════════════════════════════════════════════════

fn gemini_open_with_transcript<E: Pairing>(
    pp: &GeminiProverParam<E>,
    poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
    point: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<(GeminiProof<E>, E::ScalarField), PCSError> {
    let mu = poly.num_vars();
    let _n = checked_domain_size_from_mu(mu, "open")
        .map_err(|e| PCSError::InvalidParameters(e.to_string()))?;

    if point.len() != mu {
        return Err(PCSError::InvalidParameters(
            "point length mismatch".to_string(),
        ));
    }

    let f_hat = poly.to_evaluations();
    let eval = poly
        .evaluate(point)
        .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;

    let a0_commit = pp.try_commit(&f_hat)?;
    transcript.append_serializable_element(b"commitment", &a0_commit)?;
    transcript.append_serializable_element(b"point", &point.to_vec())?;
    transcript.append_field_element(b"eval", &eval)?;

    gemini_core_open_prebound(pp, poly, point, f_hat, eval, transcript, a0_commit)
}

/// Core Gemini opening: the transcript already has commitment, point, and
/// evaluation bound.  `a0_commit` is the G1Affine of C_f (already computed).
fn gemini_core_open_prebound<E: Pairing>(
    pp: &GeminiProverParam<E>,
    poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
    point: &[E::ScalarField],
    f_hat: Vec<E::ScalarField>,
    eval: E::ScalarField,
    transcript: &mut IOPTranscript<E::ScalarField>,
    a0_commit: E::G1Affine,
) -> Result<(GeminiProof<E>, E::ScalarField), PCSError> {
    let mu = poly.num_vars();
    let n = 1usize << mu;

    let _t_total = ScopedTimer::new(BACKEND, mu, n, "gemini_open_total", 1, "total");

    let _t_fold = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_open_fold_polys",
        mu,
        "fold-poly-rounds",
    );
    let mut a_polys: Vec<Vec<E::ScalarField>> = Vec::with_capacity(mu);
    a_polys.push(f_hat.clone());
    for i in 0..mu.saturating_sub(1) {
        let next = fold_polynomial(&a_polys[i], point[i]);
        a_polys.push(next);
    }
    drop(_t_fold);

    let _t_cm = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_open_commit_folds",
        a_polys.len().saturating_sub(1),
        "KZG-commit-fold",
    );
    let mut fold_comms: Vec<E::G1Affine> = Vec::with_capacity(mu.saturating_sub(1));
    for a in a_polys.iter().skip(1) {
        fold_comms.push(pp.try_commit(a)?);
    }
    drop(_t_cm);

    for cm in &fold_comms {
        transcript.append_serializable_element(b"fold_comm", cm)?;
    }

    let r = transcript.get_and_append_challenge_vectors(b"r", 1)?[0];
    validate_shplonk_r(r, "prover").map_err(|_| {
        PCSError::InvalidProver("Gemini: r=0 degenerates fold relations".to_string())
    })?;

    let _t_claims = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_open_build_shplonk_claims",
        mu,
        "eval-rounds",
    );
    let mut fold_evals: Vec<(E::ScalarField, E::ScalarField)> =
        Vec::with_capacity(mu.saturating_sub(1));
    let mut r_pow = r;
    for i in 0..mu.saturating_sub(1) {
        let a_i_r = poly_eval(&a_polys[i], r_pow);
        let a_i_neg_r = poly_eval(&a_polys[i], -r_pow);
        fold_evals.push((a_i_r, a_i_neg_r));
        r_pow = r_pow * r_pow;
    }
    let c0 = a_polys[mu.saturating_sub(1)][0];
    let c1 = a_polys[mu.saturating_sub(1)][1];
    let one = E::ScalarField::one();
    let computed_eval = (one - point[mu.saturating_sub(1)]) * c0 + point[mu.saturating_sub(1)] * c1;
    if computed_eval != eval {
        return Err(PCSError::InvalidProver(
            "fold check failed: eval mismatch".to_string(),
        ));
    }
    drop(_t_claims);

    let claims = build_gemini_claims(mu, r, &fold_evals, c0, c1);
    append_gemini_claims_to_transcript::<E>(&claims, &a0_commit, &fold_comms, transcript)?;

    let nu = transcript.get_and_append_challenge_vectors(b"Shplonk:nu", 1)?[0];
    validate_shplonk_nu(nu, "prover").map_err(|_| {
        PCSError::InvalidProver("Shplonk: nu=0 makes batching degenerate".to_string())
    })?;

    let _t_batch_q = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_open_compute_batched_quotient",
        1,
        "batched-quotient",
    );
    let batched_q = compute_shplonk_batched_quotient(&claims, &a_polys, nu, n);
    drop(_t_batch_q);

    let _t_cm_q = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_open_commit_batched_quotient",
        1,
        "KZG-commit-Q",
    );
    let shplonk_q_commit = pp.try_commit(&batched_q)?;
    transcript.append_serializable_element(b"Shplonk:Q", &shplonk_q_commit)?;
    drop(_t_cm_q);

    let z = transcript.get_and_append_challenge_vectors(b"Shplonk:z", 1)?[0];
    validate_shplonk_z_against_claims(z, &claims, "prover")
        .map_err(|_| PCSError::InvalidProver(format!("Shplonk: z={z:?} invalid")))?;

    let _t_partial = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_open_compute_partial_eval",
        1,
        "G-partial-eval",
    );
    let g_coeffs = compute_shplonk_g_coeffs(&claims, &a_polys, &batched_q, nu, z, n)?;
    drop(_t_partial);

    let g_z = poly_eval(&g_coeffs, z);
    if g_z != E::ScalarField::zero() {
        return Err(PCSError::InvalidProver(format!(
            "Shplonk: G(z)={g_z:?} != 0"
        )));
    }

    let _t_witness = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_open_final_kzg_witness",
        1,
        "KZG-commit-witness",
    );
    let witness_coeffs = kzg_quotient(&g_coeffs, z, E::ScalarField::zero());
    let kzg_witness = pp.try_commit(&witness_coeffs)?;
    drop(_t_witness);

    let proof = GeminiProof {
        fold_comms,
        fold_evals,
        final_coeffs: (c0, c1),
        shplonk_q_commit,
        kzg_witness,
        mu,
    };
    drop(_t_total);
    Ok((proof, eval))
}

// ═══════════════════════════════════════════════════════════════════
// Transcript-aware single verification (Shplonk-batched)
// ═══════════════════════════════════════════════════════════════════

fn gemini_verify_with_transcript<E: Pairing>(
    vp: &GeminiVerifierParam<E>,
    commitment: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &GeminiProof<E>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError> {
    let mu = proof.mu;

    if mu > vp.max_num_vars {
        return Err(PCSError::InvalidProof(format!(
            "verify: proof.mu {} exceeds vp.max_num_vars {}",
            mu, vp.max_num_vars
        )));
    }

    let n = checked_domain_size_from_mu(mu, "verify")?;
    if n > vp.max_degree {
        return Err(PCSError::InvalidProof(format!(
            "verify: n={} exceeds vp.max_degree={}",
            n, vp.max_degree
        )));
    }

    if point.len() != mu {
        return Err(PCSError::InvalidProof(format!(
            "verify: point length {} != proof.mu {}",
            point.len(),
            mu
        )));
    }

    let _t_total = ScopedTimer::new(BACKEND, mu, n, "gemini_verify_total", 1, "total");

    let num_folds = mu.saturating_sub(1);
    if proof.fold_comms.len() != num_folds {
        return Err(PCSError::InvalidProof(format!(
            "verify: fold_comms length {} != expected {}",
            proof.fold_comms.len(),
            num_folds
        )));
    }
    if proof.fold_evals.len() != num_folds {
        return Err(PCSError::InvalidProof(format!(
            "verify: fold_evals length {} != expected {}",
            proof.fold_evals.len(),
            num_folds
        )));
    }

    transcript.append_serializable_element(b"commitment", &commitment.0)?;
    transcript.append_serializable_element(b"point", &point.to_vec())?;
    transcript.append_field_element(b"eval", value)?;

    for cm in &proof.fold_comms {
        transcript.append_serializable_element(b"fold_comm", cm)?;
    }

    let r = transcript.get_and_append_challenge_vectors(b"r", 1)?[0];
    validate_shplonk_r(r, "verifier")?;

    let _t_fold_check = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_verify_fold_checks",
        mu.saturating_sub(1),
        "fold-equations",
    );
    let mut r_pow = r;
    for i in 0..num_folds {
        let (a_i_r, a_i_neg_r) = proof.fold_evals[i];
        let a_next_r_sq = if i + 1 < num_folds {
            proof.fold_evals[i + 1].0
        } else {
            let (c0, c1) = proof.final_coeffs;
            let r_sq = r_pow * r_pow;
            c0 + c1 * r_sq
        };
        let two_r_pow = E::ScalarField::from(2u64) * r_pow;
        let lhs = two_r_pow * a_next_r_sq;
        let even_part = a_i_r + a_i_neg_r;
        let odd_part = a_i_r - a_i_neg_r;
        let rhs = (E::ScalarField::one() - point[i]) * r_pow * even_part + point[i] * odd_part;
        if lhs != rhs {
            return Ok(false);
        }
        r_pow = r_pow * r_pow;
    }
    drop(_t_fold_check);

    let (c0, c1) = proof.final_coeffs;
    let one = E::ScalarField::one();
    let expected_eval = (one - point[mu.saturating_sub(1)]) * c0 + point[mu.saturating_sub(1)] * c1;
    if expected_eval != *value {
        return Ok(false);
    }

    let claims = build_gemini_claims(mu, r, &proof.fold_evals, c0, c1);
    append_gemini_claims_to_transcript::<E>(&claims, &commitment.0, &proof.fold_comms, transcript)?;

    let nu = transcript.get_and_append_challenge_vectors(b"Shplonk:nu", 1)?[0];
    validate_shplonk_nu(nu, "verifier")?;
    transcript.append_serializable_element(b"Shplonk:Q", &proof.shplonk_q_commit)?;
    let z = transcript.get_and_append_challenge_vectors(b"Shplonk:z", 1)?[0];
    validate_shplonk_z_against_claims(z, &claims, "verifier")?;

    let _t_msm = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_verify_shplonk_msm",
        1,
        "shplonk-msm",
    );
    let g_commit = compute_shplonk_g_commit::<E>(
        &claims,
        &proof.shplonk_q_commit,
        &commitment.0,
        &proof.fold_comms,
        nu,
        z,
        &vp.g,
    )?;
    drop(_t_msm);

    let _t_pairing = ScopedTimer::new(BACKEND, mu, n, "gemini_verify_kzg_pairing", 1, "1-pairing");
    let valid = kzg_verify_pairing(vp, &g_commit, z, E::ScalarField::zero(), &proof.kzg_witness)?;
    drop(_t_pairing);
    drop(_t_total);

    Ok(valid)
}

fn kzg_verify_pairing<E: Pairing>(
    vp: &GeminiVerifierParam<E>,
    com: &E::G1Affine,
    point: E::ScalarField,
    eval: E::ScalarField,
    proof: &E::G1Affine,
) -> Result<bool, PCSError> {
    let neg_eval_g = (-vp.g.into_group() * eval).into_affine();
    let cm_minus_vg = (com.into_group() + neg_eval_g).into_affine();
    let sx = (vp.h_x.into_group() - vp.h.into_group() * point).into_affine();
    let neg_proof = (-proof.into_group()).into_affine();
    let result = E::multi_pairing([cm_minus_vg, neg_proof], [vp.h, sx]);
    Ok(result == PairingOutput(E::TargetField::one()))
}

// ═══════════════════════════════════════════════════════════════════
// Sumcheck batching — multi-open
// ═══════════════════════════════════════════════════════════════════

fn gemini_sumcheck_multi_open<E: Pairing>(
    pp: &GeminiProverParam<E>,
    polynomials: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    points: &[Vec<E::ScalarField>],
    evals: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<BatchProof<E, GeminiPCS<E>>, PCSError> {
    if polynomials.is_empty() {
        return Err(PCSError::InvalidParameters(
            "empty polynomial list".to_string(),
        ));
    }
    if polynomials.len() != points.len() || polynomials.len() != evals.len() {
        return Err(PCSError::InvalidParameters("length mismatch".to_string()));
    }
    let num_var = polynomials[0].num_vars;
    let _n = checked_domain_size_from_mu(num_var, "multi_open")
        .map_err(|e| PCSError::InvalidParameters(e.to_string()))?;
    let k = polynomials.len();
    for p in polynomials {
        if p.num_vars != num_var {
            return Err(PCSError::InvalidParameters(
                "inconsistent num_vars".to_string(),
            ));
        }
    }
    for pt in points {
        if pt.len() != num_var {
            return Err(PCSError::InvalidParameters(
                "point length mismatch".to_string(),
            ));
        }
    }

    for pt in points {
        transcript.append_serializable_element(b"eval_point", pt)?;
    }
    for e in evals {
        transcript.append_field_element(b"eval", e)?;
    }

    let ell = k.next_power_of_two().ilog2() as usize;
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;
    let eq_t_list = if ell == 0 {
        vec![E::ScalarField::one()]
    } else {
        build_eq_x_r_vec(t.as_ref())?
    };

    let point_indices = points.iter().fold(BTreeMap::<_, _>::new(), |mut m, pt| {
        let i = m.len();
        m.entry(pt).or_insert(i);
        m
    });
    let deduped_points: Vec<_> = BTreeMap::from_iter(point_indices.iter().map(|(pt, i)| (*i, *pt)))
        .into_values()
        .collect();

    let merged_tilde_gs = polynomials
        .iter()
        .zip(points.iter())
        .zip(eq_t_list.iter())
        .fold(
            iter::repeat_with(DenseMultilinearExtension::zero)
                .map(Arc::new)
                .take(point_indices.len())
                .collect::<Vec<_>>(),
            |mut merged, ((poly, pt), c)| {
                *Arc::make_mut(&mut merged[point_indices[pt]]) += (*c, poly.deref());
                merged
            },
        );

    let tilde_eqs: Vec<_> = deduped_points
        .iter()
        .map(|pt| {
            Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_var,
                build_eq_x_r_vec(pt)?,
            )))
        })
        .collect::<Result<Vec<_>, PCSError>>()?;

    let mut sum_check_vp = VirtualPolynomial::new(num_var);
    for (g, eq) in merged_tilde_gs.iter().zip(tilde_eqs.into_iter()) {
        sum_check_vp.add_mle_list([g.clone(), eq], E::ScalarField::one())?;
    }

    let sumcheck_proof = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::prove(
        &sum_check_vp,
        transcript,
    ) {
        Ok(p) => p,
        Err(_) => return Err(PCSError::InvalidProver("Sumcheck failed".to_string())),
    };

    let a2 = &sumcheck_proof.point[..num_var];
    let mut g_prime = Arc::new(DenseMultilinearExtension::zero());
    for (g, pt) in merged_tilde_gs.iter().zip(deduped_points.iter()) {
        let eq = eq_eval(a2, pt)?;
        *Arc::make_mut(&mut g_prime) += (eq, g.deref());
    }

    let mut open_t = IOPTranscript::new(b"gemini-gprime-open");
    open_t.append_field_element(b"mu", &E::ScalarField::from(num_var as u64))?;
    let (g_prime_proof, _g_prime_eval) =
        gemini_open_with_transcript(pp, &g_prime, a2, &mut open_t)?;

    Ok(BatchProof {
        sum_check_proof: sumcheck_proof,
        f_i_eval_at_point_i: evals.to_vec(),
        g_prime_proof,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Sumcheck batching — batch verify
// ═══════════════════════════════════════════════════════════════════

fn gemini_sumcheck_batch_verify<E: Pairing>(
    vp: &GeminiVerifierParam<E>,
    f_i_commitments: &[Commitment<E>],
    points: &[Vec<E::ScalarField>],
    proof: &BatchProof<E, GeminiPCS<E>>,
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

    if num_var == 0 {
        return Err(PCSError::InvalidProof(
            "batch_verify: num_var is zero".to_string(),
        ));
    }
    if num_var > vp.max_num_vars {
        return Err(PCSError::InvalidProof(format!(
            "batch_verify: num_var {} exceeds vp.max_num_vars {}",
            num_var, vp.max_num_vars
        )));
    }
    let _n = checked_domain_size_from_mu(num_var, "batch_verify")?;
    if _n > vp.max_degree {
        return Err(PCSError::InvalidProof(format!(
            "batch_verify: n={_n} exceeds vp.max_degree={}",
            vp.max_degree
        )));
    }

    for pt in points {
        if pt.len() != num_var {
            return Err(PCSError::InvalidProof("point length mismatch".to_string()));
        }
    }

    for pt in points {
        transcript.append_serializable_element(b"eval_point", pt)?;
    }
    for e in &proof.f_i_eval_at_point_i {
        transcript.append_field_element(b"eval", e)?;
    }

    let ell = k.next_power_of_two().ilog2() as usize;
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;
    let a2 = &proof.sum_check_proof.point[..num_var];
    let eq_t_list = if ell == 0 {
        vec![E::ScalarField::one()]
    } else {
        build_eq_x_r_vec(t.as_ref())?
    };

    let mut scalars = vec![];
    let mut bases = vec![];
    for (i, pt) in points.iter().enumerate() {
        scalars.push(eq_eval(a2, pt)? * eq_t_list[i]);
        bases.push(f_i_commitments[i].0);
    }
    let g_prime_commit = E::G1::msm_unchecked(&bases, &scalars);

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
        Err(_) => return Ok(false),
    };

    let mut verify_t = IOPTranscript::new(b"gemini-gprime-open");
    verify_t.append_field_element(b"mu", &E::ScalarField::from(num_var as u64))?;
    gemini_verify_with_transcript(
        vp,
        &Commitment(g_prime_commit.into_affine()),
        a2,
        &subclaim.expected_evaluation,
        &proof.g_prime_proof,
        &mut verify_t,
    )
}

// ═══════════════════════════════════════════════════════════════════
// Polynomial utilities for sumcheck batching
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

    fn setup(nv: usize) -> (GeminiProverParam<E>, GeminiVerifierParam<E>) {
        let mut rng = test_rng();
        GeminiPCS::<E>::trim(
            &GeminiPCS::<E>::gen_srs_for_testing(&mut rng, nv).unwrap(),
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

    #[test]
    fn test_gemini_single_open_verify() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [2usize, 4, 6] {
            let (ck, vk) = setup(nv);
            let p = rpoly(nv, &mut rng);
            let pt = rpt(nv, &mut rng);
            let com = GeminiPCS::<E>::commit(&ck, &p)?;
            let (proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt)?;
            assert!(
                GeminiPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?,
                "Gemini open/verify nv={nv}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_gemini_reject_wrong_eval() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [2usize, 4] {
            let (ck, vk) = setup(nv);
            let p = rpoly(nv, &mut rng);
            let pt = rpt(nv, &mut rng);
            let com = GeminiPCS::<E>::commit(&ck, &p)?;
            let (proof, _val) = GeminiPCS::<E>::open(&ck, &p, &pt)?;
            let fv = Fr::rand(&mut rng);
            assert!(
                !GeminiPCS::<E>::verify(&vk, &com, &pt, &fv, &proof)?,
                "wrong eval should reject nv={nv}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_gemini_reject_wrong_point() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p)?;
        let (proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt)?;
        assert!(!GeminiPCS::<E>::verify(
            &vk,
            &com,
            &rpt(nv, &mut rng),
            &val,
            &proof
        )?);
        Ok(())
    }

    #[test]
    fn test_gemini_reject_wrong_commitment() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p1 = rpoly(nv, &mut rng);
        let p2 = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com2 = GeminiPCS::<E>::commit(&ck, &p2)?;
        let (proof, val) = GeminiPCS::<E>::open(&ck, &p1, &pt)?;
        assert!(!GeminiPCS::<E>::verify(&vk, &com2, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_gemini_reject_tampered_fold_commit() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt)?;
        proof.fold_comms[0] = (proof.fold_comms[0].into_group() * Fr::from(2u64)).into_affine();
        assert!(!GeminiPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_gemini_reject_tampered_eval() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt)?;
        let (a_r, _) = proof.fold_evals[0];
        proof.fold_evals[0] = (a_r + Fr::one(), proof.fold_evals[0].1);
        assert!(!GeminiPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_gemini_reject_tampered_shplonk_q_commit() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt)?;
        proof.shplonk_q_commit =
            (proof.shplonk_q_commit.into_group() * Fr::from(3u64)).into_affine();
        assert!(!GeminiPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_gemini_reject_tampered_kzg_witness() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt)?;
        proof.kzg_witness = (proof.kzg_witness.into_group() * Fr::from(5u64)).into_affine();
        assert!(!GeminiPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_gemini_verify_rejects_wrong_point_len() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p)?;
        let (proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt)?;
        let short_pt = rpt(2, &mut rng);
        let r = GeminiPCS::<E>::verify(&vk, &com, &short_pt, &val, &proof);
        assert!(r.is_err(), "short point should return Error");
        Ok(())
    }

    #[test]
    fn test_gemini_verify_rejects_huge_mu_without_panic() {
        let mut rng = test_rng();
        let nv = 2;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p).unwrap();
        let (mut proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt).unwrap();
        proof.mu = usize::BITS as usize;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            GeminiPCS::<E>::verify(&vk, &com, &pt, &val, &proof)
        }));
        match r {
            Ok(verdict) => assert!(
                verdict.is_err() || !verdict.unwrap(),
                "huge mu should fail without panic"
            ),
            Err(_) => panic!("caught panic on huge mu — should not panic"),
        }
    }

    #[test]
    fn test_gemini_multi_open_k1() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..1).map(|_| rpoly(nv, &mut rng)).collect();
        let pts: Vec<_> = polys.iter().map(|_| rpt(nv, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(pts.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| GeminiPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = GeminiPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(GeminiPCS::<E>::batch_verify(
            &vk, &comms, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_gemini_multi_open_distinct() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..3).map(|_| rpoly(nv, &mut rng)).collect();
        let pts: Vec<_> = polys.iter().map(|_| rpt(nv, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(pts.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| GeminiPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = GeminiPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(GeminiPCS::<E>::batch_verify(
            &vk, &comms, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_gemini_multi_open_repeated() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let pt = rpt(nv, &mut rng);
        let polys: Vec<_> = (0..3).map(|_| rpoly(nv, &mut rng)).collect();
        let pts: Vec<_> = vec![pt.clone(), pt.clone(), pt.clone()];
        let evals: Vec<Fr> = polys
            .iter()
            .zip(pts.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| GeminiPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = GeminiPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(GeminiPCS::<E>::batch_verify(
            &vk, &comms, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_gemini_batch_reject_wrong_eval() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..3).map(|_| rpoly(nv, &mut rng)).collect();
        let pts: Vec<_> = polys.iter().map(|_| rpt(nv, &mut rng)).collect();
        let mut evals: Vec<Fr> = polys
            .iter()
            .zip(pts.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| GeminiPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        evals[0] += Fr::one();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = GeminiPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(!GeminiPCS::<E>::batch_verify(
            &vk, &comms, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_gemini_batch_reject_wrong_point() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..3).map(|_| rpoly(nv, &mut rng)).collect();
        let pts: Vec<_> = polys.iter().map(|_| rpt(nv, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(pts.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| GeminiPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = GeminiPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut wp = pts.clone();
        wp[0] = rpt(nv, &mut rng);
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert_rejects(GeminiPCS::<E>::batch_verify(
            &vk, &comms, &wp, &proof, &mut tv,
        ));
        Ok(())
    }

    #[test]
    fn test_gemini_batch_reject_wrong_commitment() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..3).map(|_| rpoly(nv, &mut rng)).collect();
        let pts: Vec<_> = polys.iter().map(|_| rpt(nv, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(pts.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| GeminiPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = GeminiPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let extra = GeminiPCS::<E>::commit(&ck, &rpoly(nv, &mut rng))?;
        let mut wc = comms.clone();
        wc[0] = extra;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(!GeminiPCS::<E>::batch_verify(
            &vk, &wc, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_gemini_batch_reject_malformed_lengths() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..3).map(|_| rpoly(nv, &mut rng)).collect();
        let pts: Vec<_> = polys.iter().map(|_| rpt(nv, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(pts.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| GeminiPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let mut proof = GeminiPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let r = GeminiPCS::<E>::batch_verify(
            &vk,
            &comms[..2],
            &pts,
            &proof,
            &mut IOPTranscript::new(b"t"),
        );
        assert!(r.is_err() || !r.unwrap());
        proof.f_i_eval_at_point_i.pop();
        let r2 =
            GeminiPCS::<E>::batch_verify(&vk, &comms, &pts, &proof, &mut IOPTranscript::new(b"t"));
        assert!(r2.is_err() || !r2.unwrap());
        Ok(())
    }

    // ── vk bound checks ──

    #[test]
    fn test_gemini_verify_rejects_mu_above_vk_bound() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let big_nv = 6;
        let small_nv = 4;
        let srs = GeminiPCS::<E>::gen_srs_for_testing(&mut rng, big_nv)?;
        let (big_ck, _) = GeminiPCS::<E>::trim(&srs, None, Some(big_nv))?;
        let (_, small_vk) = GeminiPCS::<E>::trim(&srs, None, Some(small_nv))?;
        let p = rpoly(big_nv, &mut rng);
        let pt = rpt(big_nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&big_ck, &p)?;
        let (proof, val) = GeminiPCS::<E>::open(&big_ck, &p, &pt)?;
        let r = GeminiPCS::<E>::verify(&small_vk, &com, &pt, &val, &proof);
        assert!(r.is_err(), "mu above vk bound should return Error");
        Ok(())
    }

    #[test]
    fn test_gemini_verify_rejects_degree_above_vk_bound() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let big_nv = 6;
        let small_nv = 4;
        let srs = GeminiPCS::<E>::gen_srs_for_testing(&mut rng, big_nv)?;
        let (big_ck, _) = GeminiPCS::<E>::trim(&srs, None, Some(big_nv))?;
        let (_, small_vk) = srs.trim(1 << small_nv)?;
        let p = rpoly(big_nv, &mut rng);
        let pt = rpt(big_nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&big_ck, &p)?;
        let (proof, val) = GeminiPCS::<E>::open(&big_ck, &p, &pt)?;
        let r = GeminiPCS::<E>::verify(&small_vk, &com, &pt, &val, &proof);
        assert!(r.is_err(), "degree above vk bound should return Error");
        Ok(())
    }

    // ── SRS too small without panic ──

    #[test]
    fn test_gemini_commit_rejects_srs_too_small_without_panic() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let big_nv = 4;
        let tiny_nv = 2;
        let srs = GeminiPCS::<E>::gen_srs_for_testing(&mut rng, big_nv)?;
        let (tiny_ck, _) = GeminiPCS::<E>::trim(&srs, None, Some(tiny_nv))?;
        let big_poly = rpoly(big_nv, &mut rng);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            GeminiPCS::<E>::commit(&tiny_ck, &big_poly)
        }));
        match r {
            Ok(verdict) => assert!(
                verdict.is_err(),
                "commit with too-small SRS should return Err"
            ),
            Err(_) => panic!("commit should not panic on too-small SRS"),
        }
        Ok(())
    }

    #[test]
    fn test_gemini_open_rejects_srs_too_small_without_panic() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let big_nv = 4;
        let tiny_nv = 2;
        let srs = GeminiPCS::<E>::gen_srs_for_testing(&mut rng, big_nv)?;
        let (tiny_ck, _) = GeminiPCS::<E>::trim(&srs, None, Some(tiny_nv))?;
        let big_poly = rpoly(big_nv, &mut rng);
        let pt = rpt(big_nv, &mut rng);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            GeminiPCS::<E>::open(&tiny_ck, &big_poly, &pt)
        }));
        match r {
            Ok(verdict) => assert!(
                verdict.is_err(),
                "open with too-small SRS should return Err"
            ),
            Err(_) => panic!("open should not panic on too-small SRS"),
        }
        Ok(())
    }

    // ── Batch verify huge num_var / above vk bound ──

    #[test]
    fn test_gemini_batch_verify_rejects_huge_num_var_without_panic() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 2;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..1).map(|_| rpoly(nv, &mut rng)).collect();
        let pts: Vec<_> = polys.iter().map(|_| rpt(nv, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(pts.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| GeminiPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let mut proof = GeminiPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        proof.sum_check_proof.point = vec![Fr::zero(); usize::BITS as usize];
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            GeminiPCS::<E>::batch_verify(&vk, &comms, &pts, &proof, &mut tv)
        }));
        match r {
            Ok(verdict) => assert!(
                verdict.is_err() || !verdict.unwrap(),
                "huge num_var should fail"
            ),
            Err(_) => panic!("caught panic on huge num_var — should not panic"),
        }
        Ok(())
    }

    #[test]
    fn test_gemini_batch_verify_rejects_num_var_above_vk_bound() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let big_nv = 4;
        let small_nv = 2;
        let srs = GeminiPCS::<E>::gen_srs_for_testing(&mut rng, big_nv)?;
        let (big_ck, _) = GeminiPCS::<E>::trim(&srs, None, Some(big_nv))?;
        let (_, small_vk) = GeminiPCS::<E>::trim(&srs, None, Some(small_nv))?;
        let polys: Vec<_> = (0..1).map(|_| rpoly(big_nv, &mut rng)).collect();
        let pts: Vec<_> = polys.iter().map(|_| rpt(big_nv, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(pts.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| GeminiPCS::<E>::commit(&big_ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = GeminiPCS::<E>::multi_open(&big_ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        let r = GeminiPCS::<E>::batch_verify(&small_vk, &comms, &pts, &proof, &mut tv);
        assert!(r.is_err(), "num_var above vk bound should return Error");
        Ok(())
    }

    // ── Shplonk-batched proof element count ──

    #[test]
    fn test_gemini_proof_kzg_count_is_minimal() {
        let mut rng = test_rng();
        for nv in [2, 4, 6] {
            let (ck, vk) = setup(nv);
            let p = rpoly(nv, &mut rng);
            let pt = rpt(nv, &mut rng);
            let _com = GeminiPCS::<E>::commit(&ck, &p).unwrap();
            let (proof, _val) = GeminiPCS::<E>::open(&ck, &p, &pt).unwrap();
            assert_eq!(
                proof.fold_comms.len(),
                nv - 1,
                "nv={nv}: expected {} fold comms, got {}",
                nv - 1,
                proof.fold_comms.len()
            );
            assert!(GeminiPCS::<E>::verify(&vk, &_com, &pt, &_val, &proof).unwrap());
        }
    }

    // ── Tampered next-round eval must be rejected ──

    #[test]
    fn test_gemini_reject_tampered_next_round_eval() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 5;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt)?;
        if nv > 2 && proof.fold_evals.len() > 1 {
            proof.fold_evals[1].0 += Fr::ONE;
        }
        assert!(
            !GeminiPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?,
            "tampered next-round eval should reject"
        );
        Ok(())
    }

    // ── Tampered final_coeffs should break last fold relation ──

    #[test]
    fn test_gemini_reject_tampered_final_coeff_affects_last_fold() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt)?;
        proof.final_coeffs.0 += Fr::ONE;
        assert!(
            !GeminiPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?,
            "tampered final coeff should reject"
        );
        Ok(())
    }

    // ── Malformed fold_comms length guards ──

    #[test]
    fn test_gemini_verify_rejects_short_fold_comms_without_panic() {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p).unwrap();
        let (mut proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt).unwrap();
        proof.fold_comms.pop();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            GeminiPCS::<E>::verify(&vk, &com, &pt, &val, &proof)
        }));
        match r {
            Ok(verdict) => assert!(
                verdict.is_err() || !verdict.unwrap(),
                "short fold_comms should fail"
            ),
            Err(_) => panic!("short fold_comms should not panic"),
        }
    }

    #[test]
    fn test_gemini_verify_rejects_long_fold_comms_without_panic() {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p).unwrap();
        let (mut proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt).unwrap();
        let extra = proof.fold_comms[0];
        proof.fold_comms.push(extra);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            GeminiPCS::<E>::verify(&vk, &com, &pt, &val, &proof)
        }));
        match r {
            Ok(verdict) => assert!(
                verdict.is_err() || !verdict.unwrap(),
                "long fold_comms should fail"
            ),
            Err(_) => panic!("long fold_comms should not panic"),
        }
    }

    // ── Shplonk algebraic tests (all call production helpers) ──

    #[test]
    fn test_kzg_quotient_correctness() {
        let mut rng = test_rng();
        for deg in [1, 2, 4, 8] {
            let coeffs: Vec<Fr> = (0..=deg).map(|_| Fr::rand(&mut rng)).collect();
            let point = Fr::rand(&mut rng);
            let value = poly_eval(&coeffs, point);
            let q = kzg_quotient(&coeffs, point, value);
            let test_x = Fr::rand(&mut rng);
            let f_test = poly_eval(&coeffs, test_x);
            let q_test = poly_eval(&q, test_x);
            let recovered = (test_x - point) * q_test + value;
            assert_eq!(f_test, recovered, "kzg_quotient failed at deg={deg}");
        }
    }

    #[test]
    fn test_shplonk_g_vanishes_at_z() {
        let mut rng = test_rng();
        for nv in [1, 2, 4] {
            let (_ck, _vk) = setup(nv);
            let p = rpoly(nv, &mut rng);
            let pt = rpt(nv, &mut rng);
            let mu = p.num_vars();
            let n = 1usize << mu;
            let f_hat = p.to_evaluations();

            let mut a_polys: Vec<Vec<Fr>> = Vec::with_capacity(mu);
            a_polys.push(f_hat.clone());
            for i in 0..mu.saturating_sub(1) {
                a_polys.push(fold_polynomial(&a_polys[i], pt[i]));
            }
            let c0 = a_polys[mu.saturating_sub(1)][0];
            let c1 = a_polys[mu.saturating_sub(1)][1];

            let r = Fr::rand(&mut rng);
            let mut fold_evals: Vec<(Fr, Fr)> = vec![];
            let mut rp = r;
            for i in 0..mu.saturating_sub(1) {
                fold_evals.push((poly_eval(&a_polys[i], rp), poly_eval(&a_polys[i], -rp)));
                rp = rp * rp;
            }

            let claims = build_gemini_claims(mu, r, &fold_evals, c0, c1);
            let nu = Fr::rand(&mut rng);
            let batched_q = compute_shplonk_batched_quotient(&claims, &a_polys, nu, n);

            let z = loop {
                let z = Fr::rand(&mut rng);
                if validate_shplonk_z_against_claims(z, &claims, "test").is_ok() {
                    break z;
                }
            };

            let g_coeffs =
                compute_shplonk_g_coeffs(&claims, &a_polys, &batched_q, nu, z, n).unwrap();
            let g_z = poly_eval(&g_coeffs, z);
            assert_eq!(g_z, Fr::zero(), "G(z) must be zero for nv={nv}");
        }
    }

    #[test]
    fn test_shplonk_commit_g_matches_verifier_reduction() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [1, 2, 4] {
            let (ck, vk) = setup(nv);
            let p = rpoly(nv, &mut rng);
            let pt = rpt(nv, &mut rng);
            let mu = p.num_vars();
            let n = 1usize << mu;
            let f_hat = p.to_evaluations();
            let com = ck.try_commit(&f_hat)?;

            let mut a_polys: Vec<Vec<Fr>> = Vec::with_capacity(mu);
            a_polys.push(f_hat.clone());
            for i in 0..mu.saturating_sub(1) {
                a_polys.push(fold_polynomial(&a_polys[i], pt[i]));
            }
            let mut fold_comms: Vec<_> = vec![];
            for a in a_polys.iter().skip(1) {
                fold_comms.push(ck.try_commit(a)?);
            }

            let c0 = a_polys[mu.saturating_sub(1)][0];
            let c1 = a_polys[mu.saturating_sub(1)][1];

            let r = Fr::rand(&mut rng);
            let mut fold_evals: Vec<(Fr, Fr)> = vec![];
            let mut rp = r;
            for i in 0..mu.saturating_sub(1) {
                fold_evals.push((poly_eval(&a_polys[i], rp), poly_eval(&a_polys[i], -rp)));
                rp = rp * rp;
            }

            let claims = build_gemini_claims(mu, r, &fold_evals, c0, c1);
            let nu = Fr::rand(&mut rng);
            let batched_q = compute_shplonk_batched_quotient(&claims, &a_polys, nu, n);
            let q_commit = ck.try_commit(&batched_q)?;

            let z = loop {
                let z = Fr::rand(&mut rng);
                if validate_shplonk_z_against_claims(z, &claims, "test").is_ok() {
                    break z;
                }
            };

            let g_coeffs = compute_shplonk_g_coeffs(&claims, &a_polys, &batched_q, nu, z, n)?;
            let g_direct_commit = ck.try_commit(&g_coeffs)?;

            let g_msm_commit =
                compute_shplonk_g_commit::<E>(&claims, &q_commit, &com, &fold_comms, nu, z, &vk.g)?;

            assert_eq!(
                g_direct_commit, g_msm_commit,
                "direct Commit(G) must equal verifier-reduction [G] for nv={nv}"
            );

            let witness_coeffs = kzg_quotient(&g_coeffs, z, Fr::zero());
            let witness = ck.try_commit(&witness_coeffs)?;
            assert!(
                kzg_verify_pairing(&vk, &g_msm_commit, z, Fr::zero(), &witness)?,
                "KZG pairing must verify G(z)=0 for nv={nv}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_shplonk_g_commit_differs_with_tampered_input() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let mu = nv;
        let n = 1 << nv;
        let f_hat = p.to_evaluations();
        let com = ck.try_commit(&f_hat)?;

        let mut a_polys: Vec<Vec<Fr>> = Vec::with_capacity(mu);
        a_polys.push(f_hat.clone());
        for i in 0..mu.saturating_sub(1) {
            a_polys.push(fold_polynomial(&a_polys[i], pt[i]));
        }
        let mut fold_comms: Vec<_> = vec![];
        for a in a_polys.iter().skip(1) {
            fold_comms.push(ck.try_commit(a)?);
        }

        let c0 = a_polys[mu.saturating_sub(1)][0];
        let c1 = a_polys[mu.saturating_sub(1)][1];

        let r = Fr::rand(&mut rng);
        let mut fold_evals: Vec<(Fr, Fr)> = vec![];
        let mut rp = r;
        for i in 0..mu.saturating_sub(1) {
            fold_evals.push((poly_eval(&a_polys[i], rp), poly_eval(&a_polys[i], -rp)));
            rp = rp * rp;
        }

        let claims = build_gemini_claims(mu, r, &fold_evals, c0, c1);
        let nu = Fr::rand(&mut rng);
        let batched_q = compute_shplonk_batched_quotient(&claims, &a_polys, nu, n);
        let q_commit = ck.try_commit(&batched_q)?;

        let z = loop {
            let z = Fr::rand(&mut rng);
            if validate_shplonk_z_against_claims(z, &claims, "test").is_ok() {
                break z;
            }
        };

        let g_commit =
            compute_shplonk_g_commit::<E>(&claims, &q_commit, &com, &fold_comms, nu, z, &vk.g)?;

        // Tamper each input and expect a different G commit
        let tampered_q = (q_commit.into_group() * Fr::from(2u64)).into_affine();
        let g_tampered_q =
            compute_shplonk_g_commit::<E>(&claims, &tampered_q, &com, &fold_comms, nu, z, &vk.g)?;
        assert_ne!(g_commit, g_tampered_q, "tampered Q commit must differ");

        let tampered_c0 = c0 + Fr::one();
        let mut claims_tampered_eval = claims.clone();
        claims_tampered_eval.last_mut().unwrap().evaluation = tampered_c0;
        let g_tampered_eval = compute_shplonk_g_commit::<E>(
            &claims_tampered_eval,
            &q_commit,
            &com,
            &fold_comms,
            nu,
            z,
            &vk.g,
        )?;
        assert_ne!(g_commit, g_tampered_eval, "tampered evaluation must differ");

        Ok(())
    }

    // ── Transcript binding tests (all call production
    // append_gemini_claims_to_transcript) ──

    fn build_test_transcript_and_derive_nu(
        claims: &[GeminiClaim<Fr>],
        orig_cm: &<E as Pairing>::G1Affine,
        fold_cms: &[<E as Pairing>::G1Affine],
    ) -> Fr {
        let mut t1 = IOPTranscript::<Fr>::new(b"gemini-open");
        t1.append_field_element(b"mu", &Fr::from(claims.len() as u64))
            .unwrap();
        append_gemini_claims_to_transcript::<E>(claims, orig_cm, fold_cms, &mut t1).unwrap();
        t1.get_and_append_challenge_vectors(b"Shplonk:nu", 1)
            .unwrap()[0]
    }

    #[test]
    fn test_transcript_nu_changes_on_positive_eval() {
        let mut rng = test_rng();
        let c0 = Fr::rand(&mut rng);
        let c1 = Fr::rand(&mut rng);
        let r = Fr::rand(&mut rng);
        let fe = vec![(Fr::rand(&mut rng), Fr::rand(&mut rng))];
        let claims = build_gemini_claims(2, r, &fe, c0, c1);
        let g = <E as Pairing>::G1Affine::generator();
        let _gs = vec![g, g];

        let nu_a = build_test_transcript_and_derive_nu(&claims, &g, &[g]);
        let mut claims_b = claims.clone();
        claims_b[0].evaluation += Fr::one();
        let nu_b = build_test_transcript_and_derive_nu(&claims_b, &g, &[g]);
        assert_ne!(nu_a, nu_b, "changing positive eval must change nu");
    }

    #[test]
    fn test_transcript_nu_changes_on_negative_eval() {
        let mut rng = test_rng();
        let c0 = Fr::rand(&mut rng);
        let c1 = Fr::rand(&mut rng);
        let r = Fr::rand(&mut rng);
        let fe = vec![(Fr::rand(&mut rng), Fr::rand(&mut rng))];
        let claims = build_gemini_claims(2, r, &fe, c0, c1);
        let g = <E as Pairing>::G1Affine::generator();

        let nu_a = build_test_transcript_and_derive_nu(&claims, &g, &[g]);
        let mut claims_b = claims.clone();
        claims_b[1].evaluation += Fr::one();
        let nu_b = build_test_transcript_and_derive_nu(&claims_b, &g, &[g]);
        assert_ne!(nu_a, nu_b, "changing negative eval must change nu");
    }

    #[test]
    fn test_transcript_nu_changes_on_c0() {
        let mut rng = test_rng();
        let c0 = Fr::rand(&mut rng);
        let c1 = Fr::rand(&mut rng);
        let r = Fr::rand(&mut rng);
        let fe = vec![(Fr::rand(&mut rng), Fr::rand(&mut rng))];
        let claims = build_gemini_claims(2, r, &fe, c0, c1);
        let g = <E as Pairing>::G1Affine::generator();

        let nu_a = build_test_transcript_and_derive_nu(&claims, &g, &[g]);
        let mut claims_b = claims.clone();
        claims_b[2].evaluation += Fr::one();
        let nu_b = build_test_transcript_and_derive_nu(&claims_b, &g, &[g]);
        assert_ne!(nu_a, nu_b, "changing c0 must change nu");
    }

    #[test]
    fn test_transcript_nu_changes_on_c0_plus_c1() {
        let mut rng = test_rng();
        let c0 = Fr::rand(&mut rng);
        let c1 = Fr::rand(&mut rng);
        let r = Fr::rand(&mut rng);
        let fe = vec![(Fr::rand(&mut rng), Fr::rand(&mut rng))];
        let claims = build_gemini_claims(2, r, &fe, c0, c1);
        let g = <E as Pairing>::G1Affine::generator();

        let nu_a = build_test_transcript_and_derive_nu(&claims, &g, &[g]);
        let mut claims_b = claims.clone();
        claims_b[3].evaluation += Fr::one();
        let nu_b = build_test_transcript_and_derive_nu(&claims_b, &g, &[g]);
        assert_ne!(nu_a, nu_b, "changing c0+c1 must change nu");
    }

    #[test]
    fn test_transcript_nu_changes_on_fold_commit() {
        let mut rng = test_rng();
        let c0 = Fr::rand(&mut rng);
        let c1 = Fr::rand(&mut rng);
        let r = Fr::rand(&mut rng);
        let fe = vec![(Fr::rand(&mut rng), Fr::rand(&mut rng))];
        let claims = build_gemini_claims(2, r, &fe, c0, c1);
        let g = <E as Pairing>::G1Affine::generator();

        let nu_a = build_test_transcript_and_derive_nu(&claims, &g, &[g]);
        let tampered_fc = (g.into_group() * Fr::from(2u64)).into_affine();
        let nu_b = build_test_transcript_and_derive_nu(&claims, &g, &[tampered_fc]);
        assert_ne!(nu_a, nu_b, "changing fold commit must change nu");
    }

    #[test]
    fn test_transcript_nu_changes_on_original_commitment() {
        let mut rng = test_rng();
        let c0 = Fr::rand(&mut rng);
        let c1 = Fr::rand(&mut rng);
        let r = Fr::rand(&mut rng);
        let fe = vec![(Fr::rand(&mut rng), Fr::rand(&mut rng))];
        let claims = build_gemini_claims(2, r, &fe, c0, c1);
        let g = <E as Pairing>::G1Affine::generator();

        let nu_a = build_test_transcript_and_derive_nu(&claims, &g, &[g]);
        let tampered_orig = (g.into_group() * Fr::from(3u64)).into_affine();
        let nu_b = build_test_transcript_and_derive_nu(&claims, &tampered_orig, &[g]);
        assert_ne!(nu_a, nu_b, "changing original commitment must change nu");
    }

    #[test]
    fn test_transcript_nu_changes_on_claim_point() {
        let mut rng = test_rng();
        let c0 = Fr::rand(&mut rng);
        let c1 = Fr::rand(&mut rng);
        let r = Fr::rand(&mut rng);
        let fe = vec![(Fr::rand(&mut rng), Fr::rand(&mut rng))];
        let claims = build_gemini_claims(2, r, &fe, c0, c1);
        let g = <E as Pairing>::G1Affine::generator();

        let nu_a = build_test_transcript_and_derive_nu(&claims, &g, &[g]);
        let mut claims_b = claims.clone();
        claims_b[2].point += Fr::one();
        let nu_b = build_test_transcript_and_derive_nu(&claims_b, &g, &[g]);
        assert_ne!(nu_a, nu_b, "changing claim point must change nu");
    }

    #[test]
    fn test_transcript_nu_changes_on_claim_idx() {
        let mut rng = test_rng();
        let c0 = Fr::rand(&mut rng);
        let c1 = Fr::rand(&mut rng);
        let r = Fr::rand(&mut rng);
        let fe = vec![(Fr::rand(&mut rng), Fr::rand(&mut rng))];
        let mut claims = build_gemini_claims(2, r, &fe, c0, c1);
        let g = <E as Pairing>::G1Affine::generator();

        let nu_a = build_test_transcript_and_derive_nu(&claims, &g, &[g]);
        claims.swap(0, 1);
        let nu_b = build_test_transcript_and_derive_nu(&claims, &g, &[g]);
        assert_ne!(nu_a, nu_b, "changing claim order must change nu");
    }

    // ── Challenge validation helpers (call production validate_* functions) ──

    #[test]
    fn test_validate_shplonk_r_rejects_zero() {
        assert!(validate_shplonk_r(Fr::zero(), "test").is_err());
    }

    #[test]
    fn test_validate_shplonk_r_accepts_nonzero() {
        let mut rng = test_rng();
        let r = loop {
            let r = Fr::rand(&mut rng);
            if !r.is_zero() {
                break r;
            }
        };
        assert!(validate_shplonk_r(r, "test").is_ok());
    }

    #[test]
    fn test_validate_shplonk_nu_rejects_zero() {
        assert!(validate_shplonk_nu(Fr::zero(), "test").is_err());
    }

    #[test]
    fn test_validate_shplonk_nu_accepts_nonzero() {
        let mut rng = test_rng();
        let nu = loop {
            let nu = Fr::rand(&mut rng);
            if !nu.is_zero() {
                break nu;
            }
        };
        assert!(validate_shplonk_nu(nu, "test").is_ok());
    }

    #[test]
    fn test_validate_shplonk_z_rejects_zero() {
        let mut rng = test_rng();
        let claims = build_gemini_claims(
            2,
            Fr::rand(&mut rng),
            &[(Fr::rand(&mut rng), Fr::rand(&mut rng))],
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
        );
        assert!(validate_shplonk_z_against_claims(Fr::zero(), &claims, "test").is_err());
    }

    #[test]
    fn test_validate_shplonk_z_rejects_collision_with_fold_point() {
        let mut rng = test_rng();
        let r = Fr::rand(&mut rng);
        let mut rp = r;
        for _ in 0..3 {
            let claims = build_gemini_claims(
                4,
                r,
                &[
                    (Fr::rand(&mut rng), Fr::rand(&mut rng)),
                    (Fr::rand(&mut rng), Fr::rand(&mut rng)),
                    (Fr::rand(&mut rng), Fr::rand(&mut rng)),
                ],
                Fr::rand(&mut rng),
                Fr::rand(&mut rng),
            );
            // rp is a claim point (r or some r^{2^i})
            assert!(
                validate_shplonk_z_against_claims(rp, &claims, "test").is_err(),
                "z collides with positive claim point"
            );
            assert!(
                validate_shplonk_z_against_claims(-rp, &claims, "test").is_err(),
                "z collides with negative claim point"
            );
            rp = rp * rp;
        }
    }

    #[test]
    fn test_validate_shplonk_z_rejects_collision_with_final_0() {
        let mut rng = test_rng();
        let claims = build_gemini_claims(
            2,
            Fr::rand(&mut rng),
            &[(Fr::rand(&mut rng), Fr::rand(&mut rng))],
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
        );
        assert!(
            validate_shplonk_z_against_claims(Fr::zero(), &claims, "test").is_err(),
            "z=0 must be rejected"
        );
    }

    #[test]
    fn test_validate_shplonk_z_rejects_collision_with_final_1() {
        let mut rng = test_rng();
        let claims = build_gemini_claims(
            2,
            Fr::rand(&mut rng),
            &[(Fr::rand(&mut rng), Fr::rand(&mut rng))],
            Fr::rand(&mut rng),
            Fr::rand(&mut rng),
        );
        assert!(
            validate_shplonk_z_against_claims(Fr::one(), &claims, "test").is_err(),
            "z=1 must be rejected"
        );
    }

    #[test]
    fn test_validate_shplonk_z_accepts_non_colliding() {
        let mut rng = test_rng();
        for nv in [1usize, 2, 4] {
            let fe: Vec<(Fr, Fr)> = (0..nv.saturating_sub(1))
                .map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)))
                .collect();
            let claims = build_gemini_claims(
                nv,
                Fr::rand(&mut rng),
                &fe,
                Fr::rand(&mut rng),
                Fr::rand(&mut rng),
            );
            let z = loop {
                let z = Fr::rand(&mut rng);
                if validate_shplonk_z_against_claims(z, &claims, "test").is_ok() {
                    break z;
                }
            };
            assert!(
                validate_shplonk_z_against_claims(z, &claims, "test").is_ok(),
                "valid z must be accepted for nv={nv}"
            );
        }
    }

    // ── Shplonk Q commit tamper must reject (production path) ──

    #[test]
    fn test_gemini_reject_tampered_q_commit() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt)?;
        proof.shplonk_q_commit =
            (proof.shplonk_q_commit.into_group() * Fr::from(2u64)).into_affine();
        assert!(!GeminiPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    // ── open_with_commitment ──

    #[test]
    fn test_open_with_commitment_matches_trait_open() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [4usize, 6, 8] {
            let (ck, vk) = setup(nv);
            let p = rpoly(nv, &mut rng);
            let pt = rpt(nv, &mut rng);
            let com = GeminiPCS::<E>::commit(&ck, &p)?;
            let (proof_a, val_a) = GeminiPCS::<E>::open_with_commitment(&ck, &p, &pt, &com)?;
            let (proof_b, val_b) = GeminiPCS::<E>::open(&ck, &p, &pt)?;
            assert_eq!(val_a, val_b);
            assert!(GeminiPCS::<E>::verify(&vk, &com, &pt, &val_a, &proof_a)?);
            assert!(GeminiPCS::<E>::verify(&vk, &com, &pt, &val_b, &proof_b)?);
        }
        Ok(())
    }

    #[test]
    fn test_open_with_commitment_valid_proof_accepted() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [4usize, 6, 8, 10] {
            let (ck, vk) = setup(nv);
            let p = rpoly(nv, &mut rng);
            let pt = rpt(nv, &mut rng);
            let com = GeminiPCS::<E>::commit(&ck, &p)?;
            let (proof, val) = GeminiPCS::<E>::open_with_commitment(&ck, &p, &pt, &com)?;
            assert!(GeminiPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        }
        Ok(())
    }

    #[test]
    fn test_open_with_commitment_rejects_wrong_commitment() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [4usize, 8] {
            let (ck, vk) = setup(nv);
            let p = rpoly(nv, &mut rng);
            let p2 = rpoly(nv, &mut rng);
            let pt = rpt(nv, &mut rng);
            let wrong_com = GeminiPCS::<E>::commit(&ck, &p2)?;
            let r = GeminiPCS::<E>::open_with_commitment(&ck, &p, &pt, &wrong_com);
            if let Ok((proof, val)) = r {
                let com = GeminiPCS::<E>::commit(&ck, &p)?;
                assert!(
                    !GeminiPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?,
                    "wrong commitment should not produce verifiable proof"
                );
                assert!(
                    !GeminiPCS::<E>::verify(&vk, &wrong_com, &pt, &val, &proof)?,
                    "proof under the supplied wrong commitment should not verify"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_open_with_commitment_wrong_point_len_no_panic() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, _) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let _pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p)?;
        let short = rpt(2, &mut rng);
        assert!(GeminiPCS::<E>::open_with_commitment(&ck, &p, &short, &com).is_err());
        let long = rpt(8, &mut rng);
        assert!(GeminiPCS::<E>::open_with_commitment(&ck, &p, &long, &com).is_err());
        Ok(())
    }

    fn assert_rejects(r: Result<bool, PCSError>) {
        match r {
            Ok(true) => panic!("expected reject"),
            Ok(false) => {},
            Err(_) => {},
        }
    }
}
