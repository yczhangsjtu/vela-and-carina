//! Gemini PCS — multilinear PCS based on KZG univariate commitments.
//!
//! Converts multilinear polynomial evaluations to univariate coefficient form,
//! then commits using KZG G1 MSM. The opening protocol uses the Gemini
//! transformation to reduce multilinear openings to univariate KZG proofs.

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
    pub all_kzg_proofs: Vec<E::G1Affine>,
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

fn kzg_prove_coeffs<E: Pairing>(
    pp: &GeminiProverParam<E>,
    coeffs: &[E::ScalarField],
    point: E::ScalarField,
) -> Result<(E::ScalarField, E::G1Affine), PCSError> {
    let eval = poly_eval(coeffs, point);
    let mut div = vec![E::ScalarField::zero(); coeffs.len()];
    for (i, &c) in coeffs.iter().enumerate() {
        div[i] = c;
    }
    div[0] -= eval;
    let n = div.len();
    let mut q = vec![E::ScalarField::zero(); n - 1];
    let mut carry = E::ScalarField::zero();
    for i in (1..n).rev() {
        let term = div[i] + carry;
        q[i - 1] = term;
        carry = term * point;
    }
    let proof = pp.try_commit(&q)?;
    Ok((eval, proof))
}

// ═══════════════════════════════════════════════════════════════════
// Transcript-aware single opening
// ═══════════════════════════════════════════════════════════════════

fn gemini_open_with_transcript<E: Pairing>(
    pp: &GeminiProverParam<E>,
    poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
    point: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<(GeminiProof<E>, E::ScalarField), PCSError> {
    let mu = poly.num_vars();
    let n = checked_domain_size_from_mu(mu, "open")
        .map_err(|e| PCSError::InvalidParameters(e.to_string()))?;

    if point.len() != mu {
        return Err(PCSError::InvalidParameters(
            "point length mismatch".to_string(),
        ));
    }

    let _t_total = ScopedTimer::new(BACKEND, mu, n, "gemini_open_total", 1, "total");

    let f_hat = poly.to_evaluations();
    let eval = poly
        .evaluate(point)
        .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;

    let a0_commit = pp.try_commit(&f_hat)?;
    transcript.append_serializable_element(b"commitment", &a0_commit)?;
    transcript.append_serializable_element(b"point", &point.to_vec())?;
    transcript.append_field_element(b"eval", &eval)?;

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
    for i in 0..mu - 1 {
        let next = fold_polynomial(&a_polys[i], point[i]);
        a_polys.push(next);
    }
    drop(_t_fold);

    let _t_cm = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_open_commit_folds",
        a_polys.len() - 1,
        "KZG-commit-fold",
    );
    let mut fold_comms: Vec<E::G1Affine> = Vec::with_capacity(mu - 1);
    for a in a_polys.iter().skip(1) {
        fold_comms.push(pp.try_commit(a)?);
    }
    drop(_t_cm);

    for cm in &fold_comms {
        transcript.append_serializable_element(b"fold_comm", cm)?;
    }

    let r = transcript.get_and_append_challenge_vectors(b"r", 1)?[0];

    let _t_evals = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_open_compute_evals",
        a_polys.len(),
        "eval-rounds",
    );
    let mut fold_evals: Vec<(E::ScalarField, E::ScalarField)> = Vec::with_capacity(mu - 1);
    let mut r_pow = r;
    for i in 0..mu - 1 {
        let a_i_r = poly_eval(&a_polys[i], r_pow);
        let a_i_neg_r = poly_eval(&a_polys[i], -r_pow);
        fold_evals.push((a_i_r, a_i_neg_r));
        r_pow = r_pow * r_pow;
    }
    let c0 = a_polys[mu - 1][0];
    let c1 = a_polys[mu - 1][1];
    let one = E::ScalarField::one();
    let computed_eval = (one - point[mu - 1]) * c0 + point[mu - 1] * c1;
    if computed_eval != eval {
        return Err(PCSError::InvalidProver(
            "fold check failed: eval mismatch".to_string(),
        ));
    }
    drop(_t_evals);

    let _t_kzg = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_open_kzg_proofs",
        2 * (mu - 1) + 2,
        "KZG-proofs",
    );
    let mut all_kzg_proofs: Vec<E::G1Affine> = Vec::with_capacity(2 * (mu - 1) + 2);
    let mut r_pow = r;
    for i in 0..mu - 1 {
        let (_, pi_r) = kzg_prove_coeffs(pp, &a_polys[i], r_pow)?;
        all_kzg_proofs.push(pi_r);
        let (_, pi_neg_r) = kzg_prove_coeffs(pp, &a_polys[i], -r_pow)?;
        all_kzg_proofs.push(pi_neg_r);
        r_pow = r_pow * r_pow;
    }
    let zero = E::ScalarField::zero();
    let (c0_val, pi_c0) = kzg_prove_coeffs(pp, &a_polys[mu - 1], zero)?;
    let (c0_plus_c1, pi_c1) = kzg_prove_coeffs(pp, &a_polys[mu - 1], one)?;
    if c0_val != c0 {
        return Err(PCSError::InvalidProver("c0 KZG eval mismatch".to_string()));
    }
    if c0_plus_c1 != c0 + c1 {
        return Err(PCSError::InvalidProver("c1 KZG eval mismatch".to_string()));
    }
    all_kzg_proofs.push(pi_c0);
    all_kzg_proofs.push(pi_c1);
    drop(_t_kzg);

    let proof = GeminiProof {
        fold_comms,
        fold_evals,
        final_coeffs: (c0, c1),
        all_kzg_proofs,
        mu,
    };
    drop(_t_total);
    Ok((proof, eval))
}

// ═══════════════════════════════════════════════════════════════════
// Transcript-aware single verification
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

    transcript.append_serializable_element(b"commitment", &commitment.0)?;
    transcript.append_serializable_element(b"point", &point.to_vec())?;
    transcript.append_field_element(b"eval", value)?;

    for cm in &proof.fold_comms {
        transcript.append_serializable_element(b"fold_comm", cm)?;
    }

    let r = transcript.get_and_append_challenge_vectors(b"r", 1)?[0];

    if proof.fold_evals.len() != mu - 1 {
        return Err(PCSError::InvalidProof(format!(
            "verify: fold_evals length {} != expected {}",
            proof.fold_evals.len(),
            mu - 1
        )));
    }

    if proof.all_kzg_proofs.len() != 2 * (mu - 1) + 2 {
        return Err(PCSError::InvalidProof(format!(
            "verify: kzg proofs length {} != expected {}",
            proof.all_kzg_proofs.len(),
            2 * (mu - 1) + 2
        )));
    }

    let _t_fold_check = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_verify_fold_checks",
        mu - 1,
        "fold-equations",
    );
    let mut r_pow = r;
    for i in 0..mu - 1 {
        let (a_i_r, a_i_neg_r) = proof.fold_evals[i];
        let a_next_r_sq = if i + 1 < mu - 1 {
            // A_{i+1}(r_{i+1}) = A_{i+1}(r_i^2) from the next fold-eval
            proof.fold_evals[i + 1].0
        } else {
            // Last round: compute A_{mu-1}(r_{mu-2}^2) from coefficients
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

    let _t_kzg = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "gemini_verify_kzg_checks",
        proof.all_kzg_proofs.len(),
        "pairings",
    );

    let mut r_pow = r;
    for i in 0..mu - 1 {
        let pi_offset = 2 * i;

        let (a_i_r, a_i_neg_r) = proof.fold_evals[i];

        let com_i = if i == 0 {
            commitment.0
        } else {
            proof.fold_comms[i - 1]
        };

        let pi_r_proof = proof.all_kzg_proofs[pi_offset];
        let pi_neg_r_proof = proof.all_kzg_proofs[pi_offset + 1];

        if !kzg_verify_pairing(vp, &com_i, r_pow, a_i_r, &pi_r_proof)? {
            return Ok(false);
        }
        if !kzg_verify_pairing(vp, &com_i, -r_pow, a_i_neg_r, &pi_neg_r_proof)? {
            return Ok(false);
        }

        r_pow = r_pow * r_pow;
    }

    let (c0, c1) = proof.final_coeffs;
    let expected_eval = (E::ScalarField::one() - point[mu - 1]) * c0 + point[mu - 1] * c1;
    if expected_eval != *value {
        return Ok(false);
    }

    let final_com = if mu == 1 {
        commitment.0
    } else {
        proof.fold_comms[mu - 2]
    };

    let zero = E::ScalarField::zero();
    let one = E::ScalarField::one();
    let pi_c0 = proof.all_kzg_proofs[2 * (mu - 1)];
    let pi_c1 = proof.all_kzg_proofs[2 * (mu - 1) + 1];

    if !kzg_verify_pairing(vp, &final_com, zero, c0, &pi_c0)? {
        return Ok(false);
    }
    if !kzg_verify_pairing(vp, &final_com, one, c0 + c1, &pi_c1)? {
        return Ok(false);
    }

    drop(_t_kzg);
    drop(_t_total);

    Ok(true)
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
    fn test_gemini_reject_tampered_kzg_proof() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = GeminiPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = GeminiPCS::<E>::open(&ck, &p, &pt)?;
        proof.all_kzg_proofs[0] =
            (proof.all_kzg_proofs[0].into_group() * Fr::from(3u64)).into_affine();
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

    // ── Minimal KZG proof count ──

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
                proof.all_kzg_proofs.len(),
                2 * (nv - 1) + 2,
                "nv={nv}: expected {} KZG proofs, got {}",
                2 * (nv - 1) + 2,
                proof.all_kzg_proofs.len()
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

    fn assert_rejects(r: Result<bool, PCSError>) {
        match r {
            Ok(true) => panic!("expected reject"),
            Ok(false) => {},
            Err(_) => {},
        }
    }
}
