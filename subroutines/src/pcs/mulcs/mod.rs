//! Mulcs PCS — Claymore identity-based multilinear PCS with sumcheck batching.
//!
//! Batch opening: Mulcs-specific sumcheck batching reduces multiple openings
//! to a single opening of an aggregated polynomial g'.

use crate::{
    pcs::{
        multilinear_kzg::batching::BatchProof,
        prelude::{Commitment, PCSError},
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

use crate::pcs::profile;
pub(crate) mod srs;
mod util;

use srs::{MulcsProverParam, MulcsUniversalParams, MulcsVerifierParam};
// The symmetric construction was moved to the standalone ReciPCS module.
// Keep these legacy convenience aliases; they no longer select a second
// implementation or a separate benchmark backend.
pub use crate::pcs::recipcs::{ReciPCS as MulcsSymmetricPCS, ReciProof as MulcsSymmetricProof};
pub(crate) use util::UnivarPoly;

pub struct MulcsPCS<E: Pairing> {
    phantom: PhantomData<E>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct MulcsProof<E: Pairing> {
    pub cm_hbar: E::G1Affine,
    pub y_f: E::ScalarField,
    pub y_f_prime: E::ScalarField,
    pub y_hbar: E::ScalarField,
    pub y_hbar_prime: E::ScalarField,
    pub z: E::ScalarField,
    pub pi: E::G1Affine,
    pub rf: Vec<E::ScalarField>,
    pub rh: Vec<E::ScalarField>,
    pub mu: usize,
}

impl<E: Pairing> PolynomialCommitmentScheme<E> for MulcsPCS<E> {
    type ProverParam = MulcsProverParam<E>;
    type VerifierParam = MulcsVerifierParam<E>;
    type SRS = MulcsUniversalParams<E>;
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    type Commitment = Commitment<E>;
    type Proof = MulcsProof<E>;
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
        srs.borrow().trim(2 * (1 << nv))
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
        let _t = profile::ScopedTimer::new("Mulcs", nv, n, "commit_to_evals", n, "to_evaluations");
        let scalars = poly.to_evaluations();
        drop(_t);
        let _t = profile::ScopedTimer::new("Mulcs", nv, n, "commit_msm", scalars.len(), "KZG-MSM");
        let cm = pp.commit(&scalars);
        drop(_t);
        Ok(Commitment(cm))
    }

    fn open(
        pp: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let mut t = IOPTranscript::new(b"mulcs-open");
        t.append_field_element(b"mu", &E::ScalarField::from(poly.num_vars as u64))?;
        open_with_transcript(pp.borrow(), poly, point, &mut t)
    }

    fn multi_open(
        pp: impl Borrow<Self::ProverParam>,
        polys: &[Self::Polynomial],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Self::BatchProof, PCSError> {
        mulcs_sumcheck_multi_open(pp.borrow(), polys, points, evals, transcript)
    }

    fn verify(
        vp: &Self::VerifierParam,
        com: &Self::Commitment,
        point: &Self::Point,
        val: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        let mut t = IOPTranscript::new(b"mulcs-open");
        t.append_field_element(b"mu", &E::ScalarField::from(proof.mu as u64))?;
        verify_with_transcript(vp, com, point, val, proof, &mut t)
    }

    fn batch_verify(
        vp: &Self::VerifierParam,
        coms: &[Self::Commitment],
        points: &[Self::Point],
        bp: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        mulcs_sumcheck_batch_verify(vp, coms, points, bp, transcript)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Verifier safety helper
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
// Transcript-aware single opening (profiled)
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn open_with_transcript<E: Pairing>(
    pp: &MulcsProverParam<E>,
    polynomial: &Arc<DenseMultilinearExtension<E::ScalarField>>,
    point: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<(MulcsProof<E>, E::ScalarField), PCSError> {
    let nv = polynomial.num_vars();
    let n = 1 << nv;
    let mu = nv;
    let gamma = pp.gamma;

    let _t_total = profile::ScopedTimer::new("Mulcs", mu, n, "mulcs_open_total", 1, "total");

    let _t_evals =
        profile::ScopedTimer::new("Mulcs", mu, n, "mulcs_open_to_evals", n, "to_evaluations");
    let coeffs = polynomial.to_evaluations();
    let f_v = UnivarPoly::new(coeffs.clone());
    let y = polynomial
        .evaluate(point)
        .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;
    drop(_t_evals);

    let _t_h = profile::ScopedTimer::new("Mulcs", mu, n, "mulcs_open_compute_h", 1, "Claymore-h");
    let h = UnivarPoly::compute_h(&coeffs, mu, point, y);
    drop(_t_h);

    let _t_delta =
        profile::ScopedTimer::new("Mulcs", mu, n, "mulcs_open_derive_delta", 1, "FS-challenge");
    let delta_buf = transcript.get_and_append_challenge_vectors(b"mulcs_delta", 1)?;
    let delta = delta_buf[0];
    drop(_t_delta);

    let _t_hbar = profile::ScopedTimer::new(
        "Mulcs",
        mu,
        n,
        "mulcs_open_compute_h_bar",
        1,
        "Claymore-hbar",
    );
    let h_bar = UnivarPoly::compute_h_bar(&h, gamma, n, delta);
    drop(_t_hbar);

    let _t_cm_hbar = profile::ScopedTimer::new(
        "Mulcs",
        mu,
        n,
        "mulcs_open_commit_hbar",
        h_bar.coeffs.len(),
        "KZG-commit-hbar",
    );
    let cm_hbar = pp.commit(&h_bar.coeffs);
    drop(_t_cm_hbar);

    transcript.append_serializable_element(b"cm_hbar", &cm_hbar)?;

    let _t_z = profile::ScopedTimer::new("Mulcs", mu, n, "mulcs_open_derive_z", 1, "FS-challenge");
    let z_buf = transcript.get_and_append_challenge_vectors(b"mulcs_z", 1)?;
    let z = z_buf[0];
    drop(_t_z);

    let _t_evals_z =
        profile::ScopedTimer::new("Mulcs", mu, n, "mulcs_open_eval_at_z", 4, "Horner-evals");
    let gz = gamma * z;
    let y_f = f_v.evaluate(z);
    let y_f_prime = f_v.evaluate(gz);
    let y_hbar = h_bar.evaluate(z);
    let y_hbar_prime = h_bar.evaluate(gz);
    drop(_t_evals_z);

    let (pi, rf, rh) = mulcs_batch_kzg_open_profiled(
        pp,
        &f_v,
        &h_bar,
        z,
        gamma,
        y_f,
        y_f_prime,
        y_hbar,
        y_hbar_prime,
        mu,
        n,
    );

    let proof = MulcsProof {
        cm_hbar,
        y_f,
        y_f_prime,
        y_hbar,
        y_hbar_prime,
        z,
        pi,
        rf,
        rh,
        mu,
    };
    drop(_t_total);
    Ok((proof, y))
}

/// Profiled version of mulcs_batch_kzg_open with nv/n for accurate CSV.
fn mulcs_batch_kzg_open_profiled<E: Pairing>(
    pp: &MulcsProverParam<E>,
    f_v: &UnivarPoly<E::ScalarField>,
    h_bar: &UnivarPoly<E::ScalarField>,
    z: E::ScalarField,
    gamma: E::ScalarField,
    y_f: E::ScalarField,
    y_f_prime: E::ScalarField,
    y_h: E::ScalarField,
    y_h_prime: E::ScalarField,
    mu: usize,
    n: usize,
) -> (E::G1Affine, Vec<E::ScalarField>, Vec<E::ScalarField>) {
    let gz = gamma * z;
    let _t_quot = profile::ScopedTimer::new(
        "Mulcs",
        mu,
        n,
        "mulcs_open_build_quotients",
        1,
        "lagrange+poly-div",
    );
    let f_pts = [(z, y_f), (gz, y_f_prime)];
    let h_pts = [(z, y_h), (gz, y_h_prime)];
    let (rf, z_coeffs) = build_multi_point_polys(&f_pts);
    let (rh, _) = build_multi_point_polys(&h_pts);
    let qf = poly_sub_div(&f_v.coeffs, &rf, &z_coeffs);
    let qh = poly_sub_div(&h_bar.coeffs, &rh, &z_coeffs);
    let max_deg = qf.len().max(qh.len());
    let mut q_comb = vec![E::ScalarField::zero(); max_deg];
    for i in 0..qf.len() {
        q_comb[i] += qf[i];
    }
    for i in 0..qh.len() {
        q_comb[i] += qh[i];
    }
    drop(_t_quot);

    let _t_pi = profile::ScopedTimer::new(
        "Mulcs",
        mu,
        n,
        "mulcs_open_commit_pi",
        max_deg + 1,
        "KZG-commit-pi",
    );
    let pi = pp.commit(&q_comb);
    drop(_t_pi);
    (pi, rf, rh)
}

// ═══════════════════════════════════════════════════════════════════
// Transcript-aware single verification (profiled)
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn verify_with_transcript<E: Pairing>(
    vp: &MulcsVerifierParam<E>,
    commitment: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &MulcsProof<E>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError> {
    let mu = proof.mu;
    let n = checked_domain_size_from_mu(mu, "verify")?;

    if point.len() != mu {
        return Err(PCSError::InvalidProof(format!(
            "verify: point length {} != proof.mu {}",
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

    let gamma = vp.gamma;

    let _t_total = profile::ScopedTimer::new("Mulcs", mu, n, "mulcs_verify_total", 1, "total");

    let delta_buf = transcript.get_and_append_challenge_vectors(b"mulcs_delta", 1)?;
    let _delta = delta_buf[0];
    transcript.append_serializable_element(b"cm_hbar", &proof.cm_hbar)?;
    let z_buf = transcript.get_and_append_challenge_vectors(b"mulcs_z", 1)?;
    let z = z_buf[0];
    if z != proof.z {
        return Ok(false);
    }

    let _t_pair = profile::ScopedTimer::new("Mulcs", mu, n, "mulcs_verify_pairing", 1, "1-pairing");
    let gz = gamma * z;
    let rf = two_point_remainder(z, proof.y_f, gz, proof.y_f_prime)?;
    let rh = two_point_remainder(z, proof.y_hbar, gz, proof.y_hbar_prime)?;
    let r0 = rf[0] + rh[0];
    let r1 = rf[1] + rh[1];
    let cm_r = vp.g1_one.into_group() * r0 + vp.g1_x.into_group() * r1;
    let cm_comb = commitment.0.into_group() + proof.cm_hbar.into_group() - cm_r;
    let s = z + gz;
    let p = z * gz;
    let zx_g2 = vp.g2_x2.into_group() - vp.g2_x.into_group() * s + vp.g2_one.into_group() * p;
    let neg_pi = (-proof.pi.into_group()).into_affine();
    let ok = E::multi_pairing(
        [cm_comb.into_affine(), neg_pi],
        [vp.g2_one, zx_g2.into_affine()],
    ) == PairingOutput(E::TargetField::one());
    drop(_t_pair);
    if !ok {
        return Ok(false);
    }

    let _t_clay = profile::ScopedTimer::new("Mulcs", mu, n, "mulcs_verify_claymore", 1, "claymore");
    let result = check_claymore_identity(
        gamma,
        n,
        z,
        proof.y_f,
        proof.y_hbar,
        proof.y_hbar_prime,
        point,
        *value,
    );
    drop(_t_clay);
    drop(_t_total);
    result
}

// ═══════════════════════════════════════════════════════════════════
// Sumcheck batch open (profiled)
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn mulcs_sumcheck_multi_open<E: Pairing>(
    pp: &MulcsProverParam<E>,
    polynomials: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    points: &[Vec<E::ScalarField>],
    evals: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<BatchProof<E, MulcsPCS<E>>, PCSError> {
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
        profile::ScopedTimer::new("Mulcs", num_var, n, "mulcs_multi_open_total", k, "total");

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
        "Mulcs",
        num_var,
        n,
        "mulcs_multi_open_transcript_absorb",
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
        "Mulcs",
        num_var,
        n,
        "mulcs_multi_open_build_eq_t",
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
        "Mulcs",
        num_var,
        n,
        "mulcs_multi_open_group_points",
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
        "Mulcs",
        num_var,
        n,
        "mulcs_multi_open_merge_polys",
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
        "Mulcs",
        num_var,
        n,
        "mulcs_multi_open_build_tilde_eqs",
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
        "Mulcs",
        num_var,
        n,
        "mulcs_multi_open_sumcheck_prove",
        num_var,
        "sumcheck",
    );
    let mut sum_check_vp = VirtualPolynomial::new(num_var);
    for (g, eq) in merged_tilde_gs.iter().zip(tilde_eqs.into_iter()) {
        sum_check_vp.add_mle_list([g.clone(), eq], E::ScalarField::one())?;
    }
    let proof = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::prove(
        &sum_check_vp,
        transcript,
    ) {
        Ok(p) => p,
        Err(_) => return Err(PCSError::InvalidProver("Sumcheck failed".to_string())),
    };
    drop(_t_sc);

    let a2 = &proof.point[..num_var];

    let _t_g = profile::ScopedTimer::new(
        "Mulcs",
        num_var,
        n,
        "mulcs_multi_open_build_g_prime",
        1,
        "g'=sum",
    );
    let mut g_prime = Arc::new(DenseMultilinearExtension::zero());
    for (g, point) in merged_tilde_gs.iter().zip(deduped_points.iter()) {
        let eq = eq_eval(a2, point)?;
        *Arc::make_mut(&mut g_prime) += (eq, g.deref());
    }
    drop(_t_g);

    let mut open_t = IOPTranscript::new(b"mulcs-gprime-open");
    open_t.append_serializable_element(b"point_a2", &a2.to_vec())?;
    open_t.append_field_element(b"mu", &E::ScalarField::from(num_var as u64))?;
    let (g_prime_proof, _g_prime_eval) = open_with_transcript(pp, &g_prime, a2, &mut open_t)?;

    drop(_t_total);
    Ok(BatchProof {
        sum_check_proof: proof,
        f_i_eval_at_point_i: evals.to_vec(),
        g_prime_proof,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Sumcheck batch verify (profiled)
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn mulcs_sumcheck_batch_verify<E: Pairing>(
    vp: &MulcsVerifierParam<E>,
    f_i_commitments: &[Commitment<E>],
    points: &[Vec<E::ScalarField>],
    proof: &BatchProof<E, MulcsPCS<E>>,
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
    let n = checked_domain_size_from_mu(num_var, "batch_verify")?;
    let _t_total = profile::ScopedTimer::new(
        "Mulcs",
        num_var,
        n,
        "mulcs_batch_verify_total",
        k,
        "sumcheck-batch",
    );

    let _t_abs = profile::ScopedTimer::new(
        "Mulcs",
        num_var,
        n,
        "mulcs_batch_verify_transcript_absorb",
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
        "Mulcs",
        num_var,
        n,
        "mulcs_batch_verify_build_g_prime_commit",
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
        "Mulcs",
        num_var,
        n,
        "mulcs_batch_verify_sumcheck_verify",
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
        "Mulcs",
        num_var,
        n,
        "mulcs_batch_verify_final_open",
        1,
        "final-mulcs-open",
    );
    let mut verify_t = IOPTranscript::new(b"mulcs-gprime-open");
    verify_t.append_serializable_element(b"point_a2", &a2.to_vec())?;
    verify_t.append_field_element(b"mu", &E::ScalarField::from(num_var as u64))?;
    let res = verify_with_transcript(
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
// Polynomial utilities (unchanged)
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

fn check_claymore_identity<F: Field>(
    gamma: F,
    n: usize,
    z: F,
    y_f: F,
    y_hbar: F,
    y_hbar_prime: F,
    point: &[F],
    claimed_value: F,
) -> Result<bool, PCSError> {
    let z_inv = z
        .inverse()
        .ok_or_else(|| PCSError::InvalidParameters("z is zero".to_string()))?;
    let mut y_r = F::one();
    let mut z_pow = z_inv;
    for rk in point.iter() {
        y_r *= (F::one() - *rk) + *rk * z_pow;
        z_pow = z_pow.square();
    }
    let gamma_n1 = gamma.pow([(n - 1) as u64]);
    let z_n1 = z.pow([(n - 1) as u64]);
    Ok(gamma_n1 * y_hbar - y_hbar_prime == z_n1 * (y_f * y_r - claimed_value))
}

fn two_point_remainder<F: Field>(x0: F, y0: F, x1: F, y1: F) -> Result<[F; 2], PCSError> {
    let denom = x1 - x0;
    let inv = denom
        .inverse()
        .ok_or_else(|| PCSError::InvalidProof("duplicate KZG opening points".to_string()))?;
    let slope = (y1 - y0) * inv;
    let intercept = y0 - slope * x0;
    Ok([intercept, slope])
}

fn build_multi_point_polys<F: Field>(points: &[(F, F)]) -> (Vec<F>, Vec<F>) {
    let k = points.len();
    let mut zc = vec![F::ZERO; k + 1];
    zc[0] = F::one();
    for &(zi, _) in points {
        for d in (1..=k).rev() {
            zc[d] = zc[d - 1] - zi * zc[d];
        }
        zc[0] *= -zi;
    }
    let mut rc = vec![F::ZERO; k];
    for (i, &(zi, yi)) in points.iter().enumerate() {
        let mut num = vec![F::ZERO; k];
        num[0] = F::one();
        let mut denom = F::one();
        for (j, &(zj, _)) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            denom *= zi - zj;
            for d in (1..k).rev() {
                num[d] = num[d - 1] - zj * num[d];
            }
            num[0] *= -zj;
        }
        let scale = yi * denom.inverse().unwrap();
        for d in 0..k {
            rc[d] += num[d] * scale;
        }
    }
    (rc, zc)
}

fn poly_div<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    let db = b.len() - 1;
    let da = a.len() - 1;
    if da < db {
        return vec![];
    }
    let mut q = vec![F::ZERO; da - db + 1];
    let mut rem = a.to_vec();
    let inv = b[db].inverse().unwrap();
    for i in (db..=da).rev() {
        if rem[i].is_zero() {
            continue;
        }
        let c = rem[i] * inv;
        q[i - db] = c;
        for j in 0..=db {
            rem[i - db + j] -= c * b[j];
        }
    }
    q
}

fn poly_sub_div<F: Field>(f: &[F], r: &[F], z: &[F]) -> Vec<F> {
    let deg = f.len() - 1;
    let mut sub = f.to_vec();
    sub.resize(deg + 1, F::ZERO);
    for i in 0..r.len() {
        sub[i] -= r[i];
    }
    poly_div(&sub, z)
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
        MulcsPCS::<E>::trim(
            &MulcsPCS::<E>::gen_srs_for_testing(&mut rng, nv).unwrap(),
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
    fn test_mulcs_single_commit_open_verify() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [2usize, 4, 6] {
            let (ck, vk) = setup(nv);
            let poly = rpoly(nv, &mut rng);
            let point = rpt(nv, &mut rng);
            let com = MulcsPCS::<E>::commit(&ck, &poly)?;
            let (proof, value) = MulcsPCS::<E>::open(&ck, &poly, &point)?;
            assert!(MulcsPCS::<E>::verify(&vk, &com, &point, &value, &proof)?);
            let fake_val = Fr::rand(&mut rng);
            if fake_val != value {
                assert!(!MulcsPCS::<E>::verify(
                    &vk, &com, &point, &fake_val, &proof
                )?);
            }
        }
        Ok(())
    }

    #[test]
    fn test_mulcs_single_open_rejects_tampered_z() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com = MulcsPCS::<E>::commit(&ck, &poly)?;
        let (mut proof, value) = MulcsPCS::<E>::open(&ck, &poly, &point)?;
        proof.z += Fr::ONE;
        assert!(!MulcsPCS::<E>::verify(&vk, &com, &point, &value, &proof)?);
        Ok(())
    }

    #[test]
    fn test_mulcs_single_open_rejects_wrong_value_with_fs_z() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let (ck, vk) = setup(4);
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let com = MulcsPCS::<E>::commit(&ck, &poly)?;
        let (proof, _) = MulcsPCS::<E>::open(&ck, &poly, &point)?;
        assert!(!MulcsPCS::<E>::verify(
            &vk,
            &com,
            &point,
            &Fr::rand(&mut rng),
            &proof
        )?);
        Ok(())
    }

    #[test]
    fn test_mulcs_sumcheck_batch_verify() -> Result<(), PCSError> {
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
            .map(|p| MulcsPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let proof = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(MulcsPCS::<E>::batch_verify(
            &vk, &comms, &points, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_mulcs_sumcheck_batch_verify_k1() -> Result<(), PCSError> {
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
            .map(|p| MulcsPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let proof = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(MulcsPCS::<E>::batch_verify(
            &vk, &comms, &points, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_mulcs_sumcheck_batch_rejects_wrong_eval() -> Result<(), PCSError> {
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
            .map(|p| MulcsPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut we = evals.clone();
        we[0] += Fr::ONE;
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let p = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &we, &mut tp)?;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert_rejects(
            "we",
            MulcsPCS::<E>::batch_verify(&vk, &comms, &points, &p, &mut tv),
        );
        Ok(())
    }

    #[test]
    fn test_mulcs_sumcheck_batch_rejects_wrong_point() -> Result<(), PCSError> {
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
            .map(|p| MulcsPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let p = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        let mut wp = points.clone();
        wp[0] = rpt(4, &mut rng);
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert_rejects(
            "wp",
            MulcsPCS::<E>::batch_verify(&vk, &comms, &wp, &p, &mut tv),
        );
        Ok(())
    }

    #[test]
    fn test_mulcs_sumcheck_batch_rejects_wrong_commitment() -> Result<(), PCSError> {
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
            .map(|p| MulcsPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let p = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        let extra = MulcsPCS::<E>::commit(&ck, &rpoly(4, &mut rng))?;
        let mut wc = comms.clone();
        wc[0] = extra;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(!MulcsPCS::<E>::batch_verify(
            &vk, &wc, &points, &p, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_mulcs_sumcheck_batch_rejects_malformed_lengths() -> Result<(), PCSError> {
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
            .map(|p| MulcsPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let mut proof = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        let r = MulcsPCS::<E>::batch_verify(&vk, &comms[..2], &points, &proof, &mut tv);
        assert!(r.is_err() || !r.unwrap());
        proof.f_i_eval_at_point_i.pop();
        let mut tv2 = IOPTranscript::new(b"test");
        tv2.append_field_element(b"init", &Fr::ZERO)?;
        let r2 = MulcsPCS::<E>::batch_verify(&vk, &comms, &points, &proof, &mut tv2);
        assert!(r2.is_err() || !r2.unwrap());
        Ok(())
    }

    #[test]
    fn test_mulcs_multi_open_rejects_empty_input() {
        let (ck, _) = setup(4);
        let mut tp = IOPTranscript::new(b"test");
        assert!(MulcsPCS::<E>::multi_open(
            &ck,
            &[] as &[Arc<_>],
            &[] as &[Vec<_>],
            &[] as &[Fr],
            &mut tp
        )
        .is_err());
    }

    #[test]
    fn test_mulcs_multi_open_rejects_mismatched_lengths() {
        let (ck, _) = setup(4);
        let mut rng = test_rng();
        let poly = rpoly(4, &mut rng);
        let point = rpt(4, &mut rng);
        let mut tp = IOPTranscript::new(b"test");
        assert!(MulcsPCS::<E>::multi_open(
            &ck,
            &[poly.clone()],
            &[point.clone(), point.clone()],
            &[Fr::one()],
            &mut tp
        )
        .is_err());
        let mut tp = IOPTranscript::new(b"test");
        assert!(MulcsPCS::<E>::multi_open(
            &ck,
            &[poly.clone()],
            &[point.clone()],
            &[Fr::one(), Fr::one()],
            &mut tp
        )
        .is_err());
    }

    #[test]
    fn test_mulcs_multi_open_rejects_inconsistent_num_vars() {
        let (ck, _) = setup(4);
        let mut rng = test_rng();
        let point = rpt(4, &mut rng);
        let mut tp = IOPTranscript::new(b"test");
        assert!(MulcsPCS::<E>::multi_open(
            &ck,
            &[rpoly(4, &mut rng), rpoly(3, &mut rng)],
            &[point.clone(), point.clone()],
            &[Fr::one(), Fr::one()],
            &mut tp
        )
        .is_err());
    }

    #[test]
    fn test_mulcs_multi_open_rejects_wrong_point_len() {
        let (ck, _) = setup(4);
        let mut rng = test_rng();
        let mut tp = IOPTranscript::new(b"test");
        assert!(MulcsPCS::<E>::multi_open(
            &ck,
            &[rpoly(4, &mut rng)],
            &[rpt(3, &mut rng)],
            &[Fr::one()],
            &mut tp
        )
        .is_err());
    }

    // ── Huge mu / num_var do not panic ──

    #[test]
    fn test_mulcs_verify_rejects_huge_mu_without_panic() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 2;
        let (ck, vk) = setup(nv);
        let poly = rpoly(nv, &mut rng);
        let point = rpt(nv, &mut rng);
        let com = MulcsPCS::<E>::commit(&ck, &poly)?;
        let (mut proof, value) = MulcsPCS::<E>::open(&ck, &poly, &point)?;
        proof.mu = usize::BITS as usize;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            MulcsPCS::<E>::verify(&vk, &com, &point, &value, &proof)
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

    #[test]
    fn test_mulcs_verify_rejects_wrong_point_len() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let poly = rpoly(nv, &mut rng);
        let point = rpt(nv, &mut rng);
        let com = MulcsPCS::<E>::commit(&ck, &poly)?;
        let (proof, value) = MulcsPCS::<E>::open(&ck, &poly, &point)?;
        let short_pt = rpt(2, &mut rng);
        let r = MulcsPCS::<E>::verify(&vk, &com, &short_pt, &value, &proof);
        assert!(r.is_err(), "short point should return Error, not panic");
        let long_pt = rpt(8, &mut rng);
        let r2 = MulcsPCS::<E>::verify(&vk, &com, &long_pt, &value, &proof);
        assert!(r2.is_err(), "long point should return Error, not panic");
        Ok(())
    }

    #[test]
    fn test_mulcs_batch_verify_rejects_huge_num_var_without_panic() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 2;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..1).map(|_| rpoly(nv, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| rpt(nv, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| MulcsPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let mut proof = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;

        proof.sum_check_proof.point = vec![Fr::zero(); usize::BITS as usize];
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            MulcsPCS::<E>::batch_verify(&vk, &comms, &points, &proof, &mut tv)
        }));
        match r {
            Ok(verdict) => assert!(
                verdict.is_err() || !verdict.unwrap(),
                "huge num_var ({}) should fail without panic",
                proof.sum_check_proof.point.len()
            ),
            Err(_) => panic!("caught panic on huge num_var — should not panic"),
        }
        Ok(())
    }

    fn assert_rejects(_: &str, r: Result<bool, PCSError>) {
        match r {
            Ok(true) => panic!("expected reject"),
            Ok(false) => {},
            Err(_) => {},
        }
    }
}
