//! Mulcs PCS implementation — Claymore identity-based multilinear PCS
//! using univariate KZG as the black-box commitment scheme.
//!
//! Batch opening: Mulcs-specific sumcheck batching that reduces multiple
//! openings to a single opening of an aggregated polynomial g', then
//! opens g' at the sumcheck point via transcript-aware Mulcs single open.

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
    pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, AffineRepr, CurveGroup,
};
use ark_ff::Field;
use ark_poly::MultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::Borrow, end_timer, format, log2, marker::PhantomData, rand::Rng, start_timer,
    string::ToString, sync::Arc, vec, vec::Vec, One, Zero,
};
use std::{collections::BTreeMap, iter, ops::Deref};
use transcript::IOPTranscript;

use self::util::UnivarPoly;

mod profile;
pub(crate) mod srs;
mod util;

use srs::{MulcsProverParam, MulcsUniversalParams, MulcsVerifierParam};

/// Mulcs Polynomial Commitment Scheme on multilinear polynomials.
pub struct MulcsPCS<E: Pairing> {
    #[doc(hidden)]
    phantom: PhantomData<E>,
}

/// Single-opening proof. Contains the Claymore identity components.
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

    fn gen_srs_for_testing<R: Rng>(
        rng: &mut R,
        supported_size: usize,
    ) -> Result<Self::SRS, PCSError> {
        MulcsUniversalParams::<E>::gen_srs_for_testing(rng, supported_size)
    }

    fn trim(
        srs: impl Borrow<Self::SRS>,
        _supported_degree: Option<usize>,
        supported_num_vars: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        let supported_num_vars = match supported_num_vars {
            Some(p) => p,
            None => {
                return Err(PCSError::InvalidParameters(
                    "mulcs should receive a num_var param".to_string(),
                ))
            },
        };
        let max_degree = 2 * (1 << supported_num_vars);
        srs.borrow().trim(max_degree)
    }

    fn commit(
        prover_param: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let pp = prover_param.borrow();
        let nv = poly.num_vars;
        let n = 1 << nv;
        let commit_timer = start_timer!(|| "mulcs commit");
        if pp.max_degree < n - 1 {
            return Err(PCSError::InvalidParameters(format!(
                "poly degree {} exceeds SRS max {}",
                n - 1,
                pp.max_degree
            )));
        }
        let _t_eval = profile::ScopedTimer::new(nv, n, "commit_to_evals", n, "to_evaluations");
        let scalars = poly.to_evaluations();
        drop(_t_eval);
        let _t_msm = profile::ScopedTimer::new(nv, n, "commit_msm", scalars.len(), "KZG-MSM");
        let cm = pp.commit(&scalars);
        drop(_t_msm);
        end_timer!(commit_timer);
        Ok(Commitment(cm))
    }

    /// Standalone open (no transcript). Wraps a local transcript for FS
    /// security.
    fn open(
        prover_param: impl Borrow<Self::ProverParam>,
        polynomial: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let mut t = IOPTranscript::new(b"mulcs-open");
        t.append_field_element(b"mu", &E::ScalarField::from(polynomial.num_vars as u64))?;
        open_with_transcript(prover_param.borrow(), polynomial, point, &mut t)
    }

    fn multi_open(
        prover_param: impl Borrow<Self::ProverParam>,
        polynomials: &[Self::Polynomial],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Self::BatchProof, PCSError> {
        mulcs_sumcheck_multi_open(
            prover_param.borrow(),
            polynomials,
            points,
            evals,
            transcript,
        )
    }

    /// Standalone verify (no transcript). Wraps a local transcript for FS
    /// security.
    fn verify(
        verifier_param: &Self::VerifierParam,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        let mut t = IOPTranscript::new(b"mulcs-open");
        t.append_field_element(b"mu", &E::ScalarField::from(proof.mu as u64))?;
        verify_with_transcript(verifier_param, commitment, point, value, proof, &mut t)
    }

    fn batch_verify(
        verifier_param: &Self::VerifierParam,
        commitments: &[Self::Commitment],
        points: &[Self::Point],
        batch_proof: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        mulcs_sumcheck_batch_verify(verifier_param, commitments, points, batch_proof, transcript)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Transcript-aware Mulcs single opening
// ═══════════════════════════════════════════════════════════════════

/// Transcript-aware Mulcs opening.
/// Transcript must already contain the opening point and claimed value.
/// This function appends cm_hbar, then derives z and delta from transcript.
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

    let coeffs = polynomial.to_evaluations();
    let f_v = UnivarPoly::new(coeffs.clone());
    let y = polynomial
        .evaluate(point)
        .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;

    let h = UnivarPoly::compute_h(&coeffs, mu, point, y);

    // Derive delta from transcript (Fiat-Shamir)
    let delta_buf = transcript.get_and_append_challenge_vectors(b"mulcs_delta", 1)?;
    let delta = delta_buf[0];

    let h_bar = UnivarPoly::compute_h_bar(&h, gamma, n, delta);
    let cm_hbar = pp.commit(&h_bar.coeffs);
    transcript.append_serializable_element(b"cm_hbar", &cm_hbar)?;

    // Derive z from transcript (Fiat-Shamir)
    let z_buf = transcript.get_and_append_challenge_vectors(b"mulcs_z", 1)?;
    let z = z_buf[0];

    let gz = gamma * z;
    let y_f = f_v.evaluate(z);
    let y_f_prime = f_v.evaluate(gz);
    let y_hbar = h_bar.evaluate(z);
    let y_hbar_prime = h_bar.evaluate(gz);

    let (pi, rf, rh) = mulcs_batch_kzg_open(
        pp,
        &f_v,
        &h_bar,
        z,
        gamma,
        y_f,
        y_f_prime,
        y_hbar,
        y_hbar_prime,
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
    Ok((proof, y))
}

/// Transcript-aware Mulcs verification.
/// Transcript must already contain the opening point and claimed value.
/// Replays transcript to derive z and delta, then verifies KZG + Claymore.
pub(crate) fn verify_with_transcript<E: Pairing>(
    vp: &MulcsVerifierParam<E>,
    commitment: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &MulcsProof<E>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError> {
    let mu = proof.mu;
    let n = 1 << mu;
    let gamma = vp.gamma;

    // Replay transcript to get delta
    let delta_buf = transcript.get_and_append_challenge_vectors(b"mulcs_delta", 1)?;
    let _delta = delta_buf[0]; // verifier doesn't need delta value

    transcript.append_serializable_element(b"cm_hbar", &proof.cm_hbar)?;

    // Replay transcript to get z
    let z_buf = transcript.get_and_append_challenge_vectors(b"mulcs_z", 1)?;
    let z = z_buf[0];
    if z != proof.z {
        return Ok(false);
    }

    // KZG pairing check
    let _t_pair = profile::ScopedTimer::new(mu, n, "verify_pairing", 1, "single-pairing");
    let gz = gamma * z;
    let f_pts = [(z, proof.y_f), (gz, proof.y_f_prime)];
    let h_pts = [(z, proof.y_hbar), (gz, proof.y_hbar_prime)];
    let (rf, _) = build_multi_point_polys(&f_pts);
    let (rh, _) = build_multi_point_polys(&h_pts);
    let mut r_comb = vec![E::ScalarField::zero(); 2];
    for j in 0..2 {
        r_comb[j] = rf[j] + rh[j];
    }
    let cm_r = vp.g1_one.into_group() * r_comb[0] + vp.g1_x.into_group() * r_comb[1];
    let cm_comb = commitment.0.into_group() + proof.cm_hbar.into_group() - cm_r;
    let s = z + gz;
    let p = z * gz;
    let zx_g2 = vp.g2_x2.into_group() - vp.g2_x.into_group() * s + vp.g2_one.into_group() * p;
    let pair_ok =
        E::pairing(cm_comb.into_affine(), vp.g2_one) == E::pairing(proof.pi, zx_g2.into_affine());
    drop(_t_pair);
    if !pair_ok {
        return Ok(false);
    }

    // Claymore identity
    let _t_clay = profile::ScopedTimer::new(mu, n, "verify_claymore", 1, "claymore-identity");
    let result = check_claymore_identity(
        gamma,
        mu,
        z,
        proof.y_f,
        proof.y_hbar,
        proof.y_hbar_prime,
        point,
        *value,
    );
    drop(_t_clay);
    result
}

// ═══════════════════════════════════════════════════════════════════
// Mulcs-specific sumcheck batching
// ═══════════════════════════════════════════════════════════════════

/// Mulcs-specific sumcheck batch open. Same logic as generic sumcheck batcher
/// but uses `open_with_transcript` for the final g' opening.
pub(crate) fn mulcs_sumcheck_multi_open<E: Pairing>(
    pp: &MulcsProverParam<E>,
    polynomials: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    points: &[Vec<E::ScalarField>],
    evals: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<BatchProof<E, MulcsPCS<E>>, PCSError> {
    let open_timer = start_timer!(|| format!("mulcs sumcheck multi open {} points", points.len()));
    // Sanity checks
    if polynomials.is_empty() {
        return Err(PCSError::InvalidParameters(
            "empty polynomial list".to_string(),
        ));
    }
    if polynomials.len() != points.len() || polynomials.len() != evals.len() {
        return Err(PCSError::InvalidParameters(format!(
            "batch opening length mismatch: polynomials={}, points={}, evals={}",
            polynomials.len(),
            points.len(),
            evals.len()
        )));
    }
    for eval_point in points.iter() {
        transcript.append_serializable_element(b"eval_point", eval_point)?;
    }
    for eval in evals.iter() {
        transcript.append_field_element(b"eval", eval)?;
    }
    let num_var = polynomials[0].num_vars;
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
                "point length {} != num_vars {}",
                point.len(),
                num_var
            )));
        }
    }

    let k = polynomials.len();
    let ell = log2(k) as usize;

    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;
    let eq_t_i_list = if ell == 0 {
        vec![E::ScalarField::one()]
    } else {
        build_eq_x_r_vec(t.as_ref())?
    };

    let point_indices = points
        .iter()
        .fold(BTreeMap::<_, _>::new(), |mut indices, point| {
            let idx = indices.len();
            indices.entry(point).or_insert(idx);
            indices
        });
    let deduped_points =
        BTreeMap::from_iter(point_indices.iter().map(|(point, idx)| (*idx, *point)))
            .into_values()
            .collect::<Vec<_>>();
    let merged_tilde_gs = polynomials
        .iter()
        .zip(points.iter())
        .zip(eq_t_i_list.iter())
        .fold(
            iter::repeat_with(DenseMultilinearExtension::zero)
                .map(Arc::new)
                .take(point_indices.len())
                .collect::<Vec<_>>(),
            |mut merged_tilde_gs, ((poly, point), coeff)| {
                *Arc::make_mut(&mut merged_tilde_gs[point_indices[point]]) +=
                    (*coeff, poly.deref());
                merged_tilde_gs
            },
        );

    let tilde_eqs: Vec<_> = deduped_points
        .iter()
        .map(|point| {
            let eq_b_zi = build_eq_x_r_vec(point).unwrap();
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_var, eq_b_zi,
            ))
        })
        .collect();

    let mut sum_check_vp = VirtualPolynomial::new(num_var);
    for (merged_tilde_g, tilde_eq) in merged_tilde_gs.iter().zip(tilde_eqs.into_iter()) {
        sum_check_vp.add_mle_list([merged_tilde_g.clone(), tilde_eq], E::ScalarField::one())?;
    }

    let proof = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::prove(
        &sum_check_vp,
        transcript,
    ) {
        Ok(p) => p,
        Err(_) => {
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch proving Failed".to_string(),
            ));
        },
    };

    let a2 = &proof.point[..num_var];

    let mut g_prime = Arc::new(DenseMultilinearExtension::zero());
    for (merged_tilde_g, point) in merged_tilde_gs.iter().zip(deduped_points.iter()) {
        let eq_i_a2 = eq_eval(a2, point)?;
        *Arc::make_mut(&mut g_prime) += (eq_i_a2, merged_tilde_g.deref());
    }

    // Use transcript-aware Mulcs open
    let mut open_t = IOPTranscript::new(b"mulcs-gprime-open");
    open_t.append_serializable_element(b"point_a2", &a2.to_vec())?;
    open_t.append_field_element(b"mu", &E::ScalarField::from(num_var as u64))?;
    let (g_prime_proof, _g_prime_eval) = open_with_transcript(pp, &g_prime, a2, &mut open_t)?;

    end_timer!(open_timer);
    Ok(BatchProof {
        sum_check_proof: proof,
        f_i_eval_at_point_i: evals.to_vec(),
        g_prime_proof,
    })
}

/// Mulcs-specific sumcheck batch verify. Same logic as generic sumcheck batcher
/// but uses `verify_with_transcript` for the final g' opening.
pub(crate) fn mulcs_sumcheck_batch_verify<E: Pairing>(
    vp: &MulcsVerifierParam<E>,
    f_i_commitments: &[Commitment<E>],
    points: &[Vec<E::ScalarField>],
    proof: &BatchProof<E, MulcsPCS<E>>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError> {
    let _t_total = profile::ScopedTimer::new(0, 0, "batch_verify_total", 0, "sumcheck");
    let open_timer = start_timer!(|| "mulcs sumcheck batch verify");

    if f_i_commitments.is_empty() {
        return Err(PCSError::InvalidProof("empty commitments".to_string()));
    }
    if f_i_commitments.len() != points.len()
        || f_i_commitments.len() != proof.f_i_eval_at_point_i.len()
    {
        return Err(PCSError::InvalidProof(
            "batch verify length mismatch".to_string(),
        ));
    }

    for eval_point in points.iter() {
        transcript.append_serializable_element(b"eval_point", eval_point)?;
    }
    for eval in proof.f_i_eval_at_point_i.iter() {
        transcript.append_field_element(b"eval", eval)?;
    }

    let k = f_i_commitments.len();
    let ell = log2(k) as usize;
    let num_var = proof.sum_check_proof.point.len();

    for point in points {
        if point.len() != num_var {
            return Err(PCSError::InvalidProof(format!(
                "point length {} != num_var {}",
                point.len(),
                num_var
            )));
        }
    }

    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;
    let a2 = &proof.sum_check_proof.point[..num_var];

    let eq_t_list = if ell == 0 {
        vec![E::ScalarField::one()]
    } else {
        build_eq_x_r_vec(t.as_ref())?
    };

    let mut scalars = vec![];
    let mut bases = vec![];
    for (i, point) in points.iter().enumerate() {
        let eq_i_a2 = eq_eval(a2, point)?;
        scalars.push(eq_i_a2 * eq_t_list[i]);
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
        Err(_) => {
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch verification failed".to_string(),
            ));
        },
    };
    let tilde_g_eval = subclaim.expected_evaluation;

    // Use transcript-aware Mulcs verify
    let mut verify_t = IOPTranscript::new(b"mulcs-gprime-open");
    verify_t.append_serializable_element(b"point_a2", &a2.to_vec())?;
    verify_t.append_field_element(b"mu", &E::ScalarField::from(num_var as u64))?;
    let res = verify_with_transcript(
        vp,
        &Commitment(g_prime_commit.into_affine()),
        a2,
        &tilde_g_eval,
        &proof.g_prime_proof,
        &mut verify_t,
    )?;

    end_timer!(open_timer);
    Ok(res)
}

// ═══════════════════════════════════════════════════════════════════
// KZG multi-point opening / Claymore identity / polynomial utils
// ═══════════════════════════════════════════════════════════════════

fn mulcs_batch_kzg_open<E: Pairing>(
    pp: &MulcsProverParam<E>,
    f_v: &UnivarPoly<E::ScalarField>,
    h_bar: &UnivarPoly<E::ScalarField>,
    z: E::ScalarField,
    gamma: E::ScalarField,
    y_f: E::ScalarField,
    y_f_prime: E::ScalarField,
    y_h: E::ScalarField,
    y_h_prime: E::ScalarField,
) -> (E::G1Affine, Vec<E::ScalarField>, Vec<E::ScalarField>) {
    let gz = gamma * z;
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
    let pi = pp.commit(&q_comb);
    (pi, rf, rh)
}

/// Evaluate eq polynomial.
fn eq_eval<F: Field>(x: &[F], y: &[F]) -> Result<F, PCSError> {
    if x.len() != y.len() {
        return Err(PCSError::InvalidParameters(
            "x and y have different length".to_string(),
        ));
    }
    let mut res = F::one();
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        res *= xi * yi + (F::one() - xi) * (F::one() - yi);
    }
    Ok(res)
}

fn check_claymore_identity<F: Field>(
    gamma: F,
    mu: usize,
    z: F,
    y_f: F,
    y_hbar: F,
    y_hbar_prime: F,
    point: &[F],
    claimed_value: F,
) -> Result<bool, PCSError> {
    let n = 1 << mu;
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

fn build_multi_point_polys<F: Field>(points: &[(F, F)]) -> (Vec<F>, Vec<F>) {
    let k = points.len();
    let mut z_coeffs = vec![F::ZERO; k + 1];
    z_coeffs[0] = F::one();
    for &(zi, _) in points {
        for d in (1..=k).rev() {
            z_coeffs[d] = z_coeffs[d - 1] - zi * z_coeffs[d];
        }
        z_coeffs[0] *= -zi;
    }
    let mut r_coeffs = vec![F::ZERO; k];
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
            r_coeffs[d] += num[d] * scale;
        }
    }
    (r_coeffs, z_coeffs)
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
        let srs = MulcsPCS::<E>::gen_srs_for_testing(&mut rng, nv).unwrap();
        MulcsPCS::<E>::trim(&srs, None, Some(nv)).unwrap()
    }

    fn random_point(nv: usize, rng: &mut impl ark_std::rand::Rng) -> Vec<Fr> {
        (0..nv).map(|_| Fr::rand(rng)).collect()
    }

    fn random_poly(
        nv: usize,
        rng: &mut impl ark_std::rand::Rng,
    ) -> Arc<DenseMultilinearExtension<Fr>> {
        Arc::new(DenseMultilinearExtension::rand(nv, rng))
    }

    #[test]
    fn test_mulcs_single_commit_open_verify() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [2usize, 4, 6] {
            let (ck, vk) = setup(nv);
            let poly = random_poly(nv, &mut rng);
            let point = random_point(nv, &mut rng);
            let com = MulcsPCS::<E>::commit(&ck, &poly)?;
            let (proof, value) = MulcsPCS::<E>::open(&ck, &poly, &point)?;
            assert!(
                MulcsPCS::<E>::verify(&vk, &com, &point, &value, &proof)?,
                "verify nv={nv}"
            );
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
        let nv = 4;
        let (ck, vk) = setup(nv);
        let poly = random_poly(nv, &mut rng);
        let point = random_point(nv, &mut rng);
        let com = MulcsPCS::<E>::commit(&ck, &poly)?;
        let (mut proof, value) = MulcsPCS::<E>::open(&ck, &poly, &point)?;
        proof.z += Fr::ONE;
        assert!(!MulcsPCS::<E>::verify(&vk, &com, &point, &value, &proof)?);
        Ok(())
    }

    #[test]
    fn test_mulcs_single_open_rejects_wrong_value_with_fs_z() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let poly = random_poly(nv, &mut rng);
        let point = random_point(nv, &mut rng);
        let com = MulcsPCS::<E>::commit(&ck, &poly)?;
        let (proof, _value) = MulcsPCS::<E>::open(&ck, &poly, &point)?;
        let fake_val = Fr::rand(&mut rng);
        assert!(!MulcsPCS::<E>::verify(
            &vk, &com, &point, &fake_val, &proof
        )?);
        Ok(())
    }

    #[test]
    fn test_mulcs_sumcheck_batch_verify() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..3).map(|_| random_poly(nv, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| random_point(nv, &mut rng)).collect();
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
        let nv = 4;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..1).map(|_| random_poly(nv, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| random_point(nv, &mut rng)).collect();
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
        let nv = 4;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..3).map(|_| random_poly(nv, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| random_point(nv, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| MulcsPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut wrong_evals = evals.clone();
        wrong_evals[0] += Fr::ONE;
        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let proof = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &wrong_evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert_rejects(
            "wrong_eval_sumcheck",
            MulcsPCS::<E>::batch_verify(&vk, &comms, &points, &proof, &mut tv),
        );
        Ok(())
    }

    #[test]
    fn test_mulcs_sumcheck_batch_rejects_wrong_point() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..3).map(|_| random_poly(nv, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| random_point(nv, &mut rng)).collect();
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
        let mut wrong_points = points.clone();
        wrong_points[0] = random_point(nv, &mut rng);
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert_rejects(
            "wrong_point_sumcheck",
            MulcsPCS::<E>::batch_verify(&vk, &comms, &wrong_points, &proof, &mut tv),
        );
        Ok(())
    }

    #[test]
    fn test_mulcs_sumcheck_batch_rejects_wrong_commitment() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..3).map(|_| random_poly(nv, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| random_point(nv, &mut rng)).collect();
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
        let extra = MulcsPCS::<E>::commit(&ck, &random_poly(nv, &mut rng))?;
        let mut wrong_comms = comms.clone();
        wrong_comms[0] = extra;
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(!MulcsPCS::<E>::batch_verify(
            &vk,
            &wrong_comms,
            &points,
            &proof,
            &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_mulcs_sumcheck_batch_rejects_malformed_lengths() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let polys: Vec<_> = (0..3).map(|_| random_poly(nv, &mut rng)).collect();
        let points: Vec<_> = polys.iter().map(|_| random_point(nv, &mut rng)).collect();
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
        let short = &comms[..2];
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        let r = MulcsPCS::<E>::batch_verify(&vk, short, &points, &proof, &mut tv);
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
        let (ck, _vk) = setup(4);
        let empty: &[Arc<DenseMultilinearExtension<Fr>>] = &[];
        let points: &[Vec<Fr>] = &[];
        let evals: &[Fr] = &[];
        let mut tp = IOPTranscript::new(b"test");
        let r = MulcsPCS::<E>::multi_open(&ck, empty, points, evals, &mut tp);
        assert!(r.is_err());
    }

    #[test]
    fn test_mulcs_multi_open_rejects_mismatched_lengths() {
        let (ck, _vk) = setup(4);
        let mut rng = test_rng();
        let poly = random_poly(4, &mut rng);
        let point = random_point(4, &mut rng);
        let mut tp = IOPTranscript::new(b"test");
        let r = MulcsPCS::<E>::multi_open(
            &ck,
            &[poly.clone()],
            &[point.clone(), point.clone()],
            &[Fr::one()],
            &mut tp,
        );
        assert!(r.is_err());
        let mut tp = IOPTranscript::new(b"test");
        let r = MulcsPCS::<E>::multi_open(
            &ck,
            &[poly.clone()],
            &[point.clone()],
            &[Fr::one(), Fr::one()],
            &mut tp,
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_mulcs_multi_open_rejects_inconsistent_num_vars() {
        let (ck, _vk) = setup(4);
        let mut rng = test_rng();
        let poly4 = random_poly(4, &mut rng);
        let poly3 = random_poly(3, &mut rng);
        let point = random_point(4, &mut rng);
        let mut tp = IOPTranscript::new(b"test");
        let r = MulcsPCS::<E>::multi_open(
            &ck,
            &[poly4, poly3],
            &[point.clone(), point.clone()],
            &[Fr::one(), Fr::one()],
            &mut tp,
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_mulcs_multi_open_rejects_wrong_point_len() {
        let (ck, _vk) = setup(4);
        let mut rng = test_rng();
        let poly = random_poly(4, &mut rng);
        let short_point = random_point(3, &mut rng);
        let mut tp = IOPTranscript::new(b"test");
        let r = MulcsPCS::<E>::multi_open(&ck, &[poly], &[short_point], &[Fr::one()], &mut tp);
        assert!(r.is_err());
    }

    fn assert_rejects(_backend: &str, result: Result<bool, PCSError>) {
        match result {
            Ok(true) => panic!("expected reject but got true"),
            Ok(false) => {},
            Err(_e) => {},
        }
    }
}
