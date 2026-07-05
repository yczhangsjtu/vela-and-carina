//! Mulcs PCS implementation — Claymore identity-based multilinear PCS
//! using univariate KZG as the black-box commitment scheme.

use crate::pcs::{
    prelude::{Commitment, PCSError},
    HasEvals, PolynomialCommitmentScheme, StructuredReferenceString,
};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::Field;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::Borrow, end_timer, format, marker::PhantomData, rand::Rng, start_timer,
    string::ToString, sync::Arc, vec, vec::Vec, One, Zero,
};
use srs::{MulcsProverParam, MulcsUniversalParams, MulcsVerifierParam};
use transcript::IOPTranscript;

use self::util::UnivarPoly;

mod profile;
pub(crate) mod srs;
mod util;

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

/// Batch-opening proof for multiple polynomials at possibly different points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MulcsBatchProof<E: Pairing> {
    /// Evaluations at each (poly, point) pair, in insertion order.
    pub f_i_eval_at_point_i: Vec<E::ScalarField>,
    /// Commitments to h̄ polynomials.
    pub cm_hbars: Vec<E::G1Affine>,
    /// Mulcs-specific evaluations: (y_f, y_f', y_h, y_h') per poly.
    pub mulcs_evals: Vec<(
        E::ScalarField,
        E::ScalarField,
        E::ScalarField,
        E::ScalarField,
    )>,
    /// Challenge point z (derived from Fiat-Shamir).
    pub z: E::ScalarField,
    /// Aggregated KZG batch proof.
    pub pi: E::G1Affine,
    /// log2(size) of each polynomial.
    pub mu: usize,
    /// Number of (poly, point) pairs.
    pub num_polys: usize,
}

impl<E: Pairing> HasEvals<E::ScalarField> for MulcsBatchProof<E> {
    fn evals(&self) -> &[E::ScalarField] {
        &self.f_i_eval_at_point_i
    }
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
    type BatchProof = MulcsBatchProof<E>;

    /// Build SRS for testing.
    ///
    /// `supported_size` is the number of variables for multilinear.
    /// Internal max_degree = 2 * 2^num_vars.
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
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

    /// Commit to a multilinear polynomial. Converts evaluation vector to
    /// univariate coefficient form and performs one KZG MSM over 2^NV G1.
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

    /// Open a polynomial at a single multilinear point.
    fn open(
        prover_param: impl Borrow<Self::ProverParam>,
        polynomial: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let pp = prover_param.borrow();
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
        let delta = E::ScalarField::one(); // research: verifier does not need to reproduce delta
        let h_bar = UnivarPoly::compute_h_bar(&h, gamma, n, delta);
        let cm_hbar = pp.commit(&h_bar.coeffs);

        let z = E::ScalarField::from(2u64); // research: standalone open without transcript
        let gz = gamma * z;

        let y_f = f_v.evaluate(z);
        let y_f_prime = f_v.evaluate(gz);
        let y_hbar = h_bar.evaluate(z);
        let y_hbar_prime = h_bar.evaluate(gz);

        // KZG batch multi-point: f_v and h_bar at (z, γz)
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

    fn multi_open(
        prover_param: impl Borrow<Self::ProverParam>,
        polynomials: &[Self::Polynomial],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<MulcsBatchProof<E>, PCSError> {
        multi_open_internal(
            prover_param.borrow(),
            polynomials,
            points,
            evals,
            transcript,
        )
    }

    fn verify(
        verifier_param: &Self::VerifierParam,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        verify_internal(verifier_param, commitment, point, value, proof)
    }

    fn batch_verify(
        verifier_param: &Self::VerifierParam,
        commitments: &[Self::Commitment],
        points: &[Self::Point],
        batch_proof: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        batch_verify_internal(verifier_param, commitments, points, batch_proof, transcript)
    }
}

// ─── KZG multi-point opening (f_v, h̄ at z, γz) ────────────────────

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

    // Random batch: q_comb = q_f + q_h
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

pub(crate) fn multi_open_internal<E: Pairing>(
    pp: &MulcsProverParam<E>,
    polynomials: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    points: &[Vec<E::ScalarField>],
    evals: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<MulcsBatchProof<E>, PCSError> {
    let nv = polynomials[0].num_vars;
    let n = 1 << nv;
    let mu = nv;
    let gamma = pp.gamma;
    let num_polys = polynomials.len();
    let _t_total = profile::ScopedTimer::new(nv, n, "multi_open_total", num_polys, "total");
    let open_timer = start_timer!(|| format!("mulcs multi open {} points", points.len()));

    let _t_app = profile::ScopedTimer::new(
        nv,
        n,
        "multi_open_append_pts_evals",
        num_polys,
        "transcript",
    );
    for eval_point in points.iter() {
        transcript.append_serializable_element(b"eval_point", eval_point)?;
    }
    for eval in evals.iter() {
        transcript.append_field_element(b"eval", eval)?;
    }
    drop(_t_app);

    let mut h_bars = Vec::with_capacity(num_polys);
    let mut cm_hbars = Vec::with_capacity(num_polys);
    let mut f_vs = Vec::with_capacity(num_polys);

    let _t_perpoly =
        profile::ScopedTimer::new(nv, n, "multi_open_per_poly", num_polys, "h-hbar-commit");
    for i in 0..num_polys {
        let coeffs = polynomials[i].to_evaluations();
        let f_v = UnivarPoly::new(coeffs.clone());
        let h = UnivarPoly::compute_h(&coeffs, mu, &points[i], evals[i]);
        let delta = E::ScalarField::one();
        let h_bar = UnivarPoly::compute_h_bar(&h, gamma, n, delta);
        let cm_hbar = pp.commit(&h_bar.coeffs);
        transcript.append_serializable_element(b"h", &cm_hbar)?;
        f_vs.push(f_v);
        h_bars.push(h_bar);
        cm_hbars.push(cm_hbar);
    }
    drop(_t_perpoly);

    let _t_fs = profile::ScopedTimer::new(nv, n, "multi_open_fs_z", 1, "transcript-challenge");
    let z_buf = transcript.get_and_append_challenge_vectors(b"z", 1)?;
    let z = z_buf[0];
    drop(_t_fs);
    let gz = gamma * z;

    let _t_evals =
        profile::ScopedTimer::new(nv, n, "multi_open_eval_zgz", num_polys, "eval-f-hbar");
    let mut evals_out = Vec::with_capacity(num_polys);
    for i in 0..num_polys {
        let y_f = f_vs[i].evaluate(z);
        let y_f_prime = f_vs[i].evaluate(gz);
        let y_h = h_bars[i].evaluate(z);
        let y_h_prime = h_bars[i].evaluate(gz);
        evals_out.push((y_f, y_f_prime, y_h, y_h_prime));
        transcript.append_field_element(b"y_f", &y_f)?;
        transcript.append_field_element(b"y_f'", &y_f_prime)?;
        transcript.append_field_element(b"y_h", &y_h)?;
        transcript.append_field_element(b"y_h'", &y_h_prime)?;
    }
    drop(_t_evals);

    let _t_fs2 = profile::ScopedTimer::new(
        nv,
        n,
        "multi_open_fs_inner_outer",
        1,
        "transcript-challenge",
    );
    let inner_r_buf = transcript.get_and_append_challenge_vectors(b"inner", 1)?;
    let inner_r = inner_r_buf[0];
    let outer_r_buf = transcript.get_and_append_challenge_vectors(b"outer", 1)?;
    let outer_r = outer_r_buf[0];
    drop(_t_fs2);

    let _t_quot = profile::ScopedTimer::new(
        nv,
        n,
        "multi_open_quotient_construction",
        num_polys,
        "poly-div",
    );
    let dummy_pts = [(z, E::ScalarField::zero()), (gz, E::ScalarField::zero())];
    let (_, z_coeffs) = build_multi_point_polys(&dummy_pts);
    let z_deg = z_coeffs.len().saturating_sub(1);

    let max_q_deg = (0..num_polys)
        .map(|i| {
            let fq_deg = f_vs[i].coeffs.len().saturating_sub(1).saturating_sub(z_deg);
            let hq_deg = h_bars[i]
                .coeffs
                .len()
                .saturating_sub(1)
                .saturating_sub(z_deg);
            fq_deg.max(hq_deg)
        })
        .max()
        .unwrap_or(0);

    let mut q_combined = vec![E::ScalarField::zero(); max_q_deg + 1];
    let mut outer_r_pow = E::ScalarField::one();

    for i in 0..num_polys {
        let (y_f, y_f_prime, y_h, y_h_prime) = evals_out[i];
        let f_pts = [(z, y_f), (gz, y_f_prime)];
        let h_pts = [(z, y_h), (gz, y_h_prime)];
        let (rf, _) = build_multi_point_polys(&f_pts);
        let (rh, _) = build_multi_point_polys(&h_pts);

        let qf = poly_sub_div(&f_vs[i].coeffs, &rf, &z_coeffs);
        let qh = poly_sub_div(&h_bars[i].coeffs, &rh, &z_coeffs);

        let pair_len = qf.len().max(qh.len());
        for d in 0..pair_len {
            let f_val = if d < qf.len() {
                qf[d]
            } else {
                E::ScalarField::ZERO
            };
            let h_val = if d < qh.len() {
                qh[d]
            } else {
                E::ScalarField::ZERO
            };
            if d < q_combined.len() {
                q_combined[d] += outer_r_pow * (f_val + inner_r * h_val);
            }
        }
        outer_r_pow *= outer_r;
    }
    drop(_t_quot);

    let _t_commit_q =
        profile::ScopedTimer::new(nv, n, "multi_open_commit_q", max_q_deg + 1, "KZG-MSM-final");
    let pi = pp.commit(&q_combined);
    drop(_t_commit_q);

    end_timer!(open_timer);
    Ok(MulcsBatchProof {
        f_i_eval_at_point_i: evals.to_vec(),
        cm_hbars,
        mulcs_evals: evals_out,
        z,
        pi,
        mu,
        num_polys,
    })
}

pub(crate) fn batch_verify_internal<E: Pairing>(
    vp: &MulcsVerifierParam<E>,
    commitments: &[Commitment<E>],
    points: &[Vec<E::ScalarField>],
    proof: &MulcsBatchProof<E>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError> {
    let num_polys = proof.num_polys;
    let n = 1 << proof.mu;
    let mu = proof.mu;
    let _t_total = profile::ScopedTimer::new(mu, n, "batch_verify_total", num_polys, "total");
    let open_timer = start_timer!(|| "mulcs batch verify");

    // ── Length sanity checks ──
    if commitments.len() != num_polys
        || points.len() != num_polys
        || proof.f_i_eval_at_point_i.len() != num_polys
        || proof.cm_hbars.len() != num_polys
        || proof.mulcs_evals.len() != num_polys
    {
        return Err(PCSError::InvalidProof(
            "length mismatch in batch proof".to_string(),
        ));
    }
    for point in points {
        if point.len() != proof.mu {
            return Err(PCSError::InvalidProof(format!(
                "point length {} != mu {}",
                point.len(),
                proof.mu
            )));
        }
    }

    let _t_ts = profile::ScopedTimer::new(
        mu,
        n,
        "batch_verify_transcript",
        num_polys,
        "absorb-pts-evals-hbars",
    );
    for eval_point in points.iter() {
        transcript.append_serializable_element(b"eval_point", eval_point)?;
    }
    for eval in proof.f_i_eval_at_point_i.iter() {
        transcript.append_field_element(b"eval", eval)?;
    }
    let gamma = vp.gamma;
    for cm_hbar in &proof.cm_hbars {
        transcript.append_serializable_element(b"h", cm_hbar)?;
    }
    drop(_t_ts);

    let _t_fs = profile::ScopedTimer::new(mu, n, "batch_verify_fs", 1, "challenges");
    let z_buf = transcript.get_and_append_challenge_vectors(b"z", 1)?;
    let z = z_buf[0];
    if z != proof.z {
        return Ok(false);
    }
    let gz = gamma * z;
    for (y_f, y_f_prime, y_h, y_h_prime) in &proof.mulcs_evals {
        transcript.append_field_element(b"y_f", y_f)?;
        transcript.append_field_element(b"y_f'", y_f_prime)?;
        transcript.append_field_element(b"y_h", y_h)?;
        transcript.append_field_element(b"y_h'", y_h_prime)?;
    }
    let inner_r_buf = transcript.get_and_append_challenge_vectors(b"inner", 1)?;
    let inner_r = inner_r_buf[0];
    let outer_r_buf = transcript.get_and_append_challenge_vectors(b"outer", 1)?;
    let outer_r = outer_r_buf[0];
    drop(_t_fs);

    // ── Aggregated KZG pairing check ──
    let _t_agg =
        profile::ScopedTimer::new(mu, n, "batch_verify_aggregate_cm", num_polys, "group-ops");
    let mut cm_combined = E::G1::zero();
    let mut outer_r_pow = E::ScalarField::one();
    let s = z + gz;
    let p = z * gz;
    for i in 0..num_polys {
        let (y_f, y_f_prime, y_h, y_h_prime) = proof.mulcs_evals[i];
        let f_pts = [(z, y_f), (gz, y_f_prime)];
        let h_pts = [(z, y_h), (gz, y_h_prime)];
        let (rf, _) = build_multi_point_polys(&f_pts);
        let (rh, _) = build_multi_point_polys(&h_pts);
        let mut r_comb = vec![E::ScalarField::ZERO; 2];
        for j in 0..2 {
            r_comb[j] = rf[j] + inner_r * rh[j];
        }
        let cm_r = vp.g1_one.into_group() * r_comb[0] + vp.g1_x.into_group() * r_comb[1];
        let cm_i = commitments[i].0.into_group() + proof.cm_hbars[i].into_group() * inner_r - cm_r;
        cm_combined += cm_i * outer_r_pow;
        outer_r_pow *= outer_r;
    }
    let zx_g2 = vp.g2_x2.into_group() - vp.g2_x.into_group() * s + vp.g2_one.into_group() * p;
    drop(_t_agg);

    let _t_pair = profile::ScopedTimer::new(mu, n, "batch_verify_pairing", 1, "1-pairing-check");
    if E::pairing(cm_combined.into_affine(), vp.g2_one) != E::pairing(proof.pi, zx_g2.into_affine())
    {
        end_timer!(open_timer);
        return Ok(false);
    }
    drop(_t_pair);

    // ── Per-polynomial Claymore identity checks ──
    let _t_clay = profile::ScopedTimer::new(
        mu,
        n,
        "batch_verify_claymore",
        num_polys,
        "claymore-identity",
    );
    for i in 0..num_polys {
        let (y_f, _y_f_prime, _y_h, y_h_prime) = proof.mulcs_evals[i];
        let y_hbar = proof.mulcs_evals[i].2;
        let ok = check_claymore_identity(
            gamma,
            proof.mu,
            z,
            y_f,
            y_hbar,
            y_h_prime,
            &points[i],
            proof.f_i_eval_at_point_i[i],
        )?;
        if !ok {
            end_timer!(open_timer);
            return Ok(false);
        }
    }
    drop(_t_clay);

    end_timer!(open_timer);
    Ok(true)
}

fn verify_internal<E: Pairing>(
    vp: &MulcsVerifierParam<E>,
    commitment: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &MulcsProof<E>,
) -> Result<bool, PCSError> {
    let n = 1 << proof.mu;
    let _t_pair = profile::ScopedTimer::new(proof.mu, n, "verify_pairing", 1, "single-pairing");
    let gamma = vp.gamma;
    let gz = gamma * proof.z;
    let f_pts = [(proof.z, proof.y_f), (gz, proof.y_f_prime)];
    let h_pts = [(proof.z, proof.y_hbar), (gz, proof.y_hbar_prime)];
    let (rf, _) = build_multi_point_polys(&f_pts);
    let (rh, _) = build_multi_point_polys(&h_pts);
    let mut r_comb = vec![E::ScalarField::zero(); 2];
    for j in 0..2 {
        r_comb[j] = rf[j] + rh[j];
    }
    let cm_r = vp.g1_one.into_group() * r_comb[0] + vp.g1_x.into_group() * r_comb[1];
    let cm_comb = commitment.0.into_group() + proof.cm_hbar.into_group() - cm_r;
    let s = proof.z + gz;
    let p = proof.z * gz;
    let zx_g2 = vp.g2_x2.into_group() - vp.g2_x.into_group() * s + vp.g2_one.into_group() * p;
    let pair_ok =
        E::pairing(cm_comb.into_affine(), vp.g2_one) == E::pairing(proof.pi, zx_g2.into_affine());
    drop(_t_pair);
    if !pair_ok {
        return Ok(false);
    }

    let _t_clay = profile::ScopedTimer::new(proof.mu, n, "verify_claymore", 1, "claymore-identity");
    let result = check_claymore_identity(
        gamma,
        proof.mu,
        proof.z,
        proof.y_f,
        proof.y_hbar,
        proof.y_hbar_prime,
        point,
        *value,
    );
    drop(_t_clay);
    result
}

// ─── Claymore identity check (shared between single and batch verify) ──

/// Verify the Claymore identity for a single opening:
///
///   γ^{N-1}·h̄(z) - h̄(γz) == z^{N-1}·(f_v(z)·eq(r, z⁻¹) - claimed_value)
///
/// Returns `Ok(true)` if the identity holds.
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_std::{test_rng, UniformRand};

    type E = Bls12_381;

    #[test]
    fn test_mulcs_single_commit_open_verify() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [2usize, 4, 6] {
            let srs = MulcsPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
            let (ck, vk) = MulcsPCS::<E>::trim(&srs, None, Some(nv))?;

            let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
            let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

            let com = MulcsPCS::<E>::commit(&ck, &poly)?;
            let (proof, value) = MulcsPCS::<E>::open(&ck, &poly, &point)?;

            assert!(
                MulcsPCS::<E>::verify(&vk, &com, &point, &value, &proof)?,
                "verify failed for nv={nv}"
            );

            let fake_val = Fr::rand(&mut rng);
            if fake_val != value {
                assert!(
                    !MulcsPCS::<E>::verify(&vk, &com, &point, &fake_val, &proof)?,
                    "should reject wrong value nv={nv}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_mulcs_multi_open_verify() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let num_polys = 3;
        let srs = MulcsPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
        let (ck, vk) = MulcsPCS::<E>::trim(&srs, None, Some(nv))?;

        let polys: Vec<Arc<DenseMultilinearExtension<Fr>>> = (0..num_polys)
            .map(|_| Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
            .collect();
        let points: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| (0..nv).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>())
            .collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<Commitment<E>> = polys
            .iter()
            .map(|p| MulcsPCS::<E>::commit(&ck, p).unwrap())
            .collect();

        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;

        let batch_proof = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;

        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;

        assert!(
            MulcsPCS::<E>::batch_verify(&vk, &comms, &points, &batch_proof, &mut tv)?,
            "batch verify failed"
        );

        Ok(())
    }

    #[test]
    fn test_mulcs_multi_open_rejects_wrong_claimed_eval() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let num_polys = 3;
        let srs = MulcsPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
        let (ck, vk) = MulcsPCS::<E>::trim(&srs, None, Some(nv))?;

        let polys: Vec<Arc<DenseMultilinearExtension<Fr>>> = (0..num_polys)
            .map(|_| Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
            .collect();
        let points: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| (0..nv).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>())
            .collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<Commitment<E>> = polys
            .iter()
            .map(|p| MulcsPCS::<E>::commit(&ck, p).unwrap())
            .collect();

        let mut wrong_evals = evals.clone();
        wrong_evals[0] += Fr::ONE; // tamper: wrong claimed eval

        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let batch_proof = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &wrong_evals, &mut tp)?;

        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(
            !MulcsPCS::<E>::batch_verify(&vk, &comms, &points, &batch_proof, &mut tv)?,
            "should reject wrong claimed eval"
        );
        Ok(())
    }

    #[test]
    fn test_mulcs_multi_open_rejects_wrong_point() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let num_polys = 3;
        let srs = MulcsPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
        let (ck, vk) = MulcsPCS::<E>::trim(&srs, None, Some(nv))?;

        let polys: Vec<Arc<DenseMultilinearExtension<Fr>>> = (0..num_polys)
            .map(|_| Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
            .collect();
        let points: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| (0..nv).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>())
            .collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<Commitment<E>> = polys
            .iter()
            .map(|p| MulcsPCS::<E>::commit(&ck, p).unwrap())
            .collect();

        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let batch_proof = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;

        let mut wrong_points = points.clone();
        wrong_points[0][0] += Fr::ONE; // tamper: wrong point

        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(
            !MulcsPCS::<E>::batch_verify(&vk, &comms, &wrong_points, &batch_proof, &mut tv)?,
            "should reject wrong point"
        );
        Ok(())
    }

    #[test]
    fn test_mulcs_multi_open_rejects_wrong_commitment() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let num_polys = 3;
        let srs = MulcsPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
        let (ck, vk) = MulcsPCS::<E>::trim(&srs, None, Some(nv))?;

        let polys: Vec<Arc<DenseMultilinearExtension<Fr>>> = (0..num_polys)
            .map(|_| Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
            .collect();
        let points: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| (0..nv).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>())
            .collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<Commitment<E>> = polys
            .iter()
            .map(|p| MulcsPCS::<E>::commit(&ck, p).unwrap())
            .collect();

        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let batch_proof = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;

        let extra_poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
        let extra_com = MulcsPCS::<E>::commit(&ck, &extra_poly)?;
        let mut wrong_comms = comms.clone();
        wrong_comms[0] = extra_com; // tamper: wrong commitment

        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(
            !MulcsPCS::<E>::batch_verify(&vk, &wrong_comms, &points, &batch_proof, &mut tv)?,
            "should reject wrong commitment"
        );
        Ok(())
    }

    #[test]
    fn test_mulcs_batch_verify_rejects_malformed_lengths() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let num_polys = 3;
        let srs = MulcsPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
        let (ck, vk) = MulcsPCS::<E>::trim(&srs, None, Some(nv))?;

        let polys: Vec<Arc<DenseMultilinearExtension<Fr>>> = (0..num_polys)
            .map(|_| Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
            .collect();
        let points: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| (0..nv).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>())
            .collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<Commitment<E>> = polys
            .iter()
            .map(|p| MulcsPCS::<E>::commit(&ck, p).unwrap())
            .collect();

        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let mut batch_proof = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;

        // Malform: remove one commitment from verifier's list
        let short_comms = &comms[..2];
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        let result = MulcsPCS::<E>::batch_verify(&vk, short_comms, &points, &batch_proof, &mut tv);
        assert!(
            result.is_err() || !result.unwrap(),
            "should reject malformed batch proof (wrong num commitments)"
        );

        // Malform: shorten cm_hbars
        let wrong_num = batch_proof.num_polys - 1;
        batch_proof.num_polys = 0; // clearly wrong
        let mut tv2 = IOPTranscript::new(b"test");
        tv2.append_field_element(b"init", &Fr::ZERO)?;
        let result2 = MulcsPCS::<E>::batch_verify(&vk, &comms, &points, &batch_proof, &mut tv2);
        assert!(
            result2.is_err() || !result2.unwrap(),
            "should reject malformed batch proof (wrong num_polys)"
        );
        batch_proof.num_polys = wrong_num; // restore to trigger length mismatch
        let mut tv3 = IOPTranscript::new(b"test");
        tv3.append_field_element(b"init", &Fr::ZERO)?;
        let result3 = MulcsPCS::<E>::batch_verify(&vk, &comms, &points, &batch_proof, &mut tv3);
        assert!(
            result3.is_err() || !result3.unwrap(),
            "should reject malformed batch proof (length mismatch)"
        );

        Ok(())
    }
}
