//! Mulcs PCS implementation — Claymore identity-based multilinear PCS
//! using univariate KZG as the black-box commitment scheme.
//!
//! Batch opening: same-point openings are combined via Fiat-Shamir random
//! combination before generating a single Mulcs proof per group, reducing
//! the number of h̄ commitments and KZG quotients.

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

/// Batch-opening proof. Supports same-point random-combination batching:
/// openings sharing the same evaluation point are combined into one group
/// via Fiat-Shamir challenges, producing a single Mulcs opening per group.
///
/// `f_i_eval_at_point_i` preserves the original per-opening eval order
/// (required by `HasEvals` for HyperPlonk verifier).
/// `cm_hbars` and `mulcs_evals` have length `num_groups`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MulcsBatchProof<E: Pairing> {
    /// Evaluations at each (poly, point) pair, in insertion order.
    pub f_i_eval_at_point_i: Vec<E::ScalarField>,
    /// Number of per-opening entries (= f_i_eval_at_point_i.len()).
    pub num_openings: usize,
    /// Number of point groups.
    pub num_groups: usize,
    /// Size of each group (sum = num_openings).
    pub group_sizes: Vec<usize>,
    /// Commitment to h̄ per group.
    pub cm_hbars: Vec<E::G1Affine>,
    /// Mulcs-specific evaluations per group: (y_f, y_f', y_h, y_h').
    pub mulcs_evals: Vec<(
        E::ScalarField,
        E::ScalarField,
        E::ScalarField,
        E::ScalarField,
    )>,
    /// Challenge z from Fiat-Shamir.
    pub z: E::ScalarField,
    /// Aggregated KZG batch proof.
    pub pi: E::G1Affine,
    /// log2(N).
    pub mu: usize,
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
        let delta = E::ScalarField::one();
        let h_bar = UnivarPoly::compute_h_bar(&h, gamma, n, delta);
        let cm_hbar = pp.commit(&h_bar.coeffs);

        let z = E::ScalarField::from(2u64);
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

// ─── Point grouping ──────────────────────────────────────────────

/// Deterministically group opening indices by identical evaluation point.
/// Returns (groups, group_sizes) where groups\[g\] is a Vec of (original_index,
/// point).
fn group_by_point<F: Field>(points: &[Vec<F>]) -> (Vec<Vec<(usize, Vec<F>)>>, Vec<usize>) {
    let mut seen: Vec<(Vec<F>, Vec<usize>)> = Vec::new();
    for (i, pt) in points.iter().enumerate() {
        let mut found = false;
        for (s_pt, s_idxs) in &mut seen {
            if s_pt == pt {
                s_idxs.push(i);
                found = true;
                break;
            }
        }
        if !found {
            seen.push((pt.clone(), vec![i]));
        }
    }
    let mut groups = Vec::new();
    let mut group_sizes = Vec::new();
    for (pt, idxs) in seen {
        group_sizes.push(idxs.len());
        groups.push(idxs.into_iter().map(|i| (i, pt.clone())).collect());
    }
    (groups, group_sizes)
}

// ─── KZG multi-point opening ─────────────────────────────────────

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

// ─── multi_open_internal (group-based) ───────────────────────────

pub(crate) fn multi_open_internal<E: Pairing>(
    pp: &MulcsProverParam<E>,
    polynomials: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    points: &[Vec<E::ScalarField>],
    evals: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<MulcsBatchProof<E>, PCSError> {
    // ── Input validation ──
    if polynomials.is_empty() {
        return Err(PCSError::InvalidParameters(
            "empty polynomial list".to_string(),
        ));
    }
    if polynomials.len() != points.len() {
        return Err(PCSError::InvalidParameters(format!(
            "polynomials.len {} != points.len {}",
            polynomials.len(),
            points.len()
        )));
    }
    if polynomials.len() != evals.len() {
        return Err(PCSError::InvalidParameters(format!(
            "polynomials.len {} != evals.len {}",
            polynomials.len(),
            evals.len()
        )));
    }
    let nv = polynomials[0].num_vars;
    for poly in polynomials.iter() {
        if poly.num_vars != nv {
            return Err(PCSError::InvalidParameters(format!(
                "inconsistent num_vars: {} vs {}",
                poly.num_vars, nv
            )));
        }
    }
    for pt in points.iter() {
        if pt.len() != nv {
            return Err(PCSError::InvalidParameters(format!(
                "point len {} != nv {nv}",
                pt.len()
            )));
        }
    }
    let n = 1 << nv;
    let mu = nv;
    let gamma = pp.gamma;
    let num_openings = polynomials.len();
    let _t_total = profile::ScopedTimer::new(nv, n, "multi_open_total", num_openings, "total");
    let open_timer = start_timer!(|| format!("mulcs multi open {} points", points.len()));

    // Phase 0: absorb points and evals (same order as always)
    let _t_app = profile::ScopedTimer::new(
        nv,
        n,
        "multi_open_append_pts_evals",
        num_openings,
        "transcript",
    );
    for eval_point in points.iter() {
        transcript.append_serializable_element(b"eval_point", eval_point)?;
    }
    for eval in evals.iter() {
        transcript.append_field_element(b"eval", eval)?;
    }
    drop(_t_app);

    // Phase 1: group by same point
    let _t_group = profile::ScopedTimer::new(
        nv,
        n,
        "multi_open_group_points",
        num_openings,
        "deterministic-grouping",
    );
    let (groups, group_sizes) = group_by_point(points);
    let num_groups = groups.len();
    drop(_t_group);

    if profile::profiling_enabled() {
        println!(
            "# point groups: num_openings={num_openings} num_groups={num_groups} sizes={:?}",
            group_sizes
        );
    }
    let _t_group_info = profile::ScopedTimer::new(
        nv,
        n,
        "multi_open_group_info",
        num_groups,
        "absorb-group-meta",
    );
    // Absorb group metadata into transcript
    transcript.append_field_element(b"num_groups", &E::ScalarField::from(num_groups as u64))?;
    for (g, group) in groups.iter().enumerate() {
        for &(idx, _) in group.iter() {
            transcript.append_field_element(b"group_idx", &E::ScalarField::from(idx as u64))?;
        }
        transcript.append_field_element(b"group_end", &E::ScalarField::from(g as u64))?;
    }
    drop(_t_group_info);

    // Phase 2: For each group, derive alpha, combine polynomials/evals, compute ONE
    // Mulcs opening
    let _t_pergroup = profile::ScopedTimer::new(
        nv,
        n,
        "multi_open_per_group",
        num_groups,
        "combined-h-hbar-commit",
    );
    let mut cm_hbars = Vec::with_capacity(num_groups);
    let mut f_vs = Vec::with_capacity(num_groups);
    let mut h_bars = Vec::with_capacity(num_groups);

    let mut t_combine = profile::MaybeTimer::new();
    let mut t_h = profile::MaybeTimer::new();
    let mut t_hbar = profile::MaybeTimer::new();
    let mut t_commit = profile::MaybeTimer::new();

    for group in &groups {
        let _group_size = group.len();
        let point = &group[0].1;

        let alpha_buf = transcript.get_and_append_challenge_vectors(b"group_alpha", 1)?;
        let alpha_base = alpha_buf[0];

        let tick = t_combine.start();
        let mut combined_coeffs = vec![E::ScalarField::zero(); n];
        let mut combined_eval = E::ScalarField::zero();
        let mut alpha_pow = E::ScalarField::one();
        for &(idx, _) in group.iter() {
            let poly_coeffs = polynomials[idx].to_evaluations();
            for k in 0..n {
                combined_coeffs[k] += alpha_pow * poly_coeffs[k];
            }
            combined_eval += alpha_pow * evals[idx];
            alpha_pow *= alpha_base;
        }
        t_combine.add(&tick);

        let f_v = UnivarPoly::new(combined_coeffs);
        let tick = t_h.start();
        let h = UnivarPoly::compute_h(&f_v.coeffs, mu, point, combined_eval);
        t_h.add(&tick);

        let delta = E::ScalarField::one();
        let tick = t_hbar.start();
        let h_bar = UnivarPoly::compute_h_bar(&h, gamma, n, delta);
        t_hbar.add(&tick);

        let tick = t_commit.start();
        let cm_hbar = pp.commit(&h_bar.coeffs);
        t_commit.add(&tick);

        transcript.append_serializable_element(b"h", &cm_hbar)?;
        f_vs.push(f_v);
        h_bars.push(h_bar);
        cm_hbars.push(cm_hbar);
    }
    // Emit fine-grained per-group sub-phase timers
    profile::emit_manual(
        nv,
        n,
        "multi_open_combine_polys",
        t_combine.ns() as f64 / 1_000_000.0,
        num_openings,
        "random-combination",
    );
    profile::emit_manual(
        nv,
        n,
        "multi_open_compute_h",
        t_h.ns() as f64 / 1_000_000.0,
        num_groups,
        "compute-h",
    );
    profile::emit_manual(
        nv,
        n,
        "multi_open_compute_h_bar",
        t_hbar.ns() as f64 / 1_000_000.0,
        num_groups,
        "compute-hbar",
    );
    profile::emit_manual(
        nv,
        n,
        "multi_open_commit_hbar",
        t_commit.ns() as f64 / 1_000_000.0,
        num_groups,
        "commit-hbar",
    );
    drop(_t_pergroup);

    // Phase 3: Fiat-Shamir z
    let _t_fs = profile::ScopedTimer::new(nv, n, "multi_open_fs_z", 1, "transcript-challenge");
    let z_buf = transcript.get_and_append_challenge_vectors(b"z", 1)?;
    let z = z_buf[0];
    drop(_t_fs);
    let gz = gamma * z;

    // Phase 4: Evaluate per group at z, gz
    let _t_evals =
        profile::ScopedTimer::new(nv, n, "multi_open_eval_zgz", num_groups, "eval-f-hbar");
    let mut evals_out = Vec::with_capacity(num_groups);
    for g in 0..num_groups {
        let y_f = f_vs[g].evaluate(z);
        let y_f_prime = f_vs[g].evaluate(gz);
        let y_h = h_bars[g].evaluate(z);
        let y_h_prime = h_bars[g].evaluate(gz);
        evals_out.push((y_f, y_f_prime, y_h, y_h_prime));
        transcript.append_field_element(b"y_f", &y_f)?;
        transcript.append_field_element(b"y_f'", &y_f_prime)?;
        transcript.append_field_element(b"y_h", &y_h)?;
        transcript.append_field_element(b"y_h'", &y_h_prime)?;
    }
    drop(_t_evals);

    // Phase 5: Fiat-Shamir inner/outer challenges
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

    // Phase 6: KZG quotient construction (over g groups, NOT per-opening)
    let _t_quot = profile::ScopedTimer::new(
        nv,
        n,
        "multi_open_quotient_construction",
        num_groups,
        "poly-div",
    );
    let dummy_pts = [(z, E::ScalarField::zero()), (gz, E::ScalarField::zero())];
    let (_, z_coeffs) = build_multi_point_polys(&dummy_pts);
    let z_deg = z_coeffs.len().saturating_sub(1);

    let max_q_deg = (0..num_groups)
        .map(|g| {
            let fq = f_vs[g].coeffs.len().saturating_sub(1).saturating_sub(z_deg);
            let hq = h_bars[g]
                .coeffs
                .len()
                .saturating_sub(1)
                .saturating_sub(z_deg);
            fq.max(hq)
        })
        .max()
        .unwrap_or(0);

    let mut q_combined = vec![E::ScalarField::zero(); max_q_deg + 1];
    let mut outer_r_pow = E::ScalarField::one();

    for g in 0..num_groups {
        let (y_f, y_f_prime, y_h, y_h_prime) = evals_out[g];
        let f_pts = [(z, y_f), (gz, y_f_prime)];
        let h_pts = [(z, y_h), (gz, y_h_prime)];
        let (rf, _) = build_multi_point_polys(&f_pts);
        let (rh, _) = build_multi_point_polys(&h_pts);
        let qf = poly_sub_div(&f_vs[g].coeffs, &rf, &z_coeffs);
        let qh = poly_sub_div(&h_bars[g].coeffs, &rh, &z_coeffs);
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
        num_openings,
        num_groups,
        group_sizes,
        cm_hbars,
        mulcs_evals: evals_out,
        z,
        pi,
        mu,
    })
}

// ─── batch_verify_internal (group-based) ─────────────────────────

pub(crate) fn batch_verify_internal<E: Pairing>(
    vp: &MulcsVerifierParam<E>,
    commitments: &[Commitment<E>],
    points: &[Vec<E::ScalarField>],
    proof: &MulcsBatchProof<E>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError> {
    let num_openings = proof.num_openings;
    let num_groups = proof.num_groups;
    let mu = proof.mu;
    let n = 1 << mu;
    let _t_total = profile::ScopedTimer::new(mu, n, "batch_verify_total", num_groups, "total");
    let open_timer = start_timer!(|| "mulcs batch verify");

    // ── Length sanity checks ──
    if num_openings == 0 {
        return Err(PCSError::InvalidProof("empty batch proof".to_string()));
    }
    if commitments.len() != num_openings
        || points.len() != num_openings
        || proof.f_i_eval_at_point_i.len() != num_openings
        || proof.cm_hbars.len() != num_groups
        || proof.mulcs_evals.len() != num_groups
        || proof.group_sizes.len() != num_groups
        || proof.group_sizes.iter().sum::<usize>() != num_openings
    {
        return Err(PCSError::InvalidProof(
            "length mismatch in batch proof".to_string(),
        ));
    }
    for point in points {
        if point.len() != mu {
            return Err(PCSError::InvalidProof(format!(
                "point length {} != mu {}",
                point.len(),
                mu
            )));
        }
    }

    // ── Reconstruct groups from points (deterministic, same as prover) ──
    let (groups, _sizes) = group_by_point(points);
    if groups.len() != num_groups {
        return Err(PCSError::InvalidProof("group count mismatch".to_string()));
    }
    for (g, group) in groups.iter().enumerate() {
        if group.len() != proof.group_sizes[g] {
            return Err(PCSError::InvalidProof(format!(
                "group {g} size mismatch: expected {}, got {}",
                proof.group_sizes[g],
                group.len()
            )));
        }
    }

    // ── Transcript replay: absorb points + evals + group metadata ──
    let _t_ts = profile::ScopedTimer::new(
        mu,
        n,
        "batch_verify_transcript",
        num_openings,
        "absorb-pts-evals-meta",
    );
    for eval_point in points.iter() {
        transcript.append_serializable_element(b"eval_point", eval_point)?;
    }
    for eval in proof.f_i_eval_at_point_i.iter() {
        transcript.append_field_element(b"eval", eval)?;
    }
    transcript.append_field_element(b"num_groups", &E::ScalarField::from(num_groups as u64))?;
    for (g, group) in groups.iter().enumerate() {
        for &(idx, _) in group.iter() {
            transcript.append_field_element(b"group_idx", &E::ScalarField::from(idx as u64))?;
        }
        transcript.append_field_element(b"group_end", &E::ScalarField::from(g as u64))?;
    }
    drop(_t_ts);

    let gamma = vp.gamma;

    // Derive alpha challenges for each group AND absorb cm_hbars (interleaved with
    // prover)
    let _t_fs =
        profile::ScopedTimer::new(mu, n, "batch_verify_fs_alpha", num_groups, "group-alphas");
    let mut group_alphas: Vec<(E::ScalarField, Vec<E::ScalarField>)> =
        Vec::with_capacity(num_groups);
    for (g, group) in groups.iter().enumerate() {
        let group_size = group.len();
        let alpha_buf = transcript.get_and_append_challenge_vectors(b"group_alpha", 1)?;
        let alpha_base = alpha_buf[0];
        let mut alpha_pows = Vec::with_capacity(group_size);
        let mut ap = E::ScalarField::one();
        for _ in 0..group_size {
            alpha_pows.push(ap);
            ap *= alpha_base;
        }
        group_alphas.push((alpha_base, alpha_pows));
        // Absorb cm_hbar for this group (prover absorbed it here too)
        transcript.append_serializable_element(b"h", &proof.cm_hbars[g])?;
    }
    drop(_t_fs);
    let _t_fs2 =
        profile::ScopedTimer::new(mu, n, "batch_verify_fs_z", 1, "z-inner-outer-challenges");
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
    drop(_t_fs2);

    // ── Step: Combine commitments per group (using same alphas) ──
    // C_g = sum alpha_g^j * commitments[idx]  for all openings in the group
    let _t_comb = profile::ScopedTimer::new(
        mu,
        n,
        "batch_verify_combine_comms",
        num_openings,
        "G1-scalar-mul",
    );
    let mut grouped_commitments: Vec<E::G1> = Vec::with_capacity(num_groups);
    let mut grouped_evals: Vec<E::ScalarField> = Vec::with_capacity(num_groups);
    for (g, group) in groups.iter().enumerate() {
        let alpha_pows = &group_alphas[g].1;
        let mut c_g = E::G1::zero();
        let mut y_g = E::ScalarField::zero();
        for (j, &(idx, _)) in group.iter().enumerate() {
            c_g += commitments[idx].0.into_group() * alpha_pows[j];
            y_g += alpha_pows[j] * proof.f_i_eval_at_point_i[idx];
        }
        grouped_commitments.push(c_g);
        grouped_evals.push(y_g);
    }
    drop(_t_comb);

    // ── Aggregated KZG pairing check (over groups) ──
    let _t_agg =
        profile::ScopedTimer::new(mu, n, "batch_verify_aggregate_cm", num_groups, "group-ops");
    let inv_dz = (z - gz)
        .inverse()
        .ok_or_else(|| PCSError::InvalidParameters("z == gamma*z".to_string()))?;
    let w_fz_c0 = -gz * inv_dz;
    let w_fz_c1 = inv_dz;
    let w_fgz_c0 = z * inv_dz;
    let w_fgz_c1 = -inv_dz;
    let w_hz_c0 = -gz * inv_dz;
    let w_hz_c1 = inv_dz;
    let w_hgz_c0 = z * inv_dz;
    let w_hgz_c1 = -inv_dz;

    let mut cm_combined = E::G1::zero();
    let mut outer_r_pow = E::ScalarField::one();
    let s = z + gz;
    let p = z * gz;
    for g in 0..num_groups {
        let (y_f, y_f_prime, y_h, y_h_prime) = proof.mulcs_evals[g];
        let rf_c0 = w_fz_c0 * y_f + w_fgz_c0 * y_f_prime;
        let rf_c1 = w_fz_c1 * y_f + w_fgz_c1 * y_f_prime;
        let rh_c0 = w_hz_c0 * y_h + w_hgz_c0 * y_h_prime;
        let rh_c1 = w_hz_c1 * y_h + w_hgz_c1 * y_h_prime;
        let r_comb0 = rf_c0 + inner_r * rh_c0;
        let r_comb1 = rf_c1 + inner_r * rh_c1;

        let cm_r = vp.g1_one.into_group() * r_comb0 + vp.g1_x.into_group() * r_comb1;
        let cm_i = grouped_commitments[g] + proof.cm_hbars[g].into_group() * inner_r - cm_r;
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

    // ── Per-group Claymore identity checks ──
    let _t_clay = profile::ScopedTimer::new(
        mu,
        n,
        "batch_verify_claymore",
        num_groups,
        "claymore-identity",
    );
    for (g, group) in groups.iter().enumerate() {
        let (y_f, _, _, y_h_prime) = proof.mulcs_evals[g];
        let y_hbar = proof.mulcs_evals[g].2;
        let point = &group[0].1;
        let claimed = grouped_evals[g];
        let ok = check_claymore_identity(gamma, mu, z, y_f, y_hbar, y_h_prime, point, claimed)?;
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

// ─── Claymore identity check ─────────────────────────────────────

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

// ─── Polynomial utilities ────────────────────────────────────────

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
                "verify failed nv={nv}"
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
    fn test_mulcs_grouping_same_point() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let point = random_point(nv, &mut rng);
        let num_polys = 4;
        let polys: Vec<_> = (0..num_polys).map(|_| random_poly(nv, &mut rng)).collect();
        let points: Vec<_> = (0..num_polys).map(|_| point.clone()).collect();
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
        assert_eq!(proof.num_groups, 1, "all same point should produce 1 group");
        assert_eq!(proof.cm_hbars.len(), 1);
        assert_eq!(proof.mulcs_evals.len(), 1);

        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(MulcsPCS::<E>::batch_verify(
            &vk, &comms, &points, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_mulcs_grouping_multiple_points() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let pt_a = random_point(nv, &mut rng);
        let pt_b = random_point(nv, &mut rng);
        let points = vec![
            pt_a.clone(),
            pt_a.clone(),
            pt_b.clone(),
            pt_b.clone(),
            pt_b.clone(),
        ];
        let polys: Vec<_> = points.iter().map(|_| random_poly(nv, &mut rng)).collect();
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
        assert_eq!(proof.num_groups, 2);
        assert_eq!(proof.cm_hbars.len(), 2);
        assert_eq!(proof.group_sizes, vec![2, 3]);

        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(MulcsPCS::<E>::batch_verify(
            &vk, &comms, &points, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_mulcs_grouping_matches_single_open() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let point = random_point(nv, &mut rng);
        let poly = random_poly(nv, &mut rng);
        let points = vec![point.clone()];
        let polys = vec![poly.clone()];
        let evals = vec![poly.evaluate(&point).unwrap()];
        let com = MulcsPCS::<E>::commit(&ck, &poly)?;

        let mut tp = IOPTranscript::new(b"test");
        tp.append_field_element(b"init", &Fr::ZERO)?;
        let proof = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        assert_eq!(proof.num_groups, 1);

        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        assert!(MulcsPCS::<E>::batch_verify(
            &vk,
            &[com],
            &points,
            &proof,
            &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_mulcs_grouping_reject_wrong_eval() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let point = random_point(nv, &mut rng);
        let polys: Vec<_> = (0..3).map(|_| random_poly(nv, &mut rng)).collect();
        let points: Vec<_> = vec![point.clone(), point.clone(), point.clone()];
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
        assert!(!MulcsPCS::<E>::batch_verify(
            &vk, &comms, &points, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_mulcs_grouping_reject_wrong_point() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let point = random_point(nv, &mut rng);
        let polys: Vec<_> = (0..3).map(|_| random_poly(nv, &mut rng)).collect();
        let points: Vec<_> = vec![point.clone(), point.clone(), point.clone()];
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
        wrong_points[0] = random_point(nv, &mut rng); // wrong point → changes grouping
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        let result = MulcsPCS::<E>::batch_verify(&vk, &comms, &wrong_points, &proof, &mut tv);
        assert!(
            result.is_err() || !result.unwrap(),
            "should reject wrong point with grouping"
        );
        Ok(())
    }

    #[test]
    fn test_mulcs_multi_open_verify() -> Result<(), PCSError> {
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
    fn test_mulcs_multi_open_rejects_wrong_claimed_eval() -> Result<(), PCSError> {
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
        assert!(!MulcsPCS::<E>::batch_verify(
            &vk, &comms, &points, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_mulcs_multi_open_rejects_wrong_point() -> Result<(), PCSError> {
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
        assert!(!MulcsPCS::<E>::batch_verify(
            &vk,
            &comms,
            &wrong_points,
            &proof,
            &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_mulcs_multi_open_rejects_wrong_commitment() -> Result<(), PCSError> {
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
    fn test_mulcs_batch_verify_rejects_malformed_lengths() -> Result<(), PCSError> {
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
        proof.num_openings = 0;
        let mut tv2 = IOPTranscript::new(b"test");
        tv2.append_field_element(b"init", &Fr::ZERO)?;
        let r2 = MulcsPCS::<E>::batch_verify(&vk, &comms, &points, &proof, &mut tv2);
        assert!(r2.is_err() || !r2.unwrap());

        // Tamper: group_sizes.len() != num_groups
        let mut proof2 = MulcsPCS::<E>::multi_open(
            &ck,
            &polys,
            &points,
            &evals,
            &mut IOPTranscript::new(b"t2"),
        )?;
        proof2.group_sizes.pop(); // too few
        let mut tv3 = IOPTranscript::new(b"test");
        tv3.append_field_element(b"init", &Fr::ZERO)?;
        let r3 = MulcsPCS::<E>::batch_verify(&vk, &comms, &points, &proof2, &mut tv3);
        assert!(
            r3.is_err() || !r3.unwrap(),
            "should reject group_sizes too short"
        );

        // Tamper: num_groups changed
        let mut proof3 = MulcsPCS::<E>::multi_open(
            &ck,
            &polys,
            &points,
            &evals,
            &mut IOPTranscript::new(b"t3"),
        )?;
        proof3.num_groups = 99;
        let mut tv4 = IOPTranscript::new(b"test");
        tv4.append_field_element(b"init", &Fr::ZERO)?;
        let r4 = MulcsPCS::<E>::batch_verify(&vk, &comms, &points, &proof3, &mut tv4);
        assert!(
            r4.is_err() || !r4.unwrap(),
            "should reject wrong num_groups"
        );

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
        assert!(r.is_err(), "should reject empty polynomial list");
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
        assert!(r.is_err(), "should reject mismatched points vs polys");
        let mut tp = IOPTranscript::new(b"test");
        let r = MulcsPCS::<E>::multi_open(
            &ck,
            &[poly.clone()],
            &[point.clone()],
            &[Fr::one(), Fr::one()],
            &mut tp,
        );
        assert!(r.is_err(), "should reject mismatched evals vs polys");
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
        assert!(r.is_err(), "should reject inconsistent num_vars");
    }

    #[test]
    fn test_mulcs_multi_open_rejects_wrong_point_len() {
        let (ck, _vk) = setup(4);
        let mut rng = test_rng();
        let poly = random_poly(4, &mut rng);
        let short_point = random_point(3, &mut rng);
        let mut tp = IOPTranscript::new(b"test");
        let r = MulcsPCS::<E>::multi_open(&ck, &[poly], &[short_point], &[Fr::one()], &mut tp);
        assert!(r.is_err(), "should reject point.len() != nv");
    }

    fn assert_rejects(backend: &str, result: Result<bool, PCSError>) {
        match result {
            Ok(true) => panic!("{backend}: expected reject but got true"),
            Ok(false) => {}, // ok
            Err(e) => eprintln!("# {backend} reject with error: {e:?}"),
        }
    }

    #[test]
    fn test_mulcs_batch_verify_rejects_group_sizes_too_long() -> Result<(), PCSError> {
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
        let mut proof = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        // group_sizes.len() > num_groups
        proof.group_sizes.push(1);
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        let r = MulcsPCS::<E>::batch_verify(&vk, &comms, &points, &proof, &mut tv);
        assert_rejects("group_sizes_too_long", r);
        Ok(())
    }

    #[test]
    fn test_mulcs_batch_verify_rejects_group_sizes_sum_mismatch() -> Result<(), PCSError> {
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
        let mut proof = MulcsPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
        // group_sizes sum == num_openings but len != num_groups: add extra entry
        proof.group_sizes = vec![2, 1]; // correct lengths, same num_groups → should pass
        proof.group_sizes = vec![2]; // wrong: len=1 != num_groups=3 (but sum=2 != num_openings=3)
        proof.group_sizes = vec![3]; // len=1 != num_groups=3, sum=3 == num_openings=3
        let mut tv = IOPTranscript::new(b"test");
        tv.append_field_element(b"init", &Fr::ZERO)?;
        let r = MulcsPCS::<E>::batch_verify(&vk, &comms, &points, &proof, &mut tv);
        assert_rejects("group_sizes_sum_ok_len_mismatch", r);
        Ok(())
    }
}
