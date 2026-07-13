//! Zeromorph PCS — multilinear PCS based on univariate KZG.
//! Reference: han0110/plonkish (MIT-licensed).
//!
//! Uses two SRS slices: commit_powers (for poly+quotients+q_hat) and
//! open_powers (shifted by offset=N, for final KZG opening proof).

use crate::{
    pcs::{
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
use ark_ff::{batch_inversion, Field};
use ark_poly::MultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::Borrow, marker::PhantomData, rand::Rng, string::ToString, sync::Arc, vec::Vec, One,
    Zero,
};
use std::{collections::BTreeMap, iter, ops::Deref};
use transcript::IOPTranscript;

use crate::pcs::{multilinear_kzg::batching::BatchProof, profile};
use srs::{ZeromorphProverParam, ZeromorphUniversalParams, ZeromorphVerifierParam};

pub(crate) mod srs;

pub struct ZeromorphPCS<E: Pairing> {
    phantom: PhantomData<E>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct ZeromorphProof<E: Pairing> {
    pub q_comms: Vec<E::G1Affine>,
    pub q_hat_comm: E::G1Affine,
    pub kzg_proof: E::G1Affine,
}

impl<E: Pairing> PolynomialCommitmentScheme<E> for ZeromorphPCS<E> {
    type ProverParam = ZeromorphProverParam<E>;
    type VerifierParam = ZeromorphVerifierParam<E>;
    type SRS = ZeromorphUniversalParams<E>;
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    type Commitment = Commitment<E>;
    type Proof = ZeromorphProof<E>;
    type BatchProof = BatchProof<E, Self>;

    fn gen_srs_for_testing<R: Rng>(rng: &mut R, s: usize) -> Result<Self::SRS, PCSError> {
        ZeromorphUniversalParams::<E>::gen_srs_for_testing(rng, s)
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
        let n = 1 << poly.num_vars;
        if pp.commit_powers.len() < n {
            return Err(PCSError::InvalidParameters("poly too large".to_string()));
        }
        let coeffs = poly.to_evaluations();
        Ok(Commitment(pp.commit_commit(&coeffs)))
    }
    fn open(
        pp: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let mut t = IOPTranscript::new(b"zm-open");
        open_with_transcript(pp.borrow(), poly, point, &mut t)
    }
    fn multi_open(
        pp: impl Borrow<Self::ProverParam>,
        polys: &[Self::Polynomial],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Self::BatchProof, PCSError> {
        zm_sumcheck(pp.borrow(), polys, points, evals, transcript)
    }
    fn verify(
        vp: &Self::VerifierParam,
        com: &Self::Commitment,
        point: &Self::Point,
        val: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        let mut t = IOPTranscript::new(b"zm-open");
        verify_with_transcript(vp, com, point, val, proof, &mut t)
    }
    fn batch_verify(
        vp: &Self::VerifierParam,
        coms: &[Self::Commitment],
        points: &[Self::Point],
        bp: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        zm_batch_verify(vp, coms, points, bp, transcript)
    }
}

impl<E: Pairing> ZeromorphPCS<E> {
    /// Open a polynomial at a point given a pre-computed commitment `cm_f`.
    ///
    /// This avoids the N-size MSM recommit that the trait `open` performs
    /// (via `open_with_transcript`).  `commitment` MUST equal
    /// `commit(pp, poly)`.
    pub fn open_with_commitment(
        pp: &ZeromorphProverParam<E>,
        poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
        point: &[E::ScalarField],
        commitment: &Commitment<E>,
    ) -> Result<(ZeromorphProof<E>, E::ScalarField), PCSError> {
        let num_vars = poly.num_vars();
        if point.len() != num_vars {
            return Err(PCSError::InvalidParameters(
                "point length mismatch".to_string(),
            ));
        }
        let mut transcript = IOPTranscript::new(b"zm-open");

        let coeffs = poly.to_evaluations();
        let eval = poly
            .evaluate(point)
            .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;

        transcript.append_serializable_element(b"commitment", &commitment.0)?;
        transcript.append_serializable_element(b"point", &point.to_vec())?;
        transcript.append_field_element(b"eval", &eval)?;

        zeromorph_core_open_prebound(pp, poly, point, &mut transcript, coeffs, eval)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Quotients — exactly matching plonkish quotients()
// ═══════════════════════════════════════════════════════════════════

/// Compute Zeromorph quotients. Returns (quotients, evaluation).
/// quotients[i] has length 2^{NV-1-i}.
/// Order: quotients[0] = q_{NV-1} (longest), quotients[NV-1] = q_0 (shortest).
fn compute_quotients<F: Field>(poly_evals: &[F], point: &[F]) -> (Vec<Vec<F>>, F) {
    let _num_vars = point.len();
    let mut remainder = poly_evals.to_vec();
    let mut quotients: Vec<Vec<F>> = point
        .iter()
        .enumerate()
        .rev() // process from highest variable to lowest
        .map(|(v_idx, &x_i)| {
            let half = 1 << v_idx;
            let (lo, hi) = remainder.split_at_mut(half);
            let mut q = vec![F::zero(); half];
            for j in 0..half {
                q[j] = hi[j] - lo[j];
                lo[j] += q[j] * x_i;
            }
            remainder.truncate(half);
            q // length = 2^{v_idx}
        })
        .collect();
    // Now quotients has: [q_{NV-1}(len 2^{NV-1}), q_{NV-2}(len 2^{NV-2}), ...,
    // q_0(len 1)] plonkish does quotients.reverse() to get [q_0(len 1), ...,
    // q_{NV-1}(len 2^{NV-1})]
    quotients.reverse();
    (quotients, remainder[0])
}

// ═══════════════════════════════════════════════════════════════════
// squares / offsets / scalars — exactly matching plonkish
// ═══════════════════════════════════════════════════════════════════

/// eval_and_quotient_scalars — exactly matching plonkish
fn eval_and_quotient_scalars<F: Field>(y: F, x: F, z: F, u: &[F]) -> (F, Vec<F>) {
    let num_vars = u.len();

    // plonkish arithmetic::squares(x): [x, x^2, x^4, ...]
    let squares_of_x = {
        let mut v = Vec::with_capacity(num_vars + 1);
        let mut cur = x;
        for _ in 0..=num_vars {
            v.push(cur);
            cur = cur.square();
        }
        v
    };

    // offsets_of_x[i] = Π_{j=i+1}^{NV} squares_of_x[j]
    let offsets_of_x = {
        let mut v = Vec::with_capacity(num_vars);
        let mut acc = F::one();
        for s in squares_of_x.iter().rev().skip(1) {
            acc *= s;
            v.push(acc);
        }
        v.reverse();
        v
    };

    // v_numer = squares_of_x[NV] - 1
    let v_numer = squares_of_x[num_vars] - F::one();

    let mut v_denoms: Vec<F> = squares_of_x
        .iter()
        .take(num_vars + 1)
        .map(|s| *s - F::one())
        .collect();
    batch_inversion(&mut v_denoms);
    let vs: Vec<F> = v_denoms.iter().map(|d| v_numer * d).collect();

    // q_scalars[i] = -(y^i * offsets[i] + z * (squares[i+1] * vs[i+1] - u[i] *
    // vs[i]))
    let mut q_scalars = Vec::with_capacity(num_vars);
    let mut y_pow = F::one();
    for i in 0..num_vars {
        let s = -(y_pow * offsets_of_x[i] + z * (squares_of_x[i] * vs[i + 1] - u[i] * vs[i]));
        q_scalars.push(s);
        y_pow *= y;
    }

    (-vs[0] * z, q_scalars)
}

/// Form q_hat(X) = Σ_{idx=0}^{NV-1} y^idx · X^{N - 2^idx} · q_idx(X)
/// where q_idx has length 2^idx.
fn form_q_hat<F: Field>(quotients: &[Vec<F>], y: F, n: usize) -> Vec<F> {
    let mut q_hat = vec![F::zero(); n];
    let mut y_pow = F::one();
    for (idx, q) in quotients.iter().enumerate() {
        // offset = N - 2^idx, q has length 2^idx after compute_quotients reverses
        // the highest-variable-first quotient list.
        let offset = n - (1 << idx);
        debug_assert_eq!(q.len(), 1 << idx);
        for j in 0..q.len() {
            q_hat[offset + j] += y_pow * q[j];
        }
        y_pow *= y;
    }
    q_hat
}

// ═══════════════════════════════════════════════════════════════════
// Transcript-aware single opening
// ═══════════════════════════════════════════════════════════════════

fn open_with_transcript<E: Pairing>(
    pp: &ZeromorphProverParam<E>,
    poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
    point: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<(ZeromorphProof<E>, E::ScalarField), PCSError> {
    let num_vars = poly.num_vars();
    let _n = 1 << num_vars;

    if point.len() != num_vars {
        return Err(PCSError::InvalidParameters(
            "point length mismatch".to_string(),
        ));
    }

    let coeffs = poly.to_evaluations();
    let commit_cm = pp.commit_commit(&coeffs);
    transcript.append_serializable_element(b"commitment", &commit_cm)?;
    transcript.append_serializable_element(b"point", &point.to_vec())?;

    let eval = poly
        .evaluate(point)
        .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;
    transcript.append_field_element(b"eval", &eval)?;

    zeromorph_core_open_prebound(pp, poly, point, transcript, coeffs, eval)
}

/// Core Zeromorph opening: the transcript already has commitment, point, and
/// evaluation bound.
fn zeromorph_core_open_prebound<E: Pairing>(
    pp: &ZeromorphProverParam<E>,
    poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
    point: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
    coeffs: Vec<E::ScalarField>,
    eval: E::ScalarField,
) -> Result<(ZeromorphProof<E>, E::ScalarField), PCSError> {
    let num_vars = poly.num_vars();
    let n = 1 << num_vars;

    // 1. Compute quotients
    let _t_quotients = profile::ScopedTimer::new(
        "Zeromorph",
        num_vars,
        n,
        "zeromorph_open_compute_quotients",
        num_vars,
        "quotients",
    );
    let (quotients, _remainder) = compute_quotients::<E::ScalarField>(&coeffs, point);
    drop(_t_quotients);

    let _t_qcomms = profile::ScopedTimer::new(
        "Zeromorph",
        num_vars,
        n,
        "zeromorph_open_commit_qs",
        num_vars,
        "KZG-commit-qs",
    );
    let q_comms: Vec<E::G1Affine> = quotients.iter().map(|q| pp.commit_commit(q)).collect();
    for qc in &q_comms {
        transcript.append_serializable_element(b"q_comm", qc)?;
    }
    drop(_t_qcomms);

    // 3. Challenge y
    let y = transcript.get_and_append_challenge_vectors(b"y", 1)?[0];

    // 4. Form and commit q_hat
    let _t_fqh = profile::ScopedTimer::new(
        "Zeromorph",
        num_vars,
        n,
        "zeromorph_open_form_q_hat",
        1,
        "form-q-hat",
    );
    let q_hat = form_q_hat::<E::ScalarField>(&quotients, y, n);
    drop(_t_fqh);
    let _t_cqh = profile::ScopedTimer::new(
        "Zeromorph",
        num_vars,
        n,
        "zeromorph_open_commit_q_hat",
        1,
        "KZG-commit-q-hat",
    );
    let q_hat_comm = pp.commit_commit(&q_hat);
    drop(_t_cqh);
    transcript.append_serializable_element(b"q_hat_comm", &q_hat_comm)?;

    // 5. Challenges x, z
    let x = transcript.get_and_append_challenge_vectors(b"x", 1)?[0];
    let z = transcript.get_and_append_challenge_vectors(b"z", 1)?[0];

    // 6. Compute scalars
    let (eval_scalar, q_scalars) = eval_and_quotient_scalars(y, x, z, point);

    // 7. Build f(X) such that f(x) = 0
    let _t_f = profile::ScopedTimer::new(
        "Zeromorph",
        num_vars,
        n,
        "zeromorph_open_build_f",
        1,
        "build-f",
    );
    let mut f = vec![E::ScalarField::zero(); n];
    for i in 0..coeffs.len() {
        f[i] += z * coeffs[i];
    }
    for i in 0..q_hat.len() {
        f[i] += q_hat[i];
    }
    f[0] += eval_scalar * eval;
    for (qi, &scalar) in quotients.iter().zip(q_scalars.iter()) {
        for (j, &v) in qi.iter().enumerate() {
            f[j] += scalar * v;
        }
    }
    drop(_t_f);

    // 8. KZG open f at x using open_pp (shifted SRS)
    let _t_kzg_q = profile::ScopedTimer::new(
        "Zeromorph",
        num_vars,
        n,
        "zeromorph_open_kzg_quotient",
        1,
        "kzg-q",
    );
    let mut q_open = vec![E::ScalarField::zero(); n - 1];
    let mut carry = E::ScalarField::zero();
    for i in (1..n).rev() {
        let term = f[i] + carry;
        q_open[i - 1] = term;
        carry = term * x;
    }
    drop(_t_kzg_q);
    let _t_pi = profile::ScopedTimer::new(
        "Zeromorph",
        num_vars,
        n,
        "zeromorph_open_commit_pi",
        n - 1,
        "KZG-commit-pi",
    );
    let kzg_proof = pp.commit_open(&q_open);
    drop(_t_pi);

    let proof = ZeromorphProof {
        q_comms,
        q_hat_comm,
        kzg_proof,
    };
    Ok((proof, eval))
}

// ═══════════════════════════════════════════════════════════════════
// Transcript-aware single verification
// ═══════════════════════════════════════════════════════════════════

fn verify_with_transcript<E: Pairing>(
    vp: &ZeromorphVerifierParam<E>,
    commitment: &Commitment<E>,
    point: &[E::ScalarField],
    eval: &E::ScalarField,
    proof: &ZeromorphProof<E>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError> {
    let num_vars = point.len();
    if proof.q_comms.len() != num_vars {
        return Err(PCSError::InvalidProof("wrong q_comms count".to_string()));
    }

    // Replay transcript absorption
    transcript.append_serializable_element(b"commitment", &commitment.0)?;
    transcript.append_serializable_element(b"point", &point.to_vec())?;
    transcript.append_field_element(b"eval", eval)?;

    for qc in &proof.q_comms {
        transcript.append_serializable_element(b"q_comm", qc)?;
    }

    let y = transcript.get_and_append_challenge_vectors(b"y", 1)?[0];
    transcript.append_serializable_element(b"q_hat_comm", &proof.q_hat_comm)?;
    let x = transcript.get_and_append_challenge_vectors(b"x", 1)?[0];
    let z = transcript.get_and_append_challenge_vectors(b"z", 1)?[0];

    let (eval_scalar, q_scalars) = eval_and_quotient_scalars(y, x, z, point);

    // Form aggregated commitment c
    let _t_msm = profile::ScopedTimer::new(
        "Zeromorph",
        num_vars,
        1 << num_vars,
        "zeromorph_verify_msm",
        num_vars + 3,
        "G1-MSM",
    );
    let mut scalars = vec![E::ScalarField::one(), z, eval_scalar * eval];
    let mut bases = vec![proof.q_hat_comm, commitment.0, vp.g];
    for (qc, &qs) in proof.q_comms.iter().zip(q_scalars.iter()) {
        scalars.push(qs);
        bases.push(*qc);
    }
    let c = E::G1::msm_unchecked(&bases, &scalars).into_affine();
    drop(_t_msm);

    // Pairing check: e(C, s_offset_g2) == e(π, s_g2 - x*g2)
    let _t_pair = profile::ScopedTimer::new(
        "Zeromorph",
        num_vars,
        1 << num_vars,
        "zeromorph_verify_pairing",
        1,
        "pairing",
    );
    let sx = (vp.s_g2.into_group() - vp.g2.into_group() * x).into_affine();
    let neg_pi = (-proof.kzg_proof.into_group()).into_affine();
    let ok =
        E::multi_pairing([c, neg_pi], [vp.s_offset_g2, sx]) == PairingOutput(E::TargetField::one());
    drop(_t_pair);
    Ok(ok)
}

// ═══════════════════════════════════════════════════════════════════
// Sumcheck batching
// ═══════════════════════════════════════════════════════════════════

fn zm_sumcheck<E: Pairing>(
    pp: &ZeromorphProverParam<E>,
    polynomials: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    points: &[Vec<E::ScalarField>],
    evals: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<BatchProof<E, ZeromorphPCS<E>>, PCSError> {
    if polynomials.is_empty() {
        return Err(PCSError::InvalidParameters(
            "empty polynomial list".to_string(),
        ));
    }
    if polynomials.len() != points.len() || polynomials.len() != evals.len() {
        return Err(PCSError::InvalidParameters("length mismatch".to_string()));
    }
    let num_var = polynomials[0].num_vars;
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

    let k = polynomials.len();
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
            |mut m, ((p, pt), c)| {
                *Arc::make_mut(&mut m[point_indices[pt]]) += (*c, p.deref());
                m
            },
        );

    let tilde_eqs: Vec<_> = deduped_points
        .iter()
        .map(|pt| {
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_var,
                build_eq_x_r_vec(pt).unwrap(),
            ))
        })
        .collect();

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
        let mut eq = E::ScalarField::one();
        for (&ai, &pi) in a2.iter().zip(pt.iter()) {
            eq *= ai * pi + (E::ScalarField::one() - ai) * (E::ScalarField::one() - pi);
        }
        *Arc::make_mut(&mut g_prime) += (eq, g.deref());
    }

    // Open g' at a2 using transcript-aware opening
    let (g_prime_proof, _g_ev) = open_with_transcript(pp, &g_prime, a2, transcript)?;

    Ok(BatchProof {
        sum_check_proof: sumcheck_proof,
        f_i_eval_at_point_i: evals.to_vec(),
        g_prime_proof,
    })
}

fn zm_batch_verify<E: Pairing>(
    vp: &ZeromorphVerifierParam<E>,
    f_i_commitments: &[Commitment<E>],
    points: &[Vec<E::ScalarField>],
    proof: &BatchProof<E, ZeromorphPCS<E>>,
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
        let mut eq = E::ScalarField::one();
        for (&ai, &pi) in a2.iter().zip(pt.iter()) {
            eq *= ai * pi + (E::ScalarField::one() - ai) * (E::ScalarField::one() - pi);
        }
        scalars.push(eq * eq_t_list[i]);
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

    // Verify g' opening with transcript-aware verify (resumes same transcript)
    verify_with_transcript(
        vp,
        &Commitment(g_prime_commit.into_affine()),
        a2,
        &subclaim.expected_evaluation,
        &proof.g_prime_proof,
        transcript,
    )
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

    fn setup(nv: usize) -> (ZeromorphProverParam<E>, ZeromorphVerifierParam<E>) {
        let mut rng = test_rng();
        ZeromorphPCS::<E>::trim(
            &ZeromorphPCS::<E>::gen_srs_for_testing(&mut rng, nv).unwrap(),
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
    fn test_zm_single_open_verify() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [2usize, 4, 6] {
            let (ck, vk) = setup(nv);
            let p = rpoly(nv, &mut rng);
            let pt = rpt(nv, &mut rng);
            let com = ZeromorphPCS::<E>::commit(&ck, &p)?;
            let (proof, val) = ZeromorphPCS::<E>::open(&ck, &p, &pt)?;
            assert!(
                ZeromorphPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?,
                "ZM open/verify nv={nv}"
            );
            let fv = Fr::rand(&mut rng);
            if fv != val {
                assert!(!ZeromorphPCS::<E>::verify(&vk, &com, &pt, &fv, &proof)?);
            }
        }
        Ok(())
    }

    #[test]
    fn test_zm_reject_wrong_point() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = ZeromorphPCS::<E>::commit(&ck, &p)?;
        let (proof, val) = ZeromorphPCS::<E>::open(&ck, &p, &pt)?;
        assert!(!ZeromorphPCS::<E>::verify(
            &vk,
            &com,
            &rpt(nv, &mut rng),
            &val,
            &proof
        )?);
        Ok(())
    }

    #[test]
    fn test_zm_reject_wrong_commitment() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p1 = rpoly(nv, &mut rng);
        let p2 = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com2 = ZeromorphPCS::<E>::commit(&ck, &p2)?;
        let (proof, val) = ZeromorphPCS::<E>::open(&ck, &p1, &pt)?;
        assert!(!ZeromorphPCS::<E>::verify(&vk, &com2, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_zm_reject_wrong_qcomm() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = ZeromorphPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = ZeromorphPCS::<E>::open(&ck, &p, &pt)?;
        proof.q_comms[0] = (proof.q_comms[0].into_group() * Fr::from(2u64)).into_affine();
        assert!(!ZeromorphPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_zm_multi_open_k1() -> Result<(), PCSError> {
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
            .map(|p| ZeromorphPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = ZeromorphPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(ZeromorphPCS::<E>::batch_verify(
            &vk, &comms, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_zm_multi_open_distinct() -> Result<(), PCSError> {
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
            .map(|p| ZeromorphPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = ZeromorphPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(ZeromorphPCS::<E>::batch_verify(
            &vk, &comms, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_zm_multi_open_repeated() -> Result<(), PCSError> {
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
            .map(|p| ZeromorphPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = ZeromorphPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(ZeromorphPCS::<E>::batch_verify(
            &vk, &comms, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_zm_batch_reject_wrong_eval() -> Result<(), PCSError> {
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
            .map(|p| ZeromorphPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        evals[0] += Fr::one();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = ZeromorphPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(!ZeromorphPCS::<E>::batch_verify(
            &vk, &comms, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_zm_batch_reject_malformed() -> Result<(), PCSError> {
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
            .map(|p| ZeromorphPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let mut proof = ZeromorphPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let r = ZeromorphPCS::<E>::batch_verify(
            &vk,
            &comms[..2],
            &pts,
            &proof,
            &mut IOPTranscript::new(b"t"),
        );
        assert!(r.is_err() || !r.unwrap());
        proof.f_i_eval_at_point_i.pop();
        let r = ZeromorphPCS::<E>::batch_verify(
            &vk,
            &comms,
            &pts,
            &proof,
            &mut IOPTranscript::new(b"t"),
        );
        assert!(r.is_err() || !r.unwrap());
        Ok(())
    }

    #[test]
    fn test_zm_reject_tampered_q_hat_comm() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = ZeromorphPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = ZeromorphPCS::<E>::open(&ck, &p, &pt)?;
        proof.q_hat_comm = (proof.q_hat_comm.into_group() * Fr::from(2u64)).into_affine();
        assert!(!ZeromorphPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_zm_reject_tampered_kzg_proof() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = ZeromorphPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = ZeromorphPCS::<E>::open(&ck, &p, &pt)?;
        proof.kzg_proof = (proof.kzg_proof.into_group() * Fr::from(3u64)).into_affine();
        assert!(!ZeromorphPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
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
            let com = ZeromorphPCS::<E>::commit(&ck, &p)?;
            let (proof_a, val_a) = ZeromorphPCS::<E>::open_with_commitment(&ck, &p, &pt, &com)?;
            let (proof_b, val_b) = ZeromorphPCS::<E>::open(&ck, &p, &pt)?;
            assert_eq!(val_a, val_b);
            assert!(ZeromorphPCS::<E>::verify(&vk, &com, &pt, &val_a, &proof_a)?);
            assert!(ZeromorphPCS::<E>::verify(&vk, &com, &pt, &val_b, &proof_b)?);
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
            let com = ZeromorphPCS::<E>::commit(&ck, &p)?;
            let (proof, val) = ZeromorphPCS::<E>::open_with_commitment(&ck, &p, &pt, &com)?;
            assert!(ZeromorphPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
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
            let wrong_com = ZeromorphPCS::<E>::commit(&ck, &p2)?;
            let r = ZeromorphPCS::<E>::open_with_commitment(&ck, &p, &pt, &wrong_com);
            if let Ok((proof, val)) = r {
                let com = ZeromorphPCS::<E>::commit(&ck, &p)?;
                assert!(
                    !ZeromorphPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?,
                    "wrong commitment should not produce verifiable proof"
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
        let com = ZeromorphPCS::<E>::commit(&ck, &p)?;
        let short = rpt(2, &mut rng);
        assert!(ZeromorphPCS::<E>::open_with_commitment(&ck, &p, &short, &com).is_err());
        let long = rpt(8, &mut rng);
        assert!(ZeromorphPCS::<E>::open_with_commitment(&ck, &p, &long, &com).is_err());
        Ok(())
    }
}
