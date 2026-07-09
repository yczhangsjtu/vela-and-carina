//! Samaritan PCS — multilinear PCS based on KZG univariate commitments.
//!
//! Converts multilinear polynomial evaluations to univariate coefficient form,
//! then commits using KZG G1 MSM. The opening protocol splits the point into
//! two parts via mu = kappa + nu, builds structured helper polynomials, and
//! proves a single KZG evaluation at a challenge point delta.

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

use srs::{SamaritanProverParam, SamaritanUniversalParams, SamaritanVerifierParam};

const BACKEND: &str = "Samaritan";

// ═══════════════════════════════════════════════════════════════════
// Proof / PCS structures
// ═══════════════════════════════════════════════════════════════════

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct SamaritanProof<E: Pairing> {
    pub v_hat_commit: E::G1Affine,
    pub v_gamma: E::ScalarField,
    pub p_hat_commit: E::G1Affine,
    pub b_hat_commit: E::G1Affine,
    pub u_hat_commit: E::G1Affine,
    pub t_hat_commit: E::G1Affine,
    pub s_hat_commit: E::G1Affine,
    pub q_eval_proof: E::G1Affine,
    pub mu: usize,
}

pub struct SamaritanPCS<E: Pairing> {
    phantom: PhantomData<E>,
}

// ═══════════════════════════════════════════════════════════════════
// PolynomialCommitmentScheme impl
// ═══════════════════════════════════════════════════════════════════

impl<E: Pairing> PolynomialCommitmentScheme<E> for SamaritanPCS<E> {
    type ProverParam = SamaritanProverParam<E>;
    type VerifierParam = SamaritanVerifierParam<E>;
    type SRS = SamaritanUniversalParams<E>;
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    type Commitment = Commitment<E>;
    type Proof = SamaritanProof<E>;
    type BatchProof = BatchProof<E, Self>;

    fn gen_srs_for_testing<R: Rng>(rng: &mut R, s: usize) -> Result<Self::SRS, PCSError> {
        SamaritanUniversalParams::<E>::gen_srs_for_testing(rng, s)
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
        let mut transcript = IOPTranscript::new(b"samaritan-open");
        transcript.append_field_element(b"mu", &E::ScalarField::from(mu as u64))?;
        samaritan_open_with_transcript(pp.borrow(), poly, point, &mut transcript)
    }

    fn multi_open(
        pp: impl Borrow<Self::ProverParam>,
        polys: &[Self::Polynomial],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Self::BatchProof, PCSError> {
        samaritan_sumcheck_multi_open(pp.borrow(), polys, points, evals, transcript)
    }

    fn verify(
        vp: &Self::VerifierParam,
        com: &Self::Commitment,
        point: &Self::Point,
        val: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        let mut transcript = IOPTranscript::new(b"samaritan-open");
        transcript.append_field_element(b"mu", &E::ScalarField::from(proof.mu as u64))?;
        samaritan_verify_with_transcript(vp, com, point, val, proof, &mut transcript)
    }

    fn batch_verify(
        vp: &Self::VerifierParam,
        coms: &[Self::Commitment],
        points: &[Self::Point],
        bp: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        samaritan_sumcheck_batch_verify(vp, coms, points, bp, transcript)
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

/// Compute n = 2^mu, kappa = round(log2(mu)), nu = mu - kappa,
/// m = 2^kappa, l = 2^nu. All shifts are checked.
fn checked_split_params(
    mu: usize,
    label: &str,
) -> Result<(usize, usize, usize, usize, usize), PCSError> {
    let n = checked_domain_size_from_mu(mu, label)?;
    let kappa = (mu as f64).log2().round() as usize;
    if kappa >= usize::BITS as usize {
        return Err(PCSError::InvalidProof(format!(
            "{label}: kappa {kappa} overflow"
        )));
    }
    let nu = if mu >= kappa {
        mu - kappa
    } else {
        return Err(PCSError::InvalidProof(format!(
            "{label}: kappa {kappa} > mu {mu}"
        )));
    };
    if nu >= usize::BITS as usize {
        return Err(PCSError::InvalidProof(format!("{label}: nu {nu} overflow")));
    }
    let m = 1usize.checked_shl(kappa as u32).ok_or_else(|| {
        PCSError::InvalidProof(format!("{label}: m shift overflow kappa={kappa}"))
    })?;
    let l = 1usize
        .checked_shl(nu as u32)
        .ok_or_else(|| PCSError::InvalidProof(format!("{label}: l shift overflow nu={nu}")))?;
    Ok((n, kappa, nu, m, l))
}

// ═══════════════════════════════════════════════════════════════════
// Polynomial helpers — simple Vec<F> coefficient operations
// ═══════════════════════════════════════════════════════════════════

fn poly_eval<F: Field>(coeffs: &[F], x: F) -> F {
    let mut result = F::zero();
    for c in coeffs.iter().rev() {
        result = result * x + *c;
    }
    result
}

#[cfg(test)]
fn poly_mul<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut result = vec![F::zero(); a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            if bj.is_zero() {
                continue;
            }
            result[i + j] += ai * bj;
        }
    }
    result
}

fn poly_sub_in_place<F: Field>(dst: &mut [F], src: &[F]) {
    for (i, &c) in src.iter().enumerate() {
        dst[i] -= c;
    }
}

fn poly_scalar_mul<F: Field>(coeffs: &[F], s: F) -> Vec<F> {
    coeffs.iter().map(|c| *c * s).collect()
}

/// KZG quotient: given coeffs of polynomial f where f(point) = 0,
/// compute quotient Q(X) = f(X) / (X - point)
fn kzg_prove_quotient<F: Field>(coeffs: &[F], point: F) -> Vec<F> {
    let n = coeffs.len();
    if n <= 1 {
        return vec![];
    }
    let mut q = vec![F::zero(); n - 1];
    let mut carry = F::zero();
    for i in (1..n).rev() {
        let term = coeffs[i] + carry;
        q[i - 1] = term;
        carry = term * point;
    }
    q
}

/// Add scaled `src` shifted by `shift` into `dst`. Ensures dst is large enough.
fn add_scaled_shifted<F: Field>(dst: &mut Vec<F>, src: &[F], scale: F, shift: usize) {
    if scale.is_zero() {
        return;
    }
    if dst.len() < shift + src.len() {
        dst.resize(shift + src.len(), F::zero());
    }
    for (i, &c) in src.iter().enumerate() {
        if !c.is_zero() {
            dst[shift + i] += scale * c;
        }
    }
}

/// Compute r(X) = v(X) * (ψ(X; z) + α·φ(X; γ, ν)).
///
/// We compute v*ψ and v*φ separately using structured tensor-product
/// multiplication, then combine: r = v*ψ + α·(v*φ).
/// Each structured product is O(l·nu) instead of O(l²).
fn structured_mul_v_psi_plus_alpha_phi<F: Field>(
    v_hat: &[F],
    point_nu: &[F],
    nu: usize,
    gamma: F,
    alpha: F,
) -> Vec<F> {
    let one = F::one();

    // v * ψ  (psi factor: z_i + (1-z_i)·X^{2^i})
    let mut v_psi = v_hat.to_vec();
    for i in 0..nu {
        let shift = 1 << i;
        let z = point_nu[i];
        let new_len = v_psi.len() + shift;
        let mut next = vec![F::zero(); new_len];
        for (j, &cj) in v_psi.iter().enumerate() {
            if !cj.is_zero() {
                next[j] += z * cj;
                next[j + shift] += (one - z) * cj;
            }
        }
        v_psi = next;
    }

    // v * φ  (phi factor: γ^{2^i} + X^{2^i})
    let mut v_phi = v_hat.to_vec();
    let mut gamma_pow = gamma;
    for i in 0..nu {
        let shift = 1 << i;
        let new_len = v_phi.len() + shift;
        let mut next = vec![F::zero(); new_len];
        for (j, &cj) in v_phi.iter().enumerate() {
            if !cj.is_zero() {
                next[j] += gamma_pow * cj;
                next[j + shift] += cj;
            }
        }
        v_phi = next;
        gamma_pow *= gamma_pow;
    }

    // r = v*ψ + α·(v*φ)
    let out_len = v_psi.len().max(v_phi.len());
    let mut result = vec![F::zero(); out_len];
    for (i, &c) in v_psi.iter().enumerate() {
        result[i] += c;
    }
    for (i, &c) in v_phi.iter().enumerate() {
        result[i] += alpha * c;
    }
    result
}

/// Compute r(X) = p(X) · ψ(X; zx)  with ψ factorized as above.
fn structured_mul_p_psi<F: Field>(p_hat: &[F], point_kappa: &[F], kappa: usize) -> Vec<F> {
    let one = F::one();
    let mut cur = p_hat.to_vec();
    for i in 0..kappa {
        let shift = 1 << i;
        let z = point_kappa[i];
        let a = z; // ψ-factor: z + (1-z)·X^{shift}
        let b = one - z;
        let new_len = cur.len() + shift;
        let mut next = vec![F::zero(); new_len];
        for (j, &cj) in cur.iter().enumerate() {
            if !cj.is_zero() {
                next[j] += a * cj;
                next[j + shift] += b * cj;
            }
        }
        cur = next;
    }
    cur
}

/// kappa = round(log2(mu))
#[cfg(test)]
fn compute_kappa(mu: usize) -> usize {
    (mu as f64).log2().round() as usize
}

/// Get evaluation set — partial evaluation of kappa variables (in-place style).
fn get_evaluation_set<F: Field>(poly_evals: &[F], point: &[F], kappa: usize) -> Vec<F> {
    let sz = poly_evals.len();
    let mut evals = poly_evals.to_vec();
    for i in 0..kappa {
        let step = 1 << (i + 1);
        let half = 1 << i;
        for j in (0..sz).step_by(step) {
            evals[j] = (F::one() - point[i]) * evals[j] + point[i] * evals[j + half];
        }
    }
    let step = 1 << kappa;
    let mut res = Vec::with_capacity(sz / step);
    for i in (0..sz).step_by(step) {
        res.push(evals[i]);
    }
    res
}

/// Evaluate ψ̂ at delta: Π_i (z_i + (1-z_i)·delta^{2^i})
fn evaluate_psi_hat_at<F: Field>(point_slice: &[F], delta: F) -> F {
    let mut acc = F::one();
    let mut delta_pow = delta;
    for &z in point_slice {
        acc *= z + (F::one() - z) * delta_pow;
        delta_pow *= delta_pow;
    }
    acc
}

/// Evaluate φ̂(X;γ,ν) at delta: Π_i (γ^{2^i} + delta^{2^i})
fn evaluate_phi_hat_at<F: Field>(gamma: F, delta: F, nu: usize) -> F {
    let mut acc = F::one();
    let mut delta_pow = delta;
    let mut gamma_pow = gamma;
    for _ in 0..nu {
        acc *= gamma_pow + delta_pow;
        gamma_pow *= gamma_pow;
        delta_pow *= delta_pow;
    }
    acc
}

// ═══════════════════════════════════════════════════════════════════
// compute_t_hat — 7-term combination (in-place allocation)
// ═══════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
fn compute_t_hat<F: Field>(
    v_psi_phi_combined: &[F],
    b_hat: &[F],
    eval: F,
    v_gamma: F,
    alpha: F,
    gamma: F,
    p_psi_combined: &[F],
    u_hat: &[F],
    f_hat: &[F],
    p_hat: &[F],
    beta: F,
    l: usize,
    m: usize,
) -> Vec<F> {
    let n = l * m;
    let zero = F::zero();
    let beta2 = beta * beta;
    let beta3 = beta2 * beta;
    let beta4 = beta3 * beta;
    let beta5 = beta4 * beta;
    let beta6 = beta5 * beta;

    let mut t_hat: Vec<F> = vec![];

    // term1: (v_psi_phi_combined - spike*X^(l-1) - b_hat) / X^l → shift 0
    {
        let spike1_val = eval + alpha * v_gamma;
        let len1 = v_psi_phi_combined.len().max(l);
        for i in 0..len1 {
            let mut val = if i < v_psi_phi_combined.len() {
                v_psi_phi_combined[i]
            } else {
                zero
            };
            if i == l.saturating_sub(1) {
                val -= spike1_val;
            }
            if i < b_hat.len() {
                val -= b_hat[i];
            }
            if i >= l && !val.is_zero() {
                let pos = i - l;
                if t_hat.len() <= pos {
                    t_hat.resize(pos + 1, zero);
                }
                t_hat[pos] += val;
            }
        }
    }

    // term2: beta * (p_psi_combined - v_gamma*X^(m-1) - u_hat) / X^m → shift 0
    {
        let len2 = p_psi_combined.len().max(m);
        for i in 0..len2 {
            let mut val = if i < p_psi_combined.len() {
                p_psi_combined[i]
            } else {
                zero
            };
            if i == m.saturating_sub(1) {
                val -= v_gamma;
            }
            if i < u_hat.len() {
                val -= u_hat[i];
            }
            if i >= m && !val.is_zero() {
                let pos = i - m;
                if t_hat.len() <= pos {
                    t_hat.resize(pos + 1, zero);
                }
                t_hat[pos] += val * beta;
            }
        }
    }

    // term3: beta^2 * (f_hat - p_hat) / (X^m - gamma) → (fold via gamma) then / X^m
    {
        let max3 = f_hat.len().max(p_hat.len());
        let mut t3 = vec![zero; max3];
        for (i, &c) in f_hat.iter().enumerate() {
            t3[i] += c * beta2;
        }
        for (i, &c) in p_hat.iter().enumerate() {
            t3[i] -= c * beta2;
        }
        for i in (m..t3.len()).rev() {
            let v = t3[i];
            t3[i - m] += v * gamma;
        }
        for i in m..t3.len() {
            if !t3[i].is_zero() {
                let pos = i - m;
                if t_hat.len() <= pos {
                    t_hat.resize(pos + 1, zero);
                }
                t_hat[pos] += t3[i];
            }
        }
    }

    // term4: beta^3 * f_hat → shift 0
    add_scaled_shifted(&mut t_hat, f_hat, beta3, 0);

    // term5: beta^4 * p_hat → shift n-m
    add_scaled_shifted(&mut t_hat, p_hat, beta4, n - m);

    // term6: beta^5 * u_hat → shift n-m+1
    add_scaled_shifted(&mut t_hat, u_hat, beta5, n - m + 1);

    // term7: beta^6 * b_hat → shift n-l+1
    add_scaled_shifted(&mut t_hat, b_hat, beta6, n - l + 1);

    // Trim trailing zeros
    while t_hat.last().map_or(false, |c| c.is_zero()) {
        t_hat.pop();
    }
    t_hat
}

// ═══════════════════════════════════════════════════════════════════
// compute_q_hat — 8-term combination (designed so q_hat(delta) = 0)
// ═══════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
fn compute_q_hat<F: Field>(
    t_hat: &[F],
    v_hat: &[F],
    psi_zy_delta: F,
    phi_delta: F,
    psi_zx_delta: F,
    b_hat: &[F],
    u_hat: &[F],
    f_hat: &[F],
    p_hat: &[F],
    alpha: F,
    beta: F,
    gamma: F,
    delta: F,
    v: F,
    v_gamma: F,
    l: usize,
    m: usize,
) -> Result<Vec<F>, PCSError> {
    let zero = F::zero();
    let n = l * m;
    let beta2 = beta * beta;
    let beta3 = beta2 * beta;
    let beta4 = beta3 * beta;
    let beta5 = beta4 * beta;
    let beta6 = beta5 * beta;

    let delta_l_inv = delta
        .pow([l as u64])
        .inverse()
        .ok_or_else(|| PCSError::InvalidParameters("delta^l is zero".to_string()))?;
    let delta_m_inv = delta
        .pow([m as u64])
        .inverse()
        .ok_or_else(|| PCSError::InvalidParameters("delta^m is zero".to_string()))?;
    let delta_m_minus_gamma_inv = (delta.pow([m as u64]) - gamma)
        .inverse()
        .ok_or_else(|| PCSError::InvalidParameters("delta^m == gamma".to_string()))?;

    // q_hat = t_hat (term1) minus the sub-terms
    let mut q_hat = t_hat.to_vec();

    // term2: delta_l_inv * (v_hat*(psi_zy+alpha*phi) - b_hat - const*X^0)
    let psi_zy_alpha_phi = psi_zy_delta + alpha * phi_delta;
    let mut t2 = poly_scalar_mul(v_hat, psi_zy_alpha_phi);
    poly_sub_in_place(&mut t2, b_hat);
    let const_val2 = delta.pow([(l - 1) as u64]) * (v + alpha * v_gamma);
    if t2.is_empty() {
        t2 = vec![-const_val2];
    } else {
        t2[0] -= const_val2;
    }
    let t2 = poly_scalar_mul(&t2, delta_l_inv);
    for (i, &c) in t2.iter().enumerate() {
        if !c.is_zero() {
            if q_hat.len() <= i {
                q_hat.resize(i + 1, zero);
            }
            q_hat[i] -= c;
        }
    }

    // term3: delta_m_inv * beta * (p_hat*psi_zx - u_hat - const*X^0)
    let mut t3 = poly_scalar_mul(p_hat, psi_zx_delta);
    poly_sub_in_place(&mut t3, u_hat);
    let const_val3 = v_gamma * delta.pow([(m - 1) as u64]);
    if t3.is_empty() {
        t3 = vec![-const_val3];
    } else {
        t3[0] -= const_val3;
    }
    let t3 = poly_scalar_mul(&t3, delta_m_inv * beta);
    for (i, &c) in t3.iter().enumerate() {
        if !c.is_zero() {
            if q_hat.len() <= i {
                q_hat.resize(i + 1, zero);
            }
            q_hat[i] -= c;
        }
    }

    // term4: beta^2 / (delta^m - gamma) * (f_hat - p_hat)
    let max4 = f_hat.len().max(p_hat.len());
    let mut t4 = vec![zero; max4];
    for (i, &c) in f_hat.iter().enumerate() {
        t4[i] += c;
    }
    for (i, &c) in p_hat.iter().enumerate() {
        t4[i] -= c;
    }
    let t4 = poly_scalar_mul(&t4, beta2 * delta_m_minus_gamma_inv);
    for (i, &c) in t4.iter().enumerate() {
        if !c.is_zero() {
            if q_hat.len() <= i {
                q_hat.resize(i + 1, zero);
            }
            q_hat[i] -= c;
        }
    }

    // term5-8: direct in-place scaled subtraction via add_scaled_shifted with
    // negative sign t5: -beta^3 * f_hat → shift 0
    for (i, &c) in f_hat.iter().enumerate() {
        if !c.is_zero() {
            if q_hat.len() <= i {
                q_hat.resize(i + 1, zero);
            }
            q_hat[i] -= c * beta3;
        }
    }
    // t6: -beta^4 * delta^(n-m) * p_hat → shift 0
    let s6 = beta4 * delta.pow([(n - m) as u64]);
    for (i, &c) in p_hat.iter().enumerate() {
        if !c.is_zero() {
            if q_hat.len() <= i {
                q_hat.resize(i + 1, zero);
            }
            q_hat[i] -= c * s6;
        }
    }
    // t7: -beta^5 * delta^(n-m+1) * u_hat → shift 0
    let s7 = beta5 * delta.pow([(n - m + 1) as u64]);
    for (i, &c) in u_hat.iter().enumerate() {
        if !c.is_zero() {
            if q_hat.len() <= i {
                q_hat.resize(i + 1, zero);
            }
            q_hat[i] -= c * s7;
        }
    }
    // t8: -beta^6 * delta^(n-l+1) * b_hat → shift 0
    let s8 = beta6 * delta.pow([(n - l + 1) as u64]);
    for (i, &c) in b_hat.iter().enumerate() {
        if !c.is_zero() {
            if q_hat.len() <= i {
                q_hat.resize(i + 1, zero);
            }
            q_hat[i] -= c * s8;
        }
    }

    while q_hat.last().map_or(false, |c| c.is_zero()) {
        q_hat.pop();
    }
    Ok(q_hat)
}

// ═══════════════════════════════════════════════════════════════════
// compute_q_hat_commit — verifier homomorphic computation
// ═══════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
fn compute_q_hat_commit<E: Pairing>(
    t_hat_commit: &E::G1Affine,
    v_hat_commit: &E::G1Affine,
    b_hat_commit: &E::G1Affine,
    p_hat_commit: &E::G1Affine,
    u_hat_commit: &E::G1Affine,
    mlp_comm: &E::G1Affine,
    g: &E::G1Affine,
    psi_zy_delta: E::ScalarField,
    phi_delta: E::ScalarField,
    psi_zx_delta: E::ScalarField,
    alpha: E::ScalarField,
    beta: E::ScalarField,
    gamma: E::ScalarField,
    delta: E::ScalarField,
    v: E::ScalarField,
    v_gamma: E::ScalarField,
    l: usize,
    m: usize,
) -> Result<E::G1Affine, PCSError> {
    let n = l * m;
    let one = E::ScalarField::one();
    let delta_inv = delta
        .inverse()
        .ok_or_else(|| PCSError::InvalidProof("delta is zero".to_string()))?;
    let delta_l_inv = delta
        .pow([l as u64])
        .inverse()
        .ok_or_else(|| PCSError::InvalidProof("delta^l is zero".to_string()))?;
    let delta_m_inv = delta
        .pow([m as u64])
        .inverse()
        .ok_or_else(|| PCSError::InvalidProof("delta^m is zero".to_string()))?;
    let delta_m_minus_gamma_inv = (delta.pow([m as u64]) - gamma)
        .inverse()
        .ok_or_else(|| PCSError::InvalidProof("delta^m == gamma".to_string()))?;

    // 13 bases: [t, v, g, b, p, g, u, mlp, p, mlp, p, u, b]
    let bases = vec![
        *t_hat_commit,
        *v_hat_commit,
        *g,
        *b_hat_commit,
        *p_hat_commit,
        *g,
        *u_hat_commit,
        *mlp_comm,
        *p_hat_commit,
        *mlp_comm,
        *p_hat_commit,
        *u_hat_commit,
        *b_hat_commit,
    ];

    let psi_zy_alpha_phi = psi_zy_delta + alpha * phi_delta;
    let beta2 = beta * beta;
    let beta3 = beta2 * beta;
    let beta4 = beta3 * beta;
    let beta5 = beta4 * beta;
    let beta6 = beta5 * beta;

    let mut scalars = vec![one; 13];
    scalars[1] = -psi_zy_alpha_phi * delta_l_inv;
    scalars[2] = (v + alpha * v_gamma) * delta_inv;
    scalars[3] = delta_l_inv;
    scalars[4] = -(psi_zx_delta * beta) * delta_m_inv;
    scalars[5] = (beta * v_gamma) * delta_inv;
    scalars[6] = beta * delta_m_inv;
    scalars[7] = -(beta2 * delta_m_minus_gamma_inv);
    scalars[8] = -scalars[7];
    scalars[9] = -beta3;
    scalars[10] = -(beta4 * delta.pow([(n - m) as u64]));
    scalars[11] = -(beta5 * delta.pow([(n - m + 1) as u64]));
    scalars[12] = -(beta6 * delta.pow([(n - l + 1) as u64]));

    Ok(E::G1::msm_unchecked(&bases, &scalars).into_affine())
}

// ═══════════════════════════════════════════════════════════════════
// Transcript-aware single opening
// ═══════════════════════════════════════════════════════════════════

fn samaritan_open_with_transcript<E: Pairing>(
    pp: &SamaritanProverParam<E>,
    poly: &Arc<DenseMultilinearExtension<E::ScalarField>>,
    point: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<(SamaritanProof<E>, E::ScalarField), PCSError> {
    let mu = poly.num_vars();
    let (n, kappa, nu, m, l) =
        checked_split_params(mu, "open").map_err(|e| PCSError::InvalidParameters(e.to_string()))?;

    if point.len() != mu {
        return Err(PCSError::InvalidParameters(
            "point length mismatch".to_string(),
        ));
    }

    let _t_total = ScopedTimer::new(BACKEND, mu, n, "samaritan_open_total", 1, "total");

    let coeffs = poly.to_evaluations();
    let eval = poly
        .evaluate(point)
        .ok_or_else(|| PCSError::InvalidParameters("evaluation failed".to_string()))?;

    let commitment = pp.try_commit(&coeffs)?;
    transcript.append_serializable_element(b"commitment", &commitment)?;
    transcript.append_serializable_element(b"point", &point.to_vec())?;
    transcript.append_field_element(b"eval", &eval)?;

    let f_hat = coeffs;

    let _t_get_eval =
        ScopedTimer::new(BACKEND, mu, n, "samaritan_open_get_eval_set", l, "eval-set");
    let g_eval_values = get_evaluation_set(&f_hat, point, kappa);
    drop(_t_get_eval);

    let _t_build_v = ScopedTimer::new(BACKEND, mu, n, "samaritan_open_build_v_hat", l, "build-v");
    let v_hat = g_eval_values;
    drop(_t_build_v);

    let _t_cm_v = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "samaritan_open_commit_v_hat",
        v_hat.len(),
        "KZG-commit-v",
    );
    let v_hat_commit = pp.try_commit(&v_hat)?;
    transcript.append_serializable_element(b"v_hat_commit", &v_hat_commit)?;
    drop(_t_cm_v);

    let gamma = transcript.get_and_append_challenge_vectors(b"gamma", 1)?[0];

    let v_gamma = poly_eval(&v_hat, gamma);
    transcript.append_field_element(b"v_gamma", &v_gamma)?;

    let _t_build_p = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "samaritan_open_build_p_hat",
        l,
        "chunks-stream",
    );
    let zero = E::ScalarField::zero();
    let one = E::ScalarField::one();
    let mut p_hat = vec![zero; m];
    let mut gamma_pow = one;
    for i in 0..l {
        let chunk = &f_hat[i * m..(i + 1) * m];
        for (j, &val) in chunk.iter().enumerate() {
            p_hat[j] += gamma_pow * val;
        }
        gamma_pow *= gamma;
    }
    drop(_t_build_p);

    let _t_cm_p = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "samaritan_open_commit_p_hat",
        p_hat.len(),
        "KZG-commit-p",
    );
    let p_hat_commit = pp.try_commit(&p_hat)?;
    transcript.append_serializable_element(b"p_hat_commit", &p_hat_commit)?;
    drop(_t_cm_p);

    let alpha = transcript.get_and_append_challenge_vectors(b"alpha", 1)?[0];

    let _t_v_psi_phi = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "samaritan_open_v_times_psi_phi",
        1,
        "v-psi-phi",
    );
    let v_psi_phi_combined =
        structured_mul_v_psi_plus_alpha_phi(&v_hat, &point[kappa..], nu, gamma, alpha);
    drop(_t_v_psi_phi);

    let b_hat: Vec<E::ScalarField> = v_psi_phi_combined
        .iter()
        .take(l.saturating_sub(1))
        .copied()
        .collect();

    let _t_cm_b = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "samaritan_open_commit_b_hat",
        b_hat.len(),
        "KZG-commit-b",
    );
    let b_hat_commit = pp.try_commit(&b_hat)?;
    transcript.append_serializable_element(b"b_hat_commit", &b_hat_commit)?;
    drop(_t_cm_b);

    let _t_p_psi = ScopedTimer::new(BACKEND, mu, n, "samaritan_open_p_times_psi", 1, "p-psi");
    let p_psi_combined = structured_mul_p_psi(&p_hat, &point[..kappa], kappa);
    drop(_t_p_psi);

    let u_hat: Vec<E::ScalarField> = p_psi_combined
        .iter()
        .take(m.saturating_sub(1))
        .copied()
        .collect();

    let _t_cm_u = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "samaritan_open_commit_u_hat",
        u_hat.len(),
        "KZG-commit-u",
    );
    let u_hat_commit = pp.try_commit(&u_hat)?;
    transcript.append_serializable_element(b"u_hat_commit", &u_hat_commit)?;
    drop(_t_cm_u);

    let beta = transcript.get_and_append_challenge_vectors(b"beta", 1)?[0];

    let _t_cthat = ScopedTimer::new(BACKEND, mu, n, "samaritan_open_compute_t_hat", 1, "t-hat");
    let t_hat = compute_t_hat(
        &v_psi_phi_combined,
        &b_hat,
        eval,
        v_gamma,
        alpha,
        gamma,
        &p_psi_combined,
        &u_hat,
        &f_hat,
        &p_hat,
        beta,
        l,
        m,
    );
    drop(_t_cthat);

    let _t_cm_t = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "samaritan_open_commit_t_hat",
        t_hat.len(),
        "KZG-commit-t",
    );
    let t_hat_commit = pp.try_commit(&t_hat)?;
    transcript.append_serializable_element(b"t_hat_commit", &t_hat_commit)?;
    drop(_t_cm_t);

    let _t_cm_s = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "samaritan_open_commit_s_hat",
        t_hat.len() + 1,
        "KZG-commit-s",
    );
    let max_deg = n;
    let shift = max_deg - n + 1;
    let s_hat: Vec<E::ScalarField> = {
        let mut v = vec![E::ScalarField::zero(); shift];
        v.extend_from_slice(&t_hat);
        v
    };
    let s_hat_commit = pp.try_commit(&s_hat)?;
    transcript.append_serializable_element(b"s_hat_commit", &s_hat_commit)?;
    drop(_t_cm_s);

    let delta = transcript.get_and_append_challenge_vectors(b"delta", 1)?[0];

    let _t_cqhat = ScopedTimer::new(BACKEND, mu, n, "samaritan_open_compute_q_hat", 1, "q-hat");
    let psi_zy_delta = evaluate_psi_hat_at(&point[kappa..], delta);
    let phi_delta = evaluate_phi_hat_at(gamma, delta, nu);
    let psi_zx_delta = evaluate_psi_hat_at(&point[..kappa], delta);

    let q_hat = compute_q_hat(
        &t_hat,
        &v_hat,
        psi_zy_delta,
        phi_delta,
        psi_zx_delta,
        &b_hat,
        &u_hat,
        &f_hat,
        &p_hat,
        alpha,
        beta,
        gamma,
        delta,
        eval,
        v_gamma,
        l,
        m,
    )?;
    drop(_t_cqhat);

    let _t_proof = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "samaritan_open_commit_q_proof",
        q_hat.len(),
        "KZG-commit-q-proof",
    );
    let q_quotient = kzg_prove_quotient(&q_hat, delta);
    let q_eval_proof = pp.try_commit(&q_quotient)?;
    drop(_t_proof);

    let proof = SamaritanProof {
        v_hat_commit,
        v_gamma,
        p_hat_commit,
        b_hat_commit,
        u_hat_commit,
        t_hat_commit,
        s_hat_commit,
        q_eval_proof,
        mu,
    };
    drop(_t_total);
    Ok((proof, eval))
}

// ═══════════════════════════════════════════════════════════════════
// Transcript-aware single verification
// ═══════════════════════════════════════════════════════════════════

fn samaritan_verify_with_transcript<E: Pairing>(
    vp: &SamaritanVerifierParam<E>,
    commitment: &Commitment<E>,
    point: &[E::ScalarField],
    value: &E::ScalarField,
    proof: &SamaritanProof<E>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError> {
    let mu = proof.mu;

    // --- vk bound checks (before any computation) ---
    if mu > vp.max_num_vars {
        return Err(PCSError::InvalidProof(format!(
            "verify: proof.mu {} exceeds vp.max_num_vars {}",
            mu, vp.max_num_vars
        )));
    }

    let (n, kappa, nu, m, l) = checked_split_params(mu, "verify")?;

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

    let _t_total = ScopedTimer::new(BACKEND, mu, n, "samaritan_verify_total", 1, "total");

    // Replay transcript absorption
    transcript.append_serializable_element(b"commitment", &commitment.0)?;
    transcript.append_serializable_element(b"point", &point.to_vec())?;
    transcript.append_field_element(b"eval", value)?;

    transcript.append_serializable_element(b"v_hat_commit", &proof.v_hat_commit)?;
    let gamma = transcript.get_and_append_challenge_vectors(b"gamma", 1)?[0];
    transcript.append_field_element(b"v_gamma", &proof.v_gamma)?;
    transcript.append_serializable_element(b"p_hat_commit", &proof.p_hat_commit)?;
    let alpha = transcript.get_and_append_challenge_vectors(b"alpha", 1)?[0];
    transcript.append_serializable_element(b"b_hat_commit", &proof.b_hat_commit)?;
    transcript.append_serializable_element(b"u_hat_commit", &proof.u_hat_commit)?;
    let beta = transcript.get_and_append_challenge_vectors(b"beta", 1)?[0];
    transcript.append_serializable_element(b"t_hat_commit", &proof.t_hat_commit)?;
    transcript.append_serializable_element(b"s_hat_commit", &proof.s_hat_commit)?;
    let delta = transcript.get_and_append_challenge_vectors(b"delta", 1)?[0];

    // Evaluate psi/phi at delta
    let psi_zy_delta = evaluate_psi_hat_at(&point[kappa..], delta);
    let phi_delta = evaluate_phi_hat_at(gamma, delta, nu);
    let psi_zx_delta = evaluate_psi_hat_at(&point[..kappa], delta);

    // Compute q_hat_commit homomorphically
    let _t_qc = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "samaritan_verify_q_hat_commit",
        13,
        "MSM-q-commit",
    );
    let q_hat_commit = compute_q_hat_commit::<E>(
        &proof.t_hat_commit,
        &proof.v_hat_commit,
        &proof.b_hat_commit,
        &proof.p_hat_commit,
        &proof.u_hat_commit,
        &commitment.0,
        &vp.g,
        psi_zy_delta,
        phi_delta,
        psi_zx_delta,
        alpha,
        beta,
        gamma,
        delta,
        *value,
        proof.v_gamma,
        l,
        m,
    )?;
    drop(_t_qc);

    // KZG pairing check: q_hat(delta) == 0
    // e(q_hat_commit, h) == e(q_proof, h*τ - delta*h)
    let _t_kzg = ScopedTimer::new(BACKEND, mu, n, "samaritan_verify_kzg_pairing", 1, "pairing");
    let sx = (vp.h_x.into_group() - vp.h.into_group() * delta).into_affine();
    let neg_q_proof = (-proof.q_eval_proof.into_group()).into_affine();
    let kzg_ok = E::multi_pairing([q_hat_commit, neg_q_proof], [vp.h, sx])
        == PairingOutput(E::TargetField::one());
    drop(_t_kzg);

    if !kzg_ok {
        return Ok(false);
    }

    // Shift check: e(t_hat, h*τ) == e(s_hat, h)
    let _t_shift = ScopedTimer::new(
        BACKEND,
        mu,
        n,
        "samaritan_verify_shift_pairing",
        1,
        "pairing",
    );
    let neg_s_hat = (-proof.s_hat_commit.into_group()).into_affine();
    let shift_ok = E::multi_pairing([proof.t_hat_commit, neg_s_hat], [vp.h_x, vp.h])
        == PairingOutput(E::TargetField::one());
    drop(_t_shift);
    drop(_t_total);

    Ok(shift_ok)
}

// ═══════════════════════════════════════════════════════════════════
// Sumcheck batching — multi-open
// ═══════════════════════════════════════════════════════════════════

fn samaritan_sumcheck_multi_open<E: Pairing>(
    pp: &SamaritanProverParam<E>,
    polynomials: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
    points: &[Vec<E::ScalarField>],
    evals: &[E::ScalarField],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<BatchProof<E, SamaritanPCS<E>>, PCSError> {
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

    // Bind mu for g' opening
    let mut open_t = IOPTranscript::new(b"samaritan-gprime-open");
    open_t.append_field_element(b"mu", &E::ScalarField::from(num_var as u64))?;
    let (g_prime_proof, _g_prime_eval) =
        samaritan_open_with_transcript(pp, &g_prime, a2, &mut open_t)?;

    Ok(BatchProof {
        sum_check_proof: sumcheck_proof,
        f_i_eval_at_point_i: evals.to_vec(),
        g_prime_proof,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Sumcheck batching — batch verify
// ═══════════════════════════════════════════════════════════════════

fn samaritan_sumcheck_batch_verify<E: Pairing>(
    vp: &SamaritanVerifierParam<E>,
    f_i_commitments: &[Commitment<E>],
    points: &[Vec<E::ScalarField>],
    proof: &BatchProof<E, SamaritanPCS<E>>,
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

    // --- guard untrusted num_var ---
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

    let mut verify_t = IOPTranscript::new(b"samaritan-gprime-open");
    verify_t.append_field_element(b"mu", &E::ScalarField::from(num_var as u64))?;
    samaritan_verify_with_transcript(
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

    fn setup(nv: usize) -> (SamaritanProverParam<E>, SamaritanVerifierParam<E>) {
        let mut rng = test_rng();
        SamaritanPCS::<E>::trim(
            &SamaritanPCS::<E>::gen_srs_for_testing(&mut rng, nv).unwrap(),
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
    fn test_samaritan_single_open_verify() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [2usize, 4, 6] {
            let (ck, vk) = setup(nv);
            let p = rpoly(nv, &mut rng);
            let pt = rpt(nv, &mut rng);
            let com = SamaritanPCS::<E>::commit(&ck, &p)?;
            let (proof, val) = SamaritanPCS::<E>::open(&ck, &p, &pt)?;
            assert!(
                SamaritanPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?,
                "Samaritan open/verify nv={nv}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_samaritan_reject_wrong_eval() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [2usize, 4] {
            let (ck, vk) = setup(nv);
            let p = rpoly(nv, &mut rng);
            let pt = rpt(nv, &mut rng);
            let com = SamaritanPCS::<E>::commit(&ck, &p)?;
            let (proof, _val) = SamaritanPCS::<E>::open(&ck, &p, &pt)?;
            let fv = Fr::rand(&mut rng);
            assert!(
                !SamaritanPCS::<E>::verify(&vk, &com, &pt, &fv, &proof)?,
                "wrong eval should reject nv={nv}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_samaritan_reject_wrong_point() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = SamaritanPCS::<E>::commit(&ck, &p)?;
        let (proof, val) = SamaritanPCS::<E>::open(&ck, &p, &pt)?;
        assert!(!SamaritanPCS::<E>::verify(
            &vk,
            &com,
            &rpt(nv, &mut rng),
            &val,
            &proof
        )?);
        Ok(())
    }

    #[test]
    fn test_samaritan_reject_wrong_commitment() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p1 = rpoly(nv, &mut rng);
        let p2 = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com2 = SamaritanPCS::<E>::commit(&ck, &p2)?;
        let (proof, val) = SamaritanPCS::<E>::open(&ck, &p1, &pt)?;
        assert!(!SamaritanPCS::<E>::verify(&vk, &com2, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_samaritan_reject_tampered_v_hat_commit() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = SamaritanPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = SamaritanPCS::<E>::open(&ck, &p, &pt)?;
        proof.v_hat_commit = (proof.v_hat_commit.into_group() * Fr::from(2u64)).into_affine();
        assert!(!SamaritanPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_samaritan_reject_tampered_p_hat_commit() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = SamaritanPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = SamaritanPCS::<E>::open(&ck, &p, &pt)?;
        proof.p_hat_commit = (proof.p_hat_commit.into_group() * Fr::from(2u64)).into_affine();
        assert!(!SamaritanPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_samaritan_reject_tampered_b_hat_commit() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = SamaritanPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = SamaritanPCS::<E>::open(&ck, &p, &pt)?;
        proof.b_hat_commit = (proof.b_hat_commit.into_group() * Fr::from(2u64)).into_affine();
        assert!(!SamaritanPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_samaritan_reject_tampered_u_hat_commit() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = SamaritanPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = SamaritanPCS::<E>::open(&ck, &p, &pt)?;
        proof.u_hat_commit = (proof.u_hat_commit.into_group() * Fr::from(2u64)).into_affine();
        assert!(!SamaritanPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_samaritan_reject_tampered_t_hat_commit() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = SamaritanPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = SamaritanPCS::<E>::open(&ck, &p, &pt)?;
        proof.t_hat_commit = (proof.t_hat_commit.into_group() * Fr::from(2u64)).into_affine();
        assert!(!SamaritanPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_samaritan_reject_tampered_s_hat_commit() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = SamaritanPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = SamaritanPCS::<E>::open(&ck, &p, &pt)?;
        proof.s_hat_commit = (proof.s_hat_commit.into_group() * Fr::from(2u64)).into_affine();
        assert!(!SamaritanPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_samaritan_reject_tampered_q_proof() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = SamaritanPCS::<E>::commit(&ck, &p)?;
        let (mut proof, val) = SamaritanPCS::<E>::open(&ck, &p, &pt)?;
        proof.q_eval_proof = (proof.q_eval_proof.into_group() * Fr::from(3u64)).into_affine();
        assert!(!SamaritanPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        Ok(())
    }

    #[test]
    fn test_samaritan_verify_rejects_wrong_point_len() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 4;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = SamaritanPCS::<E>::commit(&ck, &p)?;
        let (proof, val) = SamaritanPCS::<E>::open(&ck, &p, &pt)?;
        let short_pt = rpt(2, &mut rng);
        let r = SamaritanPCS::<E>::verify(&vk, &com, &short_pt, &val, &proof);
        assert!(r.is_err(), "short point should return Error");
        Ok(())
    }

    #[test]
    fn test_samaritan_verify_rejects_huge_mu_without_panic() {
        let mut rng = test_rng();
        let nv = 2;
        let (ck, vk) = setup(nv);
        let p = rpoly(nv, &mut rng);
        let pt = rpt(nv, &mut rng);
        let com = SamaritanPCS::<E>::commit(&ck, &p).unwrap();
        let (mut proof, val) = SamaritanPCS::<E>::open(&ck, &p, &pt).unwrap();
        proof.mu = usize::BITS as usize;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            SamaritanPCS::<E>::verify(&vk, &com, &pt, &val, &proof)
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
    fn test_samaritan_multi_open_k1() -> Result<(), PCSError> {
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
            .map(|p| SamaritanPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = SamaritanPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(SamaritanPCS::<E>::batch_verify(
            &vk, &comms, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_samaritan_multi_open_distinct() -> Result<(), PCSError> {
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
            .map(|p| SamaritanPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = SamaritanPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(SamaritanPCS::<E>::batch_verify(
            &vk, &comms, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_samaritan_multi_open_repeated() -> Result<(), PCSError> {
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
            .map(|p| SamaritanPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = SamaritanPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(SamaritanPCS::<E>::batch_verify(
            &vk, &comms, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_samaritan_batch_reject_wrong_eval() -> Result<(), PCSError> {
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
            .map(|p| SamaritanPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        evals[0] += Fr::one();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = SamaritanPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(!SamaritanPCS::<E>::batch_verify(
            &vk, &comms, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_samaritan_batch_reject_wrong_point() -> Result<(), PCSError> {
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
            .map(|p| SamaritanPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = SamaritanPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let mut wp = pts.clone();
        wp[0] = rpt(nv, &mut rng);
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert_rejects(SamaritanPCS::<E>::batch_verify(
            &vk, &comms, &wp, &proof, &mut tv,
        ));
        Ok(())
    }

    #[test]
    fn test_samaritan_batch_reject_wrong_commitment() -> Result<(), PCSError> {
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
            .map(|p| SamaritanPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = SamaritanPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let extra = SamaritanPCS::<E>::commit(&ck, &rpoly(nv, &mut rng))?;
        let mut wc = comms.clone();
        wc[0] = extra;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        assert!(!SamaritanPCS::<E>::batch_verify(
            &vk, &wc, &pts, &proof, &mut tv
        )?);
        Ok(())
    }

    #[test]
    fn test_samaritan_batch_reject_malformed_lengths() -> Result<(), PCSError> {
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
            .map(|p| SamaritanPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let mut proof = SamaritanPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        let r = SamaritanPCS::<E>::batch_verify(
            &vk,
            &comms[..2],
            &pts,
            &proof,
            &mut IOPTranscript::new(b"t"),
        );
        assert!(r.is_err() || !r.unwrap());
        proof.f_i_eval_at_point_i.pop();
        let r2 = SamaritanPCS::<E>::batch_verify(
            &vk,
            &comms,
            &pts,
            &proof,
            &mut IOPTranscript::new(b"t"),
        );
        assert!(r2.is_err() || !r2.unwrap());
        Ok(())
    }

    // ── vk bound checks ──

    #[test]
    fn test_samaritan_verify_rejects_mu_above_vk_bound() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let big_nv = 6;
        let small_nv = 4;
        let srs = SamaritanPCS::<E>::gen_srs_for_testing(&mut rng, big_nv)?;
        // Trim to small_nv to get restricted vk, large ck for proof
        let (big_ck, _) = SamaritanPCS::<E>::trim(&srs, None, Some(big_nv))?;
        let (_, small_vk) = SamaritanPCS::<E>::trim(&srs, None, Some(small_nv))?;
        let p = rpoly(big_nv, &mut rng);
        let pt = rpt(big_nv, &mut rng);
        let com = SamaritanPCS::<E>::commit(&big_ck, &p)?;
        let (proof, val) = SamaritanPCS::<E>::open(&big_ck, &p, &pt)?;
        let r = SamaritanPCS::<E>::verify(&small_vk, &com, &pt, &val, &proof);
        assert!(r.is_err(), "mu above vk bound should return Error");
        Ok(())
    }

    #[test]
    fn test_samaritan_verify_rejects_degree_above_vk_bound() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let big_nv = 6;
        let small_nv = 4;
        let srs = SamaritanPCS::<E>::gen_srs_for_testing(&mut rng, big_nv)?;
        let (big_ck, _) = SamaritanPCS::<E>::trim(&srs, None, Some(big_nv))?;
        // Trim SRS to 2^small_nv which gives small max_degree
        let (_, small_vk) = srs.trim(1 << small_nv)?;
        let p = rpoly(big_nv, &mut rng);
        let pt = rpt(big_nv, &mut rng);
        let com = SamaritanPCS::<E>::commit(&big_ck, &p)?;
        let (proof, val) = SamaritanPCS::<E>::open(&big_ck, &p, &pt)?;
        let r = SamaritanPCS::<E>::verify(&small_vk, &com, &pt, &val, &proof);
        assert!(r.is_err(), "degree above vk bound should return Error");
        Ok(())
    }

    // ── SRS too small without panic ──

    #[test]
    fn test_samaritan_commit_rejects_srs_too_small_without_panic() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let big_nv = 4;
        let tiny_nv = 2;
        let srs = SamaritanPCS::<E>::gen_srs_for_testing(&mut rng, big_nv)?;
        let (tiny_ck, _) = SamaritanPCS::<E>::trim(&srs, None, Some(tiny_nv))?;
        let big_poly = rpoly(big_nv, &mut rng);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            SamaritanPCS::<E>::commit(&tiny_ck, &big_poly)
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
    fn test_samaritan_open_rejects_srs_too_small_without_panic() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let big_nv = 4;
        let tiny_nv = 2;
        let srs = SamaritanPCS::<E>::gen_srs_for_testing(&mut rng, big_nv)?;
        let (tiny_ck, _) = SamaritanPCS::<E>::trim(&srs, None, Some(tiny_nv))?;
        let big_poly = rpoly(big_nv, &mut rng);
        let pt = rpt(big_nv, &mut rng);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            SamaritanPCS::<E>::open(&tiny_ck, &big_poly, &pt)
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
    fn test_samaritan_batch_verify_rejects_huge_num_var_without_panic() -> Result<(), PCSError> {
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
            .map(|p| SamaritanPCS::<E>::commit(&ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let mut proof = SamaritanPCS::<E>::multi_open(&ck, &polys, &pts, &evals, &mut tp)?;
        proof.sum_check_proof.point = vec![Fr::zero(); usize::BITS as usize];
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            SamaritanPCS::<E>::batch_verify(&vk, &comms, &pts, &proof, &mut tv)
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
    fn test_samaritan_batch_verify_rejects_num_var_above_vk_bound() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let big_nv = 4;
        let small_nv = 2;
        let srs = SamaritanPCS::<E>::gen_srs_for_testing(&mut rng, big_nv)?;
        let (big_ck, _) = SamaritanPCS::<E>::trim(&srs, None, Some(big_nv))?;
        let (_, small_vk) = SamaritanPCS::<E>::trim(&srs, None, Some(small_nv))?;
        let polys: Vec<_> = (0..1).map(|_| rpoly(big_nv, &mut rng)).collect();
        let pts: Vec<_> = polys.iter().map(|_| rpt(big_nv, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(pts.iter())
            .map(|(p, pt)| p.evaluate(pt).unwrap())
            .collect();
        let comms: Vec<_> = polys
            .iter()
            .map(|p| SamaritanPCS::<E>::commit(&big_ck, p).unwrap())
            .collect();
        let mut tp = IOPTranscript::new(b"t");
        tp.append_field_element(b"init", &Fr::zero())?;
        let proof = SamaritanPCS::<E>::multi_open(&big_ck, &polys, &pts, &evals, &mut tp)?;
        let mut tv = IOPTranscript::new(b"t");
        tv.append_field_element(b"init", &Fr::zero())?;
        let r = SamaritanPCS::<E>::batch_verify(&small_vk, &comms, &pts, &proof, &mut tv);
        assert!(r.is_err(), "num_var above vk bound should return Error");
        Ok(())
    }

    // ── Structured vs dense equivalence ──

    #[test]
    fn test_structured_vs_dense_psi_phi() {
        let mut rng = test_rng();
        for nv in [2, 4, 6] {
            let mu = nv;
            let kappa = compute_kappa(mu);
            let nu = mu - kappa;
            let l = 1 << nu;
            let point: Vec<Fr> = (0..mu).map(|_| Fr::rand(&mut rng)).collect();
            let v_hat: Vec<Fr> = (0..l).map(|_| Fr::rand(&mut rng)).collect();
            let gamma = Fr::rand(&mut rng);
            let alpha = Fr::rand(&mut rng);

            // Old: dense poly_mul
            let old_psi = {
                let mut acc = vec![Fr::one()];
                for i in 0..nu {
                    let s = 1 << i;
                    let z = point[kappa + i];
                    let new_len = acc.len() + s;
                    let mut next = vec![Fr::zero(); new_len];
                    for (j, &c) in acc.iter().enumerate() {
                        next[j] += z * c;
                        next[j + s] += (Fr::one() - z) * c;
                    }
                    acc = next;
                }
                acc
            };
            let old_phi = {
                let mut acc = vec![Fr::one()];
                let mut gp = gamma;
                for i in 0..nu {
                    let s = 1 << i;
                    let new_len = acc.len() + s;
                    let mut next = vec![Fr::zero(); new_len];
                    for (j, &c) in acc.iter().enumerate() {
                        next[j] += gp * c;
                        next[j + s] += c;
                    }
                    acc = next;
                    gp *= gp;
                }
                acc
            };
            let psi_plus_alpha_phi = {
                let max_len = old_psi.len().max(old_phi.len());
                let mut combined = vec![Fr::zero(); max_len];
                for (i, &c) in old_psi.iter().enumerate() {
                    combined[i] += c;
                }
                for (i, &c) in old_phi.iter().enumerate() {
                    combined[i] += alpha * c;
                }
                combined
            };
            let old_result = poly_mul(&v_hat, &psi_plus_alpha_phi);

            // New: structured
            let new_result =
                structured_mul_v_psi_plus_alpha_phi(&v_hat, &point[kappa..], nu, gamma, alpha);

            assert_eq!(
                old_result.len(),
                new_result.len(),
                "nv={nv}: length mismatch old={} new={}",
                old_result.len(),
                new_result.len()
            );
            for (i, (&o, &n)) in old_result.iter().zip(new_result.iter()).enumerate() {
                assert_eq!(o, n, "nv={nv} index={i}: old={o:?} new={n:?}");
            }
        }
    }

    // ── Structured p_hat * psi_zx equivalence ──

    #[test]
    fn test_structured_vs_dense_p_psi() {
        let mut rng = test_rng();
        for nv in [2, 4, 6] {
            let mu = nv;
            let kappa = compute_kappa(mu);
            let m = 1 << kappa;
            let point: Vec<Fr> = (0..mu).map(|_| Fr::rand(&mut rng)).collect();
            let p_hat: Vec<Fr> = (0..m).map(|_| Fr::rand(&mut rng)).collect();

            // Old: dense poly_mul(p_hat, psi_zx)
            let old_psi_zx = {
                let mut acc = vec![Fr::one()];
                for i in 0..kappa {
                    let s = 1 << i;
                    let z = point[i];
                    let new_len = acc.len() + s;
                    let mut next = vec![Fr::zero(); new_len];
                    for (j, &c) in acc.iter().enumerate() {
                        next[j] += z * c;
                        next[j + s] += (Fr::one() - z) * c;
                    }
                    acc = next;
                }
                acc
            };
            let old_result = poly_mul(&p_hat, &old_psi_zx);

            // New: structured
            let new_result = structured_mul_p_psi(&p_hat, &point[..kappa], kappa);

            assert_eq!(
                old_result.len(),
                new_result.len(),
                "nv={nv}: p_psi length mismatch old={} new={}",
                old_result.len(),
                new_result.len()
            );
            for (i, (&o, &n)) in old_result.iter().zip(new_result.iter()).enumerate() {
                assert_eq!(o, n, "nv={nv} p_psi index={i}: old={o:?} new={n:?}");
            }
        }
    }

    // ── compute_t_hat / compute_q_hat reference helpers (old term-by-term) ──

    #[allow(non_snake_case)]
    fn compute_t_hat_reference<F: Field>(
        v_psi_phi_combined: &[F],
        b_hat: &[F],
        eval: F,
        v_gamma: F,
        alpha: F,
        gamma: F,
        p_psi_combined: &[F],
        u_hat: &[F],
        f_hat: &[F],
        p_hat: &[F],
        beta: F,
        l: usize,
        m: usize,
    ) -> Vec<F> {
        let n = l * m;
        let zero = F::zero();
        let spike1 = eval + alpha * v_gamma;
        let mut t1 = v_psi_phi_combined.to_vec();
        if t1.len() <= l - 1 {
            t1.resize(l, zero);
        }
        t1[l - 1] -= spike1;
        poly_sub_in_place(&mut t1, b_hat);
        let t1: Vec<F> = if t1.len() > l {
            t1[l..].to_vec()
        } else {
            vec![]
        };

        let mut t2 = p_psi_combined.to_vec();
        if t2.len() <= m - 1 {
            t2.resize(m, zero);
        }
        t2[m - 1] -= v_gamma;
        poly_sub_in_place(&mut t2, u_hat);
        for c in &mut t2 {
            *c *= beta;
        }
        let t2: Vec<F> = if t2.len() > m {
            t2[m..].to_vec()
        } else {
            vec![]
        };

        let beta2 = beta * beta;
        let beta3 = beta2 * beta;
        let max3 = f_hat.len().max(p_hat.len());
        let mut t3 = vec![zero; max3];
        for (i, &c) in f_hat.iter().enumerate() {
            t3[i] += c * beta2;
        }
        for (i, &c) in p_hat.iter().enumerate() {
            t3[i] -= c * beta2;
        }
        for i in (m..t3.len()).rev() {
            let v = t3[i];
            t3[i - m] += v * gamma;
        }
        let t3: Vec<F> = if t3.len() > m {
            t3[m..].to_vec()
        } else {
            vec![]
        };

        let t4: Vec<F> = f_hat.iter().map(|c| *c * beta3).collect();
        let beta4 = beta3 * beta;
        let beta5 = beta4 * beta;
        let beta6 = beta5 * beta;
        let s5 = n - m;
        let mut t5 = vec![zero; s5];
        for c in p_hat {
            t5.push(*c * beta4);
        }
        let s6 = n - m + 1;
        let mut t6 = vec![zero; s6];
        for c in u_hat {
            t6.push(*c * beta5);
        }
        let s7 = n - l + 1;
        let mut t7 = vec![zero; s7];
        for c in b_hat {
            t7.push(*c * beta6);
        }

        let all: [&[F]; 7] = [&t1, &t2, &t3, &t4, &t5, &t6, &t7];
        let max_len = all.iter().map(|v| v.len()).max().unwrap_or(0);
        let mut out = vec![zero; max_len];
        for t in &all {
            for (i, &c) in t.iter().enumerate() {
                if !c.is_zero() {
                    out[i] += c;
                }
            }
        }
        while out.last().map_or(false, |c| c.is_zero()) {
            out.pop();
        }
        out
    }

    #[allow(non_snake_case)]
    fn compute_q_hat_reference<F: Field>(
        t_hat: &[F],
        v_hat: &[F],
        psi_zy_delta: F,
        phi_delta: F,
        psi_zx_delta: F,
        b_hat: &[F],
        u_hat: &[F],
        f_hat: &[F],
        p_hat: &[F],
        alpha: F,
        beta: F,
        gamma: F,
        delta: F,
        v: F,
        v_gamma: F,
        l: usize,
        m: usize,
    ) -> Result<Vec<F>, PCSError> {
        let zero = F::zero();
        let n = l * m;
        let dli = delta
            .pow([l as u64])
            .inverse()
            .ok_or_else(|| PCSError::InvalidProof("".to_string()))?;
        let dmi = delta
            .pow([m as u64])
            .inverse()
            .ok_or_else(|| PCSError::InvalidProof("".to_string()))?;
        let dmgi = (delta.pow([m as u64]) - gamma)
            .inverse()
            .ok_or_else(|| PCSError::InvalidProof("".to_string()))?;
        let t1 = t_hat.to_vec();
        let pza = psi_zy_delta + alpha * phi_delta;
        let mut t2 = poly_scalar_mul(v_hat, pza);
        poly_sub_in_place(&mut t2, b_hat);
        let cv2 = delta.pow([(l - 1) as u64]) * (v + alpha * v_gamma);
        if t2.is_empty() {
            t2 = vec![-cv2];
        } else {
            t2[0] -= cv2;
        }
        let t2 = poly_scalar_mul(&t2, dli);
        let mut t3 = poly_scalar_mul(p_hat, psi_zx_delta);
        poly_sub_in_place(&mut t3, u_hat);
        let cv3 = v_gamma * delta.pow([(m - 1) as u64]);
        if t3.is_empty() {
            t3 = vec![-cv3];
        } else {
            t3[0] -= cv3;
        }
        let t3 = poly_scalar_mul(&t3, dmi * beta);
        let beta2 = beta * beta;
        let max4 = f_hat.len().max(p_hat.len());
        let mut t4 = vec![zero; max4];
        for (i, &c) in f_hat.iter().enumerate() {
            t4[i] += c;
        }
        for (i, &c) in p_hat.iter().enumerate() {
            t4[i] -= c;
        }
        let t4 = poly_scalar_mul(&t4, beta2 * dmgi);
        let beta3 = beta2 * beta;
        let beta4 = beta3 * beta;
        let beta5 = beta4 * beta;
        let beta6 = beta5 * beta;
        let t5 = poly_scalar_mul(f_hat, beta3);
        let t6 = poly_scalar_mul(p_hat, beta4 * delta.pow([(n - m) as u64]));
        let t7 = poly_scalar_mul(u_hat, beta5 * delta.pow([(n - m + 1) as u64]));
        let t8 = poly_scalar_mul(b_hat, beta6 * delta.pow([(n - l + 1) as u64]));
        let sub: [&[F]; 7] = [&t2, &t3, &t4, &t5, &t6, &t7, &t8];
        let max_len = t1.len().max(sub.iter().map(|v| v.len()).max().unwrap_or(0));
        let mut out = vec![zero; max_len];
        for (i, &c) in t1.iter().enumerate() {
            out[i] += c;
        }
        for t in &sub {
            for (i, &c) in t.iter().enumerate() {
                if !c.is_zero() {
                    out[i] -= c;
                }
            }
        }
        while out.last().map_or(false, |c| c.is_zero()) {
            out.pop();
        }
        Ok(out)
    }

    // ── Equivalence tests ──

    #[test]
    fn test_compute_t_hat_in_place_matches_reference() {
        let mut rng = test_rng();
        let configs = [(2, 2), (4, 4), (8, 4)];
        for &(_l, _m) in &configs {
            let l = _l;
            let m = _m;
            let n = l * m;
            let vpc = vec![Fr::rand(&mut rng); n];
            let b = vec![Fr::rand(&mut rng); l];
            let ptc = vec![Fr::rand(&mut rng); n];
            let u = vec![Fr::rand(&mut rng); m];
            let fh = vec![Fr::rand(&mut rng); n];
            let ph = vec![Fr::rand(&mut rng); m];
            let eval = Fr::rand(&mut rng);
            let vg = Fr::rand(&mut rng);
            let alpha = Fr::rand(&mut rng);
            let gamma = Fr::rand(&mut rng);
            let beta = Fr::rand(&mut rng);
            let old = compute_t_hat_reference(
                &vpc, &b, eval, vg, alpha, gamma, &ptc, &u, &fh, &ph, beta, l, m,
            );
            let new = compute_t_hat(
                &vpc, &b, eval, vg, alpha, gamma, &ptc, &u, &fh, &ph, beta, l, m,
            );
            assert_eq!(
                old.len(),
                new.len(),
                "(l,m)=({l},{m}): t_hat length mismatch"
            );
            for (i, (&o, &n)) in old.iter().zip(new.iter()).enumerate() {
                assert_eq!(o, n, "(l,m)=({l},{m}) idx={i}: old={o:?} new={n:?}");
            }
        }
    }

    #[test]
    fn test_compute_q_hat_in_place_matches_reference() {
        let mut rng = test_rng();
        let configs = [(2, 2), (4, 4), (8, 4)];
        for &(_l, _m) in &configs {
            let l = _l;
            let m = _m;
            let th = vec![Fr::rand(&mut rng); l * m];
            let vh = vec![Fr::rand(&mut rng); l];
            let b = vec![Fr::rand(&mut rng); l];
            let u = vec![Fr::rand(&mut rng); m];
            let fh = vec![Fr::rand(&mut rng); l * m];
            let ph = vec![Fr::rand(&mut rng); m];
            let psi_zy = Fr::rand(&mut rng);
            let phi_d = Fr::rand(&mut rng);
            let psi_zx = Fr::rand(&mut rng);
            let alpha = Fr::rand(&mut rng);
            let beta = Fr::rand(&mut rng);
            let gamma = loop {
                let g = Fr::rand(&mut rng);
                if !g.is_zero() {
                    break g;
                }
            };
            let delta = loop {
                let d = Fr::rand(&mut rng);
                if !d.is_zero() && d.pow([m as u64]) != gamma {
                    break d;
                }
            };
            let v = Fr::rand(&mut rng);
            let vg = Fr::rand(&mut rng);
            let old = compute_q_hat_reference(
                &th, &vh, psi_zy, phi_d, psi_zx, &b, &u, &fh, &ph, alpha, beta, gamma, delta, v,
                vg, l, m,
            )
            .unwrap();
            let new = compute_q_hat(
                &th, &vh, psi_zy, phi_d, psi_zx, &b, &u, &fh, &ph, alpha, beta, gamma, delta, v,
                vg, l, m,
            )
            .unwrap();
            assert_eq!(
                old.len(),
                new.len(),
                "(l,m)=({l},{m}): q_hat length mismatch"
            );
            for (i, (&o, &n)) in old.iter().zip(new.iter()).enumerate() {
                assert_eq!(o, n, "(l,m)=({l},{m}) idx={i}: old={o:?} new={n:?}");
            }
        }
    }

    fn assert_rejects(r: Result<bool, PCSError>) {
        match r {
            Ok(true) => panic!("expected reject"),
            Ok(false) => {},
            Err(_) => {},
        }
    }
}
