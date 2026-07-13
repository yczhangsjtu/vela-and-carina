//! Structured reference string for the CHOPIN multilinear PCS.
//!
//! For `mu` variables (`mu >= 2`) CHOPIN uses the non-padded rectangular split
//!   m_left  = ceil(mu/2),   M_L = 2^m_left,
//!   m_right = floor(mu/2),  M_R = 2^m_right,   N = M_L * M_R = 2^mu.
//!
//! The commitment key is the bivariate Cartesian grid
//!   [tau^i sigma^j]_1,   0 <= i < M_L,  0 <= j < M_R    (exactly N elements),
//! and the verifier key is the three-element G2 material
//!   [1]_2, [tau]_2, [sigma]_2      (plus [1]_1 in G1).
//!
//! There is exactly ONE bivariate KZG key here: the univariate polynomials
//! `f_zR`, `f_alpha`, `S`, `W`, `W'` are all committed on the `sigma^0` slice
//! `[tau^i]_1` of this same key, and the second quotient `q2(Y)` on the
//! `tau^0` slice `[sigma^j]_1`. This is the "matryoshka" property of bivariate
//! KZG (paper §4.2): it is NOT two disjoint univariate KZG keys.
//!
//! # q1-prefix G1 layout
//! The dominant opening MSM commits `q1(X,Y) = (f-f(α,Y))/(X-α)`, whose
//! coefficients `q1_j[i]` have `0 <= i < M_L-1`, `0 <= j < M_R` (length
//! `(M_L-1)*M_R = N-M_R`). To make that a single contiguous prefix MSM, the
//! grid is stored dominant-q1-first (j-major prefix):
//!   base_index(i,j) = j*(M_L-1)+i           for i < M_L-1  (prefix, size
//! N-M_R)                   = (M_L-1)*M_R + j        for i == M_L-1 (tail, the
//! top X power) This is a bijection of the full M_L x M_R grid onto [0, N). The
//! same Vec serves the full commitment (an N-MSM), the q1 prefix MSM, the
//! tau-slice MSMs (`f_zR`, `f_alpha`, `S`, `W`, `W'`) and the sigma-slice MSM
//! (`q2`).
//!
//! WARNING: `gen_srs_for_testing` samples the trapdoors `tau, sigma` locally.
//! It is FOR TESTING ONLY and MUST NOT be used as a production trusted setup.
//! This scheme provides no hiding / zero knowledge.

use crate::pcs::{prelude::PCSError, profile, StructuredReferenceString};
use ark_ec::{
    pairing::Pairing,
    scalar_mul::{fixed_base::FixedBase, variable_base::VariableBaseMSM},
    AffineRepr, CurveGroup,
};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{format, rand::Rng, string::ToString, sync::Arc, vec::Vec, One, UniformRand, Zero};

const BACKEND: &str = "Chopin";

/// Chunk size (number of G1 elements) for the FixedBase SRS generation. The
/// window table is built once and reused; scalars and projective points are
/// materialised one chunk at a time so setup never simultaneously holds `N`
/// scalars, `N` projectives and several `N`-sized affine buffers. The only
/// unavoidable `N`-sized buffer is the output key itself.
pub(crate) const SRS_GEN_CHUNK: usize = 1 << 16;

/// Split `mu` variables into `(m_left, m_right)` with `m_left = ceil(mu/2)`,
/// `m_right = floor(mu/2)`.
#[inline]
pub(crate) fn split_exponents(mu: usize) -> (usize, usize) {
    (mu.div_ceil(2), mu / 2)
}

/// dominant-q1-first `base_index(i,j)` for a layout of dimensions
/// `big_ml x big_mr`. Requires `big_ml >= 2`.
#[inline]
pub(crate) fn grid_base_index(big_ml: usize, big_mr: usize, i: usize, j: usize) -> usize {
    let q1_len = (big_ml - 1) * big_mr;
    if i < big_ml - 1 {
        j * (big_ml - 1) + i
    } else {
        q1_len + j
    }
}

/// Inverse of [`grid_base_index`]: layout position `p` -> `(i, j)`.
#[inline]
pub(crate) fn inv_base_index(big_ml: usize, big_mr: usize, p: usize) -> (usize, usize) {
    let q1_len = (big_ml - 1) * big_mr;
    if p < q1_len {
        (p % (big_ml - 1), p / (big_ml - 1))
    } else {
        (big_ml - 1, p - q1_len)
    }
}

/// Universal parameters: the full bivariate grid key plus the three G2 powers.
#[derive(Clone, Debug)]
pub struct ChopinUniversalParams<E: Pairing> {
    /// `[tau^i sigma^j]_1` in dominant-q1-first layout for the maximal split.
    /// Shared behind an `Arc` so trimming to the same size does not copy.
    pub g1_powers: Arc<Vec<E::G1Affine>>,
    pub g2_one: E::G2Affine,
    pub g2_tau: E::G2Affine,
    pub g2_sigma: E::G2Affine,
    /// Maximal supported number of variables.
    pub max_num_vars: usize,
    pub m_left: usize,
    pub m_right: usize,
}

/// Prover parameters for a specific `num_vars`.
#[derive(Clone, Debug)]
pub struct ChopinProverParam<E: Pairing> {
    /// `[tau^i sigma^j]_1` in dominant-q1-first layout for this split. Shared
    /// with the universal params when trimming to the maximal size.
    pub g1_powers: Arc<Vec<E::G1Affine>>,
    pub num_vars: usize,
    pub m_left: usize,
    pub m_right: usize,
}

/// Verifier parameters: `[1]_1` and the three G2 elements.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct ChopinVerifierParam<E: Pairing> {
    pub g1_one: E::G1Affine,
    pub g2_one: E::G2Affine,
    pub g2_tau: E::G2Affine,
    pub g2_sigma: E::G2Affine,
    pub max_num_vars: usize,
    pub m_left: usize,
    pub m_right: usize,
}

impl<E: Pairing> ChopinProverParam<E> {
    #[inline]
    pub(crate) fn big_ml(&self) -> usize {
        1usize << self.m_left
    }
    #[inline]
    pub(crate) fn big_mr(&self) -> usize {
        1usize << self.m_right
    }
    #[inline]
    pub(crate) fn n(&self) -> usize {
        self.big_ml() * self.big_mr()
    }
    #[inline]
    pub(crate) fn q1_len(&self) -> usize {
        (self.big_ml() - 1) * self.big_mr()
    }
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn base_index(&self, i: usize, j: usize) -> usize {
        grid_base_index(self.big_ml(), self.big_mr(), i, j)
    }

    /// Full commitment `C_F = [f(τ,σ)]_1`. Reorders the canonical evaluation
    /// vector `evals[i + M_L*j]` into the dominant-q1-first layout and performs
    /// a single N-MSM over the whole grid.
    #[allow(dead_code)]
    pub(crate) fn msm_full_reordered(
        &self,
        evals: &[E::ScalarField],
    ) -> Result<E::G1Affine, PCSError> {
        let big_ml = self.big_ml();
        let big_mr = self.big_mr();
        let n = big_ml * big_mr;
        if evals.len() != n {
            return Err(PCSError::InvalidParameters(format!(
                "commit expects {} evaluations, got {}",
                n,
                evals.len()
            )));
        }
        if self.g1_powers.len() < n {
            return Err(PCSError::InvalidParameters(format!(
                "SRS G1 length {} insufficient for N={}",
                self.g1_powers.len(),
                n
            )));
        }
        let mut scalars = ark_std::vec![E::ScalarField::zero(); n];
        for j in 0..big_mr {
            let base = big_ml * j;
            for i in 0..big_ml {
                scalars[grid_base_index(big_ml, big_mr, i, j)] = evals[base + i];
            }
        }
        Ok(E::G1::msm_unchecked(&self.g1_powers[..n], &scalars).into_affine())
    }

    /// Commit `q1` (the dominant quotient) via a single contiguous prefix MSM.
    /// `q1_coeffs` is already in `j*(M_L-1)+i` order, so this is exactly the
    /// prefix `g1[0 .. (M_L-1)*M_R]`. Real scalar length is `N - M_R`.
    pub(crate) fn msm_q1_prefix(
        &self,
        q1_coeffs: &[E::ScalarField],
    ) -> Result<E::G1Affine, PCSError> {
        let q1_len = self.q1_len();
        if q1_coeffs.len() != q1_len {
            return Err(PCSError::InvalidParameters(format!(
                "q1 length {} != (M_L-1)*M_R = {}",
                q1_coeffs.len(),
                q1_len
            )));
        }
        Ok(E::G1::msm_unchecked(&self.g1_powers[..q1_len], q1_coeffs).into_affine())
    }

    /// Commit a univariate polynomial (degree `< M_L`) on the `sigma^0` slice
    /// `[tau^i]_1`. Positions `0..M_L-1` are the contiguous prefix; the single
    /// top coefficient (index `M_L-1`, if present) sits at position
    /// `(M_L-1)*M_R`. So this is a contiguous prefix MSM plus at most one
    /// scalar multiplication — it never allocates a fresh `M_L`-length base
    /// vector.
    pub(crate) fn msm_tau_slice(&self, coeffs: &[E::ScalarField]) -> Result<E::G1Affine, PCSError> {
        let big_ml = self.big_ml();
        if coeffs.len() > big_ml {
            return Err(PCSError::InvalidParameters(format!(
                "tau-slice polynomial length {} exceeds M_L = {}",
                coeffs.len(),
                big_ml
            )));
        }
        if coeffs.is_empty() {
            return Ok(E::G1Affine::default());
        }
        let prefix_len = coeffs.len().min(big_ml - 1);
        let mut acc = E::G1::msm_unchecked(&self.g1_powers[..prefix_len], &coeffs[..prefix_len]);
        if coeffs.len() == big_ml {
            // top X power tau^{M_L-1} lives in the tail region.
            let top = self.g1_powers[self.q1_len()];
            acc += top.into_group() * coeffs[big_ml - 1];
        }
        Ok(acc.into_affine())
    }

    /// Commit `q2(Y)` (degree `< M_R-1`, length `M_R-1`) on the `tau^0` slice
    /// `[sigma^j]_1` (positions `j*(M_L-1)`, strided). Bases are collected
    /// (size `M_R-1`); callers report the true scalar count in the profile.
    pub(crate) fn msm_sigma_slice(
        &self,
        coeffs: &[E::ScalarField],
    ) -> Result<E::G1Affine, PCSError> {
        let big_ml = self.big_ml();
        let big_mr = self.big_mr();
        if coeffs.len() > big_mr {
            return Err(PCSError::InvalidParameters(format!(
                "sigma-slice polynomial length {} exceeds M_R = {}",
                coeffs.len(),
                big_mr
            )));
        }
        if coeffs.is_empty() {
            return Ok(E::G1Affine::default());
        }
        let mut bases = Vec::with_capacity(coeffs.len());
        for j in 0..coeffs.len() {
            bases.push(self.g1_powers[grid_base_index(big_ml, big_mr, 0, j)]);
        }
        Ok(E::G1::msm_unchecked(&bases, coeffs).into_affine())
    }
}

impl<E: Pairing> ChopinUniversalParams<E> {
    #[inline]
    fn big_ml(&self) -> usize {
        1usize << self.m_left
    }
    #[inline]
    fn big_mr(&self) -> usize {
        1usize << self.m_right
    }

    fn build_params(
        &self,
        num_vars: usize,
    ) -> Result<(ChopinProverParam<E>, ChopinVerifierParam<E>), PCSError> {
        if num_vars < 2 {
            return Err(PCSError::InvalidParameters(format!(
                "chopin requires num_vars >= 2, got {}",
                num_vars
            )));
        }
        if num_vars > self.max_num_vars {
            return Err(PCSError::InvalidParameters(format!(
                "requested num_vars {} exceeds SRS max {}",
                num_vars, self.max_num_vars
            )));
        }
        let (m_left, m_right) = split_exponents(num_vars);
        let big_ml = 1usize << m_left;
        let big_mr = 1usize << m_right;
        let n = big_ml * big_mr;

        let g1_powers = if num_vars == self.max_num_vars {
            // Share the universal key: no N-sized copy.
            Arc::clone(&self.g1_powers)
        } else {
            // Rebuild the smaller-dimension dominant-q1-first layout by reading
            // the maximal grid through its own base_index and re-placing each
            // (i,j) at the smaller layout's base_index (correct τ^i σ^j
            // extraction and reorder, never a raw prefix).
            let max_ml = self.big_ml();
            let max_mr = self.big_mr();
            let mut powers = Vec::with_capacity(n);
            for p in 0..n {
                let (i, j) = inv_base_index(big_ml, big_mr, p);
                let src = grid_base_index(max_ml, max_mr, i, j);
                let base = self.g1_powers.get(src).ok_or_else(|| {
                    PCSError::InvalidParameters(format!(
                        "trim source index {} out of universal SRS range {}",
                        src,
                        self.g1_powers.len()
                    ))
                })?;
                powers.push(*base);
            }
            Arc::new(powers)
        };

        let pp = ChopinProverParam {
            g1_powers: Arc::clone(&g1_powers),
            num_vars,
            m_left,
            m_right,
        };
        let vp = ChopinVerifierParam {
            g1_one: g1_powers[grid_base_index(big_ml, big_mr, 0, 0)],
            g2_one: self.g2_one,
            g2_tau: self.g2_tau,
            g2_sigma: self.g2_sigma,
            max_num_vars: num_vars,
            m_left,
            m_right,
        };
        Ok((pp, vp))
    }
}

impl<E: Pairing> StructuredReferenceString<E> for ChopinUniversalParams<E> {
    type ProverParam = ChopinProverParam<E>;
    type VerifierParam = ChopinVerifierParam<E>;

    fn extract_prover_param(&self, supported_num_vars: usize) -> Self::ProverParam {
        self.build_params(supported_num_vars)
            .expect("extract_prover_param: unsupported size")
            .0
    }

    fn extract_verifier_param(&self, supported_num_vars: usize) -> Self::VerifierParam {
        self.build_params(supported_num_vars)
            .expect("extract_verifier_param: unsupported size")
            .1
    }

    /// `supported_size` is interpreted as the number of variables.
    fn trim(
        &self,
        supported_num_vars: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        self.build_params(supported_num_vars)
    }

    /// Build an SRS supporting up to `supported_num_vars` variables.
    ///
    /// WARNING: FOR TESTING ONLY. The trapdoors are sampled locally.
    fn gen_srs_for_testing<R: Rng>(
        rng: &mut R,
        supported_num_vars: usize,
    ) -> Result<Self, PCSError> {
        if supported_num_vars < 2 {
            return Err(PCSError::InvalidParameters(format!(
                "chopin requires num_vars >= 2, got {}",
                supported_num_vars
            )));
        }
        if supported_num_vars >= usize::BITS as usize {
            return Err(PCSError::InvalidParameters(format!(
                "num_vars {} too large for platform word size",
                supported_num_vars
            )));
        }
        let (m_left, m_right) = split_exponents(supported_num_vars);
        let big_ml = 1usize
            .checked_shl(m_left as u32)
            .ok_or_else(|| PCSError::InvalidParameters("M_L overflow".to_string()))?;
        let big_mr = 1usize
            .checked_shl(m_right as u32)
            .ok_or_else(|| PCSError::InvalidParameters("M_R overflow".to_string()))?;
        let n = big_ml
            .checked_mul(big_mr)
            .ok_or_else(|| PCSError::InvalidParameters("N overflow".to_string()))?;

        let _t_total = profile::ScopedTimer::new(
            BACKEND,
            supported_num_vars,
            n,
            "chopin_srs_total",
            n,
            "srs-gen",
        );

        // Two independent nonzero trapdoors; avoid tau == sigma.
        let tau = loop {
            let t = E::ScalarField::rand(rng);
            if !t.is_zero() {
                break t;
            }
        };
        let sigma = loop {
            let s = E::ScalarField::rand(rng);
            if !s.is_zero() && s != tau {
                break s;
            }
        };
        let g1 = E::G1::rand(rng);
        let g2 = E::G2::rand(rng);

        // Small power tables: M_L + M_R field elements (never the full grid).
        let mut t_tau = profile::MaybeTimer::new();
        let tk = t_tau.start();
        let mut tau_pows = Vec::with_capacity(big_ml);
        let mut acc = E::ScalarField::one();
        for _ in 0..big_ml {
            tau_pows.push(acc);
            acc *= tau;
        }
        t_tau.add(&tk);
        profile::emit_manual(
            BACKEND,
            supported_num_vars,
            n,
            "chopin_srs_build_tau_pows",
            t_tau.ns() as f64 / 1e6,
            big_ml,
            "tau-powers",
        );

        let mut t_sig = profile::MaybeTimer::new();
        let sk = t_sig.start();
        let mut sigma_pows = Vec::with_capacity(big_mr);
        let mut acc = E::ScalarField::one();
        for _ in 0..big_mr {
            sigma_pows.push(acc);
            acc *= sigma;
        }
        t_sig.add(&sk);
        profile::emit_manual(
            BACKEND,
            supported_num_vars,
            n,
            "chopin_srs_build_sigma_pows",
            t_sig.ns() as f64 / 1e6,
            big_mr,
            "sigma-powers",
        );

        let scalar_bits = E::ScalarField::MODULUS_BIT_SIZE as usize;
        let window_size = FixedBase::get_mul_window_size(n);
        let g1_table = FixedBase::get_window_table(scalar_bits, window_size, g1);

        // Chunked generation: reuse the window table, materialise scalars and
        // projective points one chunk at a time.
        let mut g1_powers: Vec<E::G1Affine> = Vec::with_capacity(n);
        let mut t_scal = profile::MaybeTimer::new();
        let mut t_msm = profile::MaybeTimer::new();
        let mut pos = 0usize;
        while pos < n {
            let end = (pos + SRS_GEN_CHUNK).min(n);
            let tk = t_scal.start();
            let mut chunk_scalars = Vec::with_capacity(end - pos);
            for p in pos..end {
                let (i, j) = inv_base_index(big_ml, big_mr, p);
                chunk_scalars.push(tau_pows[i] * sigma_pows[j]);
            }
            t_scal.add(&tk);

            let tk2 = t_msm.start();
            let proj = FixedBase::msm(scalar_bits, window_size, &g1_table, &chunk_scalars);
            let aff = E::G1::normalize_batch(&proj);
            g1_powers.extend(aff);
            t_msm.add(&tk2);

            pos = end;
        }
        profile::emit_manual(
            BACKEND,
            supported_num_vars,
            n,
            "chopin_srs_build_grid_scalars",
            t_scal.ns() as f64 / 1e6,
            n,
            "dominant-q1-first",
        );
        profile::emit_manual(
            BACKEND,
            supported_num_vars,
            n,
            "chopin_srs_fixed_base_msm",
            t_msm.ns() as f64 / 1e6,
            n,
            "fixed-base-chunked",
        );

        let g2_one = g2.into_affine();
        let g2_tau = (g2 * tau).into_affine();
        let g2_sigma = (g2 * sigma).into_affine();

        Ok(ChopinUniversalParams {
            g1_powers: Arc::new(g1_powers),
            g2_one,
            g2_tau,
            g2_sigma,
            max_num_vars: supported_num_vars,
            m_left,
            m_right,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_std::{test_rng, vec, UniformRand};

    type E = Bls12_381;

    // base_index is a bijection of the full grid onto [0, N).
    #[test]
    fn test_base_index_bijection() {
        for &(big_ml, big_mr) in &[(2usize, 2usize), (4, 2), (4, 4), (8, 4), (8, 8), (16, 8)] {
            let n = big_ml * big_mr;
            let mut seen = vec![false; n];
            for j in 0..big_mr {
                for i in 0..big_ml {
                    let p = grid_base_index(big_ml, big_mr, i, j);
                    assert!(p < n, "index {} out of range {}", p, n);
                    assert!(!seen[p], "collision at position {}", p);
                    seen[p] = true;
                    assert_eq!(inv_base_index(big_ml, big_mr, p), (i, j));
                }
            }
            assert!(seen.into_iter().all(|b| b), "not surjective");
        }
    }

    // The q1 prefix region holds exactly [tau^i sigma^j] for i < M_L-1.
    #[test]
    fn test_prefix_region_monomials() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 6;
        let srs = ChopinUniversalParams::<E>::gen_srs_for_testing(&mut rng, nv)?;
        let (pp, _vp) = srs.trim(nv)?;
        let big_ml = pp.big_ml();
        let big_mr = pp.big_mr();
        let q1_len = pp.q1_len();
        for p in 0..q1_len {
            let (i, j) = inv_base_index(big_ml, big_mr, p);
            assert!(i < big_ml - 1);
            assert_eq!(p, grid_base_index(big_ml, big_mr, i, j));
        }
        Ok(())
    }

    // Full commitment via the layout equals the canonical reference MSM.
    #[test]
    fn test_full_commit_matches_reference() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [4usize, 5, 6] {
            let srs = ChopinUniversalParams::<E>::gen_srs_for_testing(&mut rng, nv)?;
            let (pp, _vp) = srs.trim(nv)?;
            let big_ml = pp.big_ml();
            let big_mr = pp.big_mr();
            let n = pp.n();
            let f: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            // Reference: sum F[i,j] * key[base_index(i,j)] directly.
            let mut reference = <E as Pairing>::G1::zero();
            for j in 0..big_mr {
                for i in 0..big_ml {
                    let base = pp.g1_powers[pp.base_index(i, j)];
                    reference += base.into_group() * f[i + big_ml * j];
                }
            }
            assert_eq!(pp.msm_full_reordered(&f)?, reference.into_affine());
        }
        Ok(())
    }

    // SRS shape: exactly N G1 and 3 G2 elements.
    #[test]
    fn test_srs_sizes() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [2usize, 3, 4, 5, 8] {
            let srs = ChopinUniversalParams::<E>::gen_srs_for_testing(&mut rng, nv)?;
            assert_eq!(srs.g1_powers.len(), 1usize << nv, "G1 must be exactly N");
        }
        Ok(())
    }

    #[test]
    fn test_srs_rejects_small_nv() {
        let mut rng = test_rng();
        for nv in [0usize, 1] {
            assert!(ChopinUniversalParams::<E>::gen_srs_for_testing(&mut rng, nv).is_err());
        }
    }

    // Trimming to a smaller size rebuilds a consistent, correct layout.
    #[test]
    fn test_smaller_trim_consistent() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let max_nv = 10;
        let srs = ChopinUniversalParams::<E>::gen_srs_for_testing(&mut rng, max_nv)?;
        for nv in [2usize, 3, 4, 5, 6, 7, 8, 9] {
            let (pp, vp) = srs.trim(nv)?;
            let big_ml = pp.big_ml();
            let big_mr = pp.big_mr();
            assert_eq!(pp.n(), 1usize << nv);
            assert_eq!(pp.g1_powers.len(), 1usize << nv);
            let n = pp.n();
            let f: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let mut reference = <E as Pairing>::G1::zero();
            for j in 0..big_mr {
                for i in 0..big_ml {
                    let base = pp.g1_powers[pp.base_index(i, j)];
                    reference += base.into_group() * f[i + big_ml * j];
                }
            }
            assert_eq!(pp.msm_full_reordered(&f)?, reference.into_affine());
            assert_eq!(
                vp.g1_one,
                pp.g1_powers[grid_base_index(big_ml, big_mr, 0, 0)]
            );
        }
        Ok(())
    }
}
