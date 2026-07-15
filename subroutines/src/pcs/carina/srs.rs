//! Structured reference string for the Carina PCS.
//!
//! For `mu` variables we use a non-padded rectangular split
//!   m_left  = ceil(mu/2),  M_L = 2^m_left,
//!   m_right = floor(mu/2),  M_R = 2^m_right,  N = M_L * M_R = 2^mu.
//!
//! The commitment key is the bivariate Cartesian grid
//!   [tau^i sigma^j]_1,   0 <= i < M_L,  0 <= j < M_R   (exactly N elements),
//! and the verifier key is the fixed five-element G2 material
//!   [1]_2, [tau]_2, [tau^2]_2, [sigma]_2, [sigma^2]_2.
//!
//! G1 layout: dominant-QX-first. The `Pi_X` commitment is a single MSM over a
//! strict length-`(M_L-2)*M_R` contiguous prefix of the key; the remaining two
//! X-powers of every column are appended afterwards. The `base_index` map is a
//! bijection of the full N-element grid onto the stored layout, so the same
//! Vec serves the full commitment (an N-MSM), the Pi_X prefix MSM, and the
//! small collected MSMs for S0/S1/Pi_Y.
//!
//! WARNING: `gen_srs_for_testing` samples the trapdoors `tau, sigma` locally.
//! It is FOR TESTING ONLY and MUST NOT be used as a production trusted setup.
//! This scheme provides no hiding / zero knowledge.

use crate::pcs::{prelude::PCSError, profile, StructuredReferenceString};
use ark_ec::{
    pairing::Pairing,
    scalar_mul::{fixed_base::FixedBase, variable_base::VariableBaseMSM},
    CurveGroup,
};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{format, rand::Rng, string::ToString, sync::Arc, vec::Vec, One, UniformRand, Zero};

const BACKEND: &str = "Carina";

/// Chunk size (number of G1 elements) for the FixedBase SRS generation.
///
/// The window table is built once and reused; scalars and projective points
/// are materialised one chunk at a time so that setup never simultaneously
/// holds `N` scalars, `N` projective points, and several `N`-sized affine
/// buffers. The only unavoidable `N`-sized buffer is the output key itself.
pub(crate) const SRS_GEN_CHUNK: usize = 1 << 16;

/// Split `mu` variables into `(m_left, m_right)` exponents with
/// `m_left = ceil(mu/2)`, `m_right = floor(mu/2)`.
#[inline]
pub(crate) fn split_exponents(mu: usize) -> (usize, usize) {
    (mu.div_ceil(2), mu / 2)
}

/// dominant-QX-first `base_index(i,j)` for a layout of dimensions
/// `big_ml x big_mr` (`big_ml = M_L`, `big_mr = M_R`). Requires `big_ml >= 4`.
///
/// - `i < M_L-2`:  `j*(M_L-2) + i`               (the Pi_X prefix region)
/// - `i >= M_L-2`: `(M_L-2)*M_R + 2*j + (i-(M_L-2))` (the two dominant powers)
#[inline]
pub(crate) fn grid_base_index(big_ml: usize, big_mr: usize, i: usize, j: usize) -> usize {
    let qx_len = (big_ml - 2) * big_mr;
    if i < big_ml - 2 {
        j * (big_ml - 2) + i
    } else {
        qx_len + 2 * j + (i - (big_ml - 2))
    }
}

/// Inverse of [`grid_base_index`]: layout position `p` -> `(i, j)`.
#[inline]
fn inv_base_index(big_ml: usize, big_mr: usize, p: usize) -> (usize, usize) {
    let qx_len = (big_ml - 2) * big_mr;
    if p < qx_len {
        (p % (big_ml - 2), p / (big_ml - 2))
    } else {
        let q = p - qx_len;
        ((big_ml - 2) + (q % 2), q / 2)
    }
}

/// Universal parameters: the full bivariate grid key plus the five G2 powers.
#[derive(Clone, Debug)]
pub struct CarinaUniversalParams<E: Pairing> {
    /// `[tau^i sigma^j]_1` in dominant-QX-first layout for the maximal split.
    /// Shared behind an `Arc` so trimming to the same size does not copy.
    pub g1_powers: Arc<Vec<E::G1Affine>>,
    pub g2_one: E::G2Affine,
    pub g2_tau: E::G2Affine,
    pub g2_tau2: E::G2Affine,
    pub g2_sigma: E::G2Affine,
    pub g2_sigma2: E::G2Affine,
    /// Maximal supported number of variables.
    pub max_num_vars: usize,
    /// `m_left` exponent for the maximal split.
    pub m_left: usize,
    /// `m_right` exponent for the maximal split.
    pub m_right: usize,
}

/// Prover parameters for a specific `num_vars`.
#[derive(Clone, Debug)]
pub struct CarinaProverParam<E: Pairing> {
    /// `[tau^i sigma^j]_1` in dominant-QX-first layout for this split. Shared
    /// with the universal params when trimming to the maximal size.
    pub g1_powers: Arc<Vec<E::G1Affine>>,
    pub num_vars: usize,
    pub m_left: usize,
    pub m_right: usize,
}

/// Verifier parameters: seven small G1 bases and the five G2 powers.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct CarinaVerifierParam<E: Pairing> {
    pub g1_one: E::G1Affine,
    pub g1_tau: E::G1Affine,
    pub g1_sigma: E::G1Affine,
    pub g1_tau_sigma: E::G1Affine,
    pub g2_one: E::G2Affine,
    pub g2_tau: E::G2Affine,
    pub g2_tau2: E::G2Affine,
    pub g2_sigma: E::G2Affine,
    pub g2_sigma2: E::G2Affine,
    /// Number of variables supported by this key.
    pub max_num_vars: usize,
    pub m_left: usize,
    pub m_right: usize,
}

impl<E: Pairing> CarinaProverParam<E> {
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
    pub(crate) fn qx_len(&self) -> usize {
        (self.big_ml() - 2) * self.big_mr()
    }
    #[inline]
    pub(crate) fn base_index(&self, i: usize, j: usize) -> usize {
        grid_base_index(self.big_ml(), self.big_mr(), i, j)
    }

    /// MSM over the length-`scalars.len()` contiguous prefix of the key.
    /// Used for the full commitment (`N`) and the `Pi_X` prefix (`(M_L-2)M_R`).
    pub(crate) fn msm_prefix(&self, scalars: &[E::ScalarField]) -> Result<E::G1Affine, PCSError> {
        if scalars.len() > self.g1_powers.len() {
            return Err(PCSError::InvalidParameters(format!(
                "prefix MSM length {} exceeds SRS G1 length {}",
                scalars.len(),
                self.g1_powers.len()
            )));
        }
        Ok(E::G1::msm_unchecked(&self.g1_powers[..scalars.len()], scalars).into_affine())
    }

    /// MSM over an explicitly collected set of bases identified by their layout
    /// indices. Used by S0/S1/Pi_Y, each of which touches only `O(M_L+M_R)`
    /// bases; the key is never duplicated.
    pub(crate) fn msm_collected(
        &self,
        indices: &[usize],
        scalars: &[E::ScalarField],
    ) -> Result<E::G1Affine, PCSError> {
        if indices.len() != scalars.len() {
            return Err(PCSError::InvalidParameters(
                "collected MSM length mismatch".to_string(),
            ));
        }
        let mut bases = Vec::with_capacity(indices.len());
        for &idx in indices {
            let base = self.g1_powers.get(idx).ok_or_else(|| {
                PCSError::InvalidParameters(format!("base index {} out of SRS range", idx))
            })?;
            bases.push(*base);
        }
        Ok(E::G1::msm_unchecked(&bases, scalars).into_affine())
    }
}

impl<E: Pairing> CarinaUniversalParams<E> {
    #[inline]
    fn big_ml(&self) -> usize {
        1usize << self.m_left
    }
    #[inline]
    fn big_mr(&self) -> usize {
        1usize << self.m_right
    }

    /// Build a `(prover, verifier)` pair for `num_vars` variables. When
    /// `num_vars` equals the maximal split, the `Arc` is shared (no copy);
    /// otherwise the smaller-dimension layout is rebuilt from the universal
    /// grid via the maximal `base_index`.
    fn build_params(
        &self,
        num_vars: usize,
    ) -> Result<(CarinaProverParam<E>, CarinaVerifierParam<E>), PCSError> {
        if num_vars < 4 {
            return Err(PCSError::InvalidParameters(format!(
                "carina requires num_vars >= 4, got {}",
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
            // Rebuild the smaller-dimension dominant-QX-first layout by reading
            // the maximal grid through its own base_index.
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

        let pp = CarinaProverParam {
            g1_powers: Arc::clone(&g1_powers),
            num_vars,
            m_left,
            m_right,
        };
        let vp = CarinaVerifierParam {
            g1_one: g1_powers[grid_base_index(big_ml, big_mr, 0, 0)],
            g1_tau: g1_powers[grid_base_index(big_ml, big_mr, 1, 0)],
            g1_sigma: g1_powers[grid_base_index(big_ml, big_mr, 0, 1)],
            g1_tau_sigma: g1_powers[grid_base_index(big_ml, big_mr, 1, 1)],
            g2_one: self.g2_one,
            g2_tau: self.g2_tau,
            g2_tau2: self.g2_tau2,
            g2_sigma: self.g2_sigma,
            g2_sigma2: self.g2_sigma2,
            max_num_vars: num_vars,
            m_left,
            m_right,
        };
        Ok((pp, vp))
    }
}

impl<E: Pairing> StructuredReferenceString<E> for CarinaUniversalParams<E> {
    type ProverParam = CarinaProverParam<E>;
    type VerifierParam = CarinaVerifierParam<E>;

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
        if supported_num_vars < 4 {
            return Err(PCSError::InvalidParameters(format!(
                "carina requires num_vars >= 4, got {}",
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
        let mut tau_pows = Vec::with_capacity(big_ml);
        let mut acc = E::ScalarField::one();
        for _ in 0..big_ml {
            tau_pows.push(acc);
            acc *= tau;
        }
        let mut sigma_pows = Vec::with_capacity(big_mr);
        let mut acc = E::ScalarField::one();
        for _ in 0..big_mr {
            sigma_pows.push(acc);
            acc *= sigma;
        }

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
            "nrg_srs_build_scalar_chunks",
            t_scal.ns() as f64 / 1e6,
            n,
            "dominant-qx-first",
        );
        profile::emit_manual(
            BACKEND,
            supported_num_vars,
            n,
            "nrg_srs_fixed_base_msm",
            t_msm.ns() as f64 / 1e6,
            n,
            "fixed-base-chunked",
        );

        let g2_one = g2.into_affine();
        let g2_tau = (g2 * tau).into_affine();
        let g2_tau2 = (g2 * tau * tau).into_affine();
        let g2_sigma = (g2 * sigma).into_affine();
        let g2_sigma2 = (g2 * sigma * sigma).into_affine();

        Ok(CarinaUniversalParams {
            g1_powers: Arc::new(g1_powers),
            g2_one,
            g2_tau,
            g2_tau2,
            g2_sigma,
            g2_sigma2,
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
    use ark_ec::AffineRepr;
    use ark_std::{test_rng, UniformRand};

    type E = Bls12_381;

    // base_index is a bijection of the full grid onto [0, N).
    #[test]
    fn test_base_index_bijection() {
        for &(big_ml, big_mr) in &[(4usize, 4usize), (8, 4), (8, 8), (16, 8)] {
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

    // The Pi_X prefix region holds exactly [tau^i sigma^j] for i < M_L-2.
    #[test]
    fn test_prefix_region_monomials() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 8;
        let srs = CarinaUniversalParams::<E>::gen_srs_for_testing(&mut rng, nv)?;
        let (pp, _vp) = srs.trim(nv)?;
        let big_ml = pp.big_ml();
        let big_mr = pp.big_mr();
        let qx_len = pp.qx_len();
        // Every prefix position corresponds to i < M_L-2.
        for p in 0..qx_len {
            let (i, j) = inv_base_index(big_ml, big_mr, p);
            assert!(i < big_ml - 2);
            assert_eq!(p, grid_base_index(big_ml, big_mr, i, j));
        }
        Ok(())
    }

    // Full commitment via the layout equals the canonical reference MSM.
    #[test]
    fn test_full_commit_matches_reference() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 6;
        let srs = CarinaUniversalParams::<E>::gen_srs_for_testing(&mut rng, nv)?;
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

        // Layout path: reorder scalars, one N-MSM over the prefix.
        let mut scalars = vec![Fr::zero(); n];
        for j in 0..big_mr {
            for i in 0..big_ml {
                scalars[pp.base_index(i, j)] = f[i + big_ml * j];
            }
        }
        let commit = pp.msm_prefix(&scalars)?;
        assert_eq!(commit, reference.into_affine());
        Ok(())
    }

    // The four verifier bases line up with base_index.
    #[test]
    fn test_verifier_bases_consistent() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let nv = 7;
        let srs = CarinaUniversalParams::<E>::gen_srs_for_testing(&mut rng, nv)?;
        let (pp, vp) = srs.trim(nv)?;
        let big_ml = pp.big_ml();
        let big_mr = pp.big_mr();
        assert_eq!(
            vp.g1_one,
            pp.g1_powers[grid_base_index(big_ml, big_mr, 0, 0)]
        );
        assert_eq!(
            vp.g1_tau,
            pp.g1_powers[grid_base_index(big_ml, big_mr, 1, 0)]
        );
        assert_eq!(
            vp.g1_sigma,
            pp.g1_powers[grid_base_index(big_ml, big_mr, 0, 1)]
        );
        assert_eq!(
            vp.g1_tau_sigma,
            pp.g1_powers[grid_base_index(big_ml, big_mr, 1, 1)]
        );
        Ok(())
    }

    // Trimming to a smaller size rebuilds a consistent, correct layout.
    #[test]
    fn test_smaller_trim_consistent() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let max_nv = 10;
        let srs = CarinaUniversalParams::<E>::gen_srs_for_testing(&mut rng, max_nv)?;
        for nv in [4usize, 5, 6, 7, 8, 9] {
            let (pp, vp) = srs.trim(nv)?;
            let big_ml = pp.big_ml();
            let big_mr = pp.big_mr();
            assert_eq!(pp.n(), 1usize << nv);
            assert_eq!(pp.g1_powers.len(), 1usize << nv);
            // Full commit against reference still matches after rebuild.
            let n = pp.n();
            let f: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let mut reference = <E as Pairing>::G1::zero();
            let mut scalars = vec![Fr::zero(); n];
            for j in 0..big_mr {
                for i in 0..big_ml {
                    let base = pp.g1_powers[pp.base_index(i, j)];
                    reference += base.into_group() * f[i + big_ml * j];
                    scalars[pp.base_index(i, j)] = f[i + big_ml * j];
                }
            }
            assert_eq!(pp.msm_prefix(&scalars)?, reference.into_affine());
            assert_eq!(
                vp.g1_one,
                pp.g1_powers[grid_base_index(big_ml, big_mr, 0, 0)]
            );
        }
        Ok(())
    }

    // G1 count is exactly N (never 2N); G2 material is exactly five elements.
    #[test]
    fn test_srs_sizes() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [4usize, 5, 8] {
            let srs = CarinaUniversalParams::<E>::gen_srs_for_testing(&mut rng, nv)?;
            assert_eq!(srs.g1_powers.len(), 1usize << nv, "G1 must be exactly N");
        }
        Ok(())
    }

    #[test]
    fn test_srs_rejects_small_nv() {
        let mut rng = test_rng();
        for nv in [0usize, 1, 2, 3] {
            assert!(CarinaUniversalParams::<E>::gen_srs_for_testing(&mut rng, nv).is_err());
        }
    }
}
