//! Structured reference string for the Mercury ml-PCS.
//!
//! Mercury uses a plain univariate KZG reference string (the *tight* Mercury
//! SRS, not the Claymore 2N SRS):
//!   - G1 powers `[tau^0]_1 ..= [tau^{N-1}]_1`  (exactly `N = 2^mu` elements)
//!   - two G2 elements `[1]_2`, `[tau]_2`
//!
//! The maximum committed degree is `N-1` (the polynomial `f` itself and the
//! opening quotient `quot_f` of degree `N-2`); `q` has degree `N-b-1`; the
//! `O(sqrt N)` helper polynomials `g,h,s,d,w,w'` all have degree `< b`. So `N`
//! G1 powers are necessary and sufficient. Every pairing check has the form
//! `e(L,[1]_2) = e(R,[tau]_2)`, so no G2 power beyond `[tau]_2` is used.
//!
//! `gen_srs_for_testing` samples the trapdoor `tau` locally and is FOR TESTING
//! ONLY; production deployments must obtain the SRS from a powers-of-tau
//! ceremony. No hiding / zero knowledge is provided.

use crate::pcs::{prelude::PCSError, profile, StructuredReferenceString};
use ark_ec::{
    pairing::Pairing,
    scalar_mul::{fixed_base::FixedBase, variable_base::VariableBaseMSM},
    CurveGroup,
};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{format, rand::Rng, string::ToString, vec::Vec, One, UniformRand};

const BACKEND: &str = "Mercury";

/// Chunk size (number of G1 elements) for the FixedBase SRS generation. The
/// window table is built once and reused; scalars/projective points are
/// materialised one chunk at a time so setup never simultaneously holds several
/// `N`-sized scalar/projective/affine buffers (only the output key is `N`).
pub(crate) const SRS_GEN_CHUNK: usize = 1 << 16;

/// Universal parameters for Mercury.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MercuryUniversalParams<E: Pairing> {
    pub prover_param: MercuryProverParam<E>,
    pub verifier_param: MercuryVerifierParam<E>,
}

/// Prover parameters: G1 powers for the commitment MSMs.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MercuryProverParam<E: Pairing> {
    /// `[tau^0]_1 ..= [tau^{max_degree}]_1`.
    pub g1_powers: Vec<E::G1Affine>,
    pub max_degree: usize,
}

/// Verifier parameters: three group elements plus the degree bound.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct MercuryVerifierParam<E: Pairing> {
    /// `[1]_1` (the G1 generator), used as a base in the verifier MSMs.
    pub g1_one: E::G1Affine,
    /// `[1]_2`.
    pub g2_one: E::G2Affine,
    /// `[tau]_2`.
    pub g2_tau: E::G2Affine,
    /// `max_degree = N-1`.
    pub max_degree: usize,
}

impl<E: Pairing> MercuryProverParam<E> {
    /// Commit to a univariate polynomial (given by a borrowed coefficient
    /// slice) via an MSM over the matching G1 power prefix. Never copies
    /// the slice.
    pub fn commit(&self, coeffs: &[E::ScalarField]) -> Result<E::G1Affine, PCSError> {
        if coeffs.len() > self.g1_powers.len() {
            return Err(PCSError::InvalidParameters(format!(
                "polynomial degree {} exceeds SRS max degree {}",
                coeffs.len().saturating_sub(1),
                self.max_degree
            )));
        }
        if coeffs.is_empty() {
            return Ok(E::G1Affine::default());
        }
        let bases = &self.g1_powers[..coeffs.len()];
        Ok(E::G1::msm_unchecked(bases, coeffs).into_affine())
    }
}

impl<E: Pairing> StructuredReferenceString<E> for MercuryUniversalParams<E> {
    type ProverParam = MercuryProverParam<E>;
    type VerifierParam = MercuryVerifierParam<E>;

    fn extract_prover_param(&self, _supported_size: usize) -> Self::ProverParam {
        self.prover_param.clone()
    }

    fn extract_verifier_param(&self, _supported_size: usize) -> Self::VerifierParam {
        self.verifier_param.clone()
    }

    /// `supported_size` is the maximum polynomial degree (`N-1`).
    fn trim(
        &self,
        supported_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        if supported_size > self.prover_param.max_degree {
            return Err(PCSError::InvalidParameters(format!(
                "requested degree {} exceeds SRS max {}",
                supported_size, self.prover_param.max_degree
            )));
        }
        let end = supported_size
            .checked_add(1)
            .ok_or_else(|| PCSError::InvalidParameters("degree+1 overflow".to_string()))?;
        if end > self.prover_param.g1_powers.len() {
            return Err(PCSError::InvalidParameters(
                "trim range exceeds available G1 powers".to_string(),
            ));
        }
        let ck = MercuryProverParam {
            g1_powers: self.prover_param.g1_powers[..end].to_vec(),
            max_degree: supported_size,
        };
        let vk = MercuryVerifierParam {
            g1_one: self.prover_param.g1_powers[0],
            g2_one: self.verifier_param.g2_one,
            g2_tau: self.verifier_param.g2_tau,
            max_degree: supported_size,
        };
        Ok((ck, vk))
    }

    /// Build an SRS supporting `supported_num_vars` variables, i.e. degree up
    /// to `N-1 = 2^{num_vars} - 1`.
    ///
    /// WARNING: FOR TESTING ONLY. The trapdoor is known to this routine.
    fn gen_srs_for_testing<R: Rng>(
        rng: &mut R,
        supported_num_vars: usize,
    ) -> Result<Self, PCSError> {
        if supported_num_vars == 0 {
            return Err(PCSError::InvalidParameters(
                "constant polynomial not supported (num_vars = 0)".to_string(),
            ));
        }
        if supported_num_vars >= usize::BITS as usize {
            return Err(PCSError::InvalidParameters(format!(
                "num_vars {} too large for platform word size",
                supported_num_vars
            )));
        }
        // Tight degree bound: N-1 where N = 2^num_vars.
        let n = 1usize
            .checked_shl(supported_num_vars as u32)
            .ok_or_else(|| PCSError::InvalidParameters("N overflow in shift".to_string()))?;
        let max_degree = n - 1;

        let _t_total = profile::ScopedTimer::new(
            BACKEND,
            supported_num_vars,
            n,
            "mercury_srs_total",
            n,
            "srs-gen",
        );

        let tau = E::ScalarField::rand(rng);
        let g1 = E::G1::rand(rng);
        let g2 = E::G2::rand(rng);

        let scalar_bits = E::ScalarField::MODULUS_BIT_SIZE as usize;
        let window_size = FixedBase::get_mul_window_size(n);
        let g1_table = FixedBase::get_window_table(scalar_bits, window_size, g1);

        // Chunked generation of tau powers: reuse the window table, materialise
        // scalars/projective points one chunk at a time.
        let mut g1_powers: Vec<E::G1Affine> = Vec::with_capacity(n);
        let mut t_scal = profile::MaybeTimer::new();
        let mut t_msm = profile::MaybeTimer::new();
        let mut acc = E::ScalarField::one();
        let mut pos = 0usize;
        while pos < n {
            let end = (pos + SRS_GEN_CHUNK).min(n);
            let tk = t_scal.start();
            let mut chunk_scalars = Vec::with_capacity(end - pos);
            for _ in pos..end {
                chunk_scalars.push(acc);
                acc *= tau;
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
            "mercury_srs_build_powers",
            t_scal.ns() as f64 / 1e6,
            n,
            "tau-powers-chunked",
        );
        profile::emit_manual(
            BACKEND,
            supported_num_vars,
            n,
            "mercury_srs_fixed_base_msm",
            t_msm.ns() as f64 / 1e6,
            n,
            "fixed-base-chunked",
        );

        let g2_one = g2.into_affine();
        let g2_tau = (g2 * tau).into_affine();

        let pp = MercuryProverParam {
            g1_powers,
            max_degree,
        };
        let vp = MercuryVerifierParam {
            g1_one: pp.g1_powers[0],
            g2_one,
            g2_tau,
            max_degree,
        };
        Ok(MercuryUniversalParams {
            prover_param: pp,
            verifier_param: vp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::StructuredReferenceString;
    use ark_bls12_381::Bls12_381;
    use ark_std::test_rng;

    type E = Bls12_381;

    #[test]
    fn test_mercury_srs_shape() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in 2..10 {
            let srs = MercuryUniversalParams::<E>::gen_srs_for_testing(&mut rng, nv)?;
            let n = 1usize << nv;
            // Exactly N G1 powers (tight); exactly two G2 elements in the vk.
            assert_eq!(srs.prover_param.g1_powers.len(), n, "G1 count must be N");
            assert_eq!(srs.prover_param.max_degree, n - 1);
            // g1_powers[0] and the verifier g1_one agree.
            assert_eq!(srs.prover_param.g1_powers[0], srs.verifier_param.g1_one);
            let (ck, vk) = srs.trim(n - 1)?;
            assert_eq!(ck.g1_powers[0], vk.g1_one);
            assert_eq!(ck.g1_powers.len(), n);
        }
        Ok(())
    }

    #[test]
    fn test_mercury_srs_trim_prefix_consistent() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let max_nv = 10;
        let srs = MercuryUniversalParams::<E>::gen_srs_for_testing(&mut rng, max_nv)?;
        // A trimmed commitment equals the corresponding prefix commitment.
        let (full_ck, _) = srs.trim((1usize << max_nv) - 1)?;
        for nv in [2usize, 4, 6, 8] {
            let (ck, vk) = srs.trim((1usize << nv) - 1)?;
            assert_eq!(ck.g1_powers.len(), 1usize << nv);
            assert_eq!(vk.max_degree, (1usize << nv) - 1);
            // prefix of the powers matches
            assert_eq!(
                ck.g1_powers[..],
                full_ck.g1_powers[..(1usize << nv)],
                "trim prefix mismatch nv={nv}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_mercury_srs_rejects_bad_sizes() {
        let mut rng = test_rng();
        assert!(MercuryUniversalParams::<E>::gen_srs_for_testing(&mut rng, 0).is_err());
        assert!(
            MercuryUniversalParams::<E>::gen_srs_for_testing(&mut rng, usize::BITS as usize)
                .is_err()
        );
        // trim beyond the built degree returns Err (does not panic)
        let srs = MercuryUniversalParams::<E>::gen_srs_for_testing(&mut rng, 4).unwrap();
        assert!(srs.trim(1usize << 4).is_err());
    }
}
