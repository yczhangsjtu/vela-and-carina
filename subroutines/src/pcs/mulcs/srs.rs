//! Structured Reference String for Mulcs PCS.
//!
//! Wraps univariate KZG SRS. SRS generation is for testing only.
//! Uses a random field element as the structured randomness γ for
//! the Claymore identity.

use crate::pcs::{mulcs::profile::ScopedTimer, prelude::PCSError, StructuredReferenceString};
use ark_ec::{pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, CurveGroup};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{format, rand::Rng, vec::Vec, One, UniformRand};
use rayon::prelude::*;

/// Universal parameters for Mulcs PCS. Contains prover G1 powers
/// up to `max_degree` and verifier G2 elements.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MulcsUniversalParams<E: Pairing> {
    pub prover_param: MulcsProverParam<E>,
    pub verifier_param: MulcsVerifierParam<E>,
}

/// Prover parameters: G1 powers for MSM, G2 for quotient, gamma for Claymore
/// identity.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MulcsProverParam<E: Pairing> {
    pub g1_powers: Vec<E::G1Affine>,
    pub g2_one: E::G2Affine,
    pub g2_x: E::G2Affine,
    pub g2_x2: E::G2Affine,
    pub gamma: E::ScalarField,
    pub max_degree: usize,
}

/// Verifier parameters: minimal G1/G2 elements plus gamma.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MulcsVerifierParam<E: Pairing> {
    pub g1_one: E::G1Affine,
    pub g1_x: E::G1Affine,
    pub g2_one: E::G2Affine,
    pub g2_x: E::G2Affine,
    pub g2_x2: E::G2Affine,
    pub gamma: E::ScalarField,
    pub max_degree: usize,
}

impl<E: Pairing> MulcsProverParam<E> {
    /// Commit to a univariate polynomial using MSM over G1
    pub fn commit(&self, coeffs: &[E::ScalarField]) -> E::G1Affine {
        assert!(
            coeffs.len() <= self.g1_powers.len(),
            "poly degree {} exceeds SRS max {}",
            coeffs.len() - 1,
            self.max_degree
        );
        let scalars: Vec<_> = coeffs.to_vec();
        let bases = &self.g1_powers[..coeffs.len()];
        E::G1::msm_unchecked(bases, &scalars).into_affine()
    }
}

impl<E: Pairing> StructuredReferenceString<E> for MulcsUniversalParams<E> {
    type ProverParam = MulcsProverParam<E>;
    type VerifierParam = MulcsVerifierParam<E>;

    fn extract_prover_param(&self, _supported_size: usize) -> Self::ProverParam {
        self.prover_param.clone()
    }

    fn extract_verifier_param(&self, _supported_size: usize) -> Self::VerifierParam {
        self.verifier_param.clone()
    }

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
        let ck = MulcsProverParam {
            g1_powers: self.prover_param.g1_powers[..=supported_size].to_vec(),
            g2_one: self.prover_param.g2_one,
            g2_x: self.prover_param.g2_x,
            g2_x2: self.prover_param.g2_x2,
            gamma: self.prover_param.gamma,
            max_degree: supported_size,
        };
        let vk = MulcsVerifierParam {
            g1_one: self.verifier_param.g1_one,
            g1_x: self.prover_param.g1_powers[1],
            g2_one: self.verifier_param.g2_one,
            g2_x: self.verifier_param.g2_x,
            g2_x2: self.verifier_param.g2_x2,
            gamma: self.verifier_param.gamma,
            max_degree: supported_size,
        };
        Ok((ck, vk))
    }

    /// Build SRS for testing. `supported_size` = max supported num_vars.
    /// The needed max_degree is 2 * 2^num_vars (for h̄ which has degree 2N-1).
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(
        rng: &mut R,
        supported_num_vars: usize,
    ) -> Result<Self, PCSError> {
        if supported_num_vars == 0 {
            return Err(PCSError::InvalidParameters(
                "constant polynomial not supported".to_string(),
            ));
        }

        let max_degree = 2 * (1 << supported_num_vars); // 2N
        let n = 1 << supported_num_vars;

        // Total timer — covers everything including pp/vp construction
        let _t_total = ScopedTimer::new(
            supported_num_vars,
            n,
            "srs_gen_total",
            max_degree + 1,
            "total",
        );

        // Phase: sample random x, g1, g2
        let _t0 = ScopedTimer::new(supported_num_vars, n, "srs_gen_sample", 1, "random-x-g1-g2");
        let x = E::ScalarField::rand(rng);
        let g1 = E::G1::rand(rng);
        let g2 = E::G2::rand(rng);
        drop(_t0);

        // Phase: compute powers of x
        let _t1 = ScopedTimer::new(
            supported_num_vars,
            n,
            "srs_gen_x_pows",
            max_degree + 1,
            "field-mults",
        );
        let mut x_pows = Vec::with_capacity(max_degree + 1);
        let mut acc = E::ScalarField::one();
        for _ in 0..=max_degree {
            x_pows.push(acc);
            acc *= x;
        }
        drop(_t1);

        // Phase: compute G1 powers (parallel scalar multiplication)
        let _t2 = ScopedTimer::new(
            supported_num_vars,
            n,
            "srs_gen_g1_powers",
            max_degree + 1,
            "G1-scalar-mult-par",
        );
        let g1_powers: Vec<E::G1Affine> = x_pows
            .par_iter()
            .map(|&xi| (g1 * xi).into_affine())
            .collect();
        drop(_t2);

        // Phase: compute G2 elements
        let _t3 = ScopedTimer::new(supported_num_vars, n, "srs_gen_g2", 3, "G2-elements");
        let g2_one = g2.into_affine();
        let g2_x = (g2 * x).into_affine();
        let g2_x2 = (g2 * x * x).into_affine();
        drop(_t3);

        // Phase: sample gamma
        let _t4 = ScopedTimer::new(supported_num_vars, n, "srs_gen_gamma", 1, "gamma-field");
        let gamma = E::ScalarField::rand(rng);
        drop(_t4);

        let pp = MulcsProverParam {
            g1_powers,
            g2_one,
            g2_x,
            g2_x2,
            gamma,
            max_degree,
        };
        let vp = MulcsVerifierParam {
            g1_one: pp.g1_powers[0],
            g1_x: pp.g1_powers[1],
            g2_one,
            g2_x,
            g2_x2,
            gamma,
            max_degree,
        };

        drop(_t_total);

        Ok(MulcsUniversalParams {
            prover_param: pp,
            verifier_param: vp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_std::test_rng;

    #[test]
    fn test_mulcs_srs_gen() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in 4..10 {
            let srs = MulcsUniversalParams::<Bls12_381>::gen_srs_for_testing(&mut rng, nv)?;
            let (_ck, _vk) = srs.trim(2 * (1 << nv))?;
        }
        Ok(())
    }
}
