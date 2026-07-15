//! Structured reference string for VelaPCS.
//!
//! VelaPCS uses a plain univariate KZG reference string:
//!   - G1 powers [1]_1, [tau]_1, ..., [tau^{N-1}]_1   (degree bound N-1)
//!   - three G2 elements [1]_2, [tau]_2, [tau^2]_2
//!
//! The degree bound N-1 is tight: the committed polynomials are f_v (degree
//! N-1), hbar (degree N-2), and the opening quotient (degree N-3). No G2 power
//! beyond tau^2 is ever needed by the verifier.
//!
//! `gen_srs_for_testing` is FOR TESTING ONLY; production deployments must
//! obtain the SRS from a trusted setup / powers-of-tau ceremony.

use crate::pcs::{prelude::PCSError, StructuredReferenceString};
use ark_ec::{
    pairing::Pairing,
    scalar_mul::{fixed_base::FixedBase, variable_base::VariableBaseMSM},
    CurveGroup,
};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{format, rand::Rng, string::ToString, vec::Vec, One, UniformRand};

/// Universal parameters for VelaPCS.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct VelaUniversalParams<E: Pairing> {
    pub prover_param: VelaProverParam<E>,
    pub verifier_param: VelaVerifierParam<E>,
}

/// Prover parameters: G1 powers for MSM plus the three verifier G2 elements.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct VelaProverParam<E: Pairing> {
    /// [tau^0]_1 ..= [tau^{max_degree}]_1
    pub g1_powers: Vec<E::G1Affine>,
    pub g2_one: E::G2Affine,
    pub g2_tau: E::G2Affine,
    pub g2_tau2: E::G2Affine,
    pub max_degree: usize,
}

/// Verifier parameters: minimal G1/G2 elements.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct VelaVerifierParam<E: Pairing> {
    pub g1_one: E::G1Affine,
    pub g1_tau: E::G1Affine,
    pub g2_one: E::G2Affine,
    pub g2_tau: E::G2Affine,
    pub g2_tau2: E::G2Affine,
    pub max_degree: usize,
}

impl<E: Pairing> VelaProverParam<E> {
    /// Commit to a univariate polynomial (given by its coefficients) via MSM.
    pub fn commit(&self, coeffs: &[E::ScalarField]) -> Result<E::G1Affine, PCSError> {
        if coeffs.len() > self.g1_powers.len() {
            return Err(PCSError::InvalidParameters(format!(
                "polynomial degree {} exceeds SRS max degree {}",
                coeffs.len().saturating_sub(1),
                self.max_degree
            )));
        }
        let bases = &self.g1_powers[..coeffs.len()];
        Ok(E::G1::msm_unchecked(bases, coeffs).into_affine())
    }
}

impl<E: Pairing> StructuredReferenceString<E> for VelaUniversalParams<E> {
    type ProverParam = VelaProverParam<E>;
    type VerifierParam = VelaVerifierParam<E>;

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
        let ck = VelaProverParam {
            g1_powers: self.prover_param.g1_powers[..=supported_size].to_vec(),
            g2_one: self.prover_param.g2_one,
            g2_tau: self.prover_param.g2_tau,
            g2_tau2: self.prover_param.g2_tau2,
            max_degree: supported_size,
        };
        let vk = VelaVerifierParam {
            g1_one: self.prover_param.g1_powers[0],
            g1_tau: self.prover_param.g1_powers[1],
            g2_one: self.verifier_param.g2_one,
            g2_tau: self.verifier_param.g2_tau,
            g2_tau2: self.verifier_param.g2_tau2,
            max_degree: supported_size,
        };
        Ok((ck, vk))
    }

    /// Build an SRS supporting `supported_num_vars` variables, i.e. polynomials
    /// of degree up to N-1 = 2^{num_vars} - 1.
    ///
    /// WARNING: FOR TESTING ONLY. The trapdoor is known to this routine.
    fn gen_srs_for_testing<R: Rng>(
        rng: &mut R,
        supported_num_vars: usize,
    ) -> Result<Self, PCSError> {
        if supported_num_vars == 0 {
            return Err(PCSError::InvalidParameters(
                "constant polynomial not supported".to_string(),
            ));
        }
        // Tight degree bound: N-1 where N = 2^num_vars.
        let n = 1usize << supported_num_vars;
        let max_degree = n - 1;

        let tau = E::ScalarField::rand(rng);
        let g1 = E::G1::rand(rng);
        let g2 = E::G2::rand(rng);

        // powers of tau: tau^0 ..= tau^{max_degree}
        let mut tau_pows = Vec::with_capacity(max_degree + 1);
        let mut acc = E::ScalarField::one();
        for _ in 0..=max_degree {
            tau_pows.push(acc);
            acc *= tau;
        }

        let scalar_bits = E::ScalarField::MODULUS_BIT_SIZE as usize;
        let window_size = FixedBase::get_mul_window_size(max_degree + 1);
        let g1_table = FixedBase::get_window_table(scalar_bits, window_size, g1);
        let g1_proj = FixedBase::msm(scalar_bits, window_size, &g1_table, &tau_pows);
        let g1_powers: Vec<E::G1Affine> = E::G1::normalize_batch(&g1_proj);

        let g2_one = g2.into_affine();
        let g2_tau = (g2 * tau).into_affine();
        let g2_tau2 = (g2 * tau * tau).into_affine();

        let pp = VelaProverParam {
            g1_powers,
            g2_one,
            g2_tau,
            g2_tau2,
            max_degree,
        };
        let vp = VelaVerifierParam {
            g1_one: pp.g1_powers[0],
            g1_tau: pp.g1_powers[1],
            g2_one,
            g2_tau,
            g2_tau2,
            max_degree,
        };
        Ok(VelaUniversalParams {
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
    fn test_vela_srs_shape() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in 2..8 {
            let srs = VelaUniversalParams::<Bls12_381>::gen_srs_for_testing(&mut rng, nv)?;
            let n = 1usize << nv;
            // tight degree bound: exactly N powers (0..=N-1)
            assert_eq!(srs.prover_param.g1_powers.len(), n);
            assert_eq!(srs.prover_param.max_degree, n - 1);
            let (ck, vk) = srs.trim(n - 1)?;
            assert_eq!(ck.g1_powers[0], vk.g1_one);
            assert_eq!(ck.g1_powers[1], vk.g1_tau);
        }
        Ok(())
    }
}
