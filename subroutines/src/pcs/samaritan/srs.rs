//! Structured Reference String for Samaritan PCS.
//!
//! Samaritan needs KZG powers in both G1 and G2 up to N = 2^num_vars.
//! Uses FixedBase MSM for G1/G2 powers generation (same approach as
//! mKZG/Mulcs).

use crate::pcs::{prelude::PCSError, profile::ScopedTimer, StructuredReferenceString};
use ark_ec::{
    pairing::Pairing,
    scalar_mul::{fixed_base::FixedBase, variable_base::VariableBaseMSM},
    CurveGroup,
};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{format, rand::Rng, vec::Vec, One, UniformRand};

const BACKEND: &str = "Samaritan";

/// Universal parameters for Samaritan PCS. Contains prover G1/G2 powers
/// up to `max_degree` = 2^num_vars.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct SamaritanUniversalParams<E: Pairing> {
    pub prover_param: SamaritanProverParam<E>,
    pub verifier_param: SamaritanVerifierParam<E>,
}

/// Prover parameters: G1 powers for MSM, G2 powers for quotient.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct SamaritanProverParam<E: Pairing> {
    pub g1_powers: Vec<E::G1Affine>,
    pub g2_powers: Vec<E::G2Affine>,
    pub max_degree: usize,
    pub max_num_vars: usize,
}

/// Verifier parameters: minimal G1/G2 elements.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct SamaritanVerifierParam<E: Pairing> {
    pub g: E::G1Affine,
    pub h: E::G2Affine,
    pub h_x: E::G2Affine,
    pub h_x_higher: E::G2Affine,
    pub max_degree: usize,
    pub max_num_vars: usize,
}

impl<E: Pairing> SamaritanProverParam<E> {
    pub fn commit(&self, coeffs: &[E::ScalarField]) -> E::G1Affine {
        assert!(
            coeffs.len() <= self.g1_powers.len(),
            "poly degree {} exceeds SRS max {}",
            coeffs.len().saturating_sub(1),
            self.max_degree
        );
        E::G1::msm_unchecked(&self.g1_powers[..coeffs.len()], coeffs).into_affine()
    }
}

impl<E: Pairing> StructuredReferenceString<E> for SamaritanUniversalParams<E> {
    type ProverParam = SamaritanProverParam<E>;
    type VerifierParam = SamaritanVerifierParam<E>;

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
        let max_num_vars = supported_size.trailing_zeros() as usize;
        let ck = SamaritanProverParam {
            g1_powers: self.prover_param.g1_powers[..=supported_size].to_vec(),
            g2_powers: self.prover_param.g2_powers[..=supported_size].to_vec(),
            max_degree: supported_size,
            max_num_vars,
        };
        let vk = SamaritanVerifierParam {
            g: self.verifier_param.g,
            h: self.verifier_param.h,
            h_x: self.prover_param.g2_powers[1],
            h_x_higher: self.prover_param.g2_powers[supported_size],
            max_degree: supported_size,
            max_num_vars,
        };
        Ok((ck, vk))
    }

    /// Build SRS for testing. `supported_num_vars` is the number of
    /// variables. max_degree = 2^num_vars.
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    fn gen_srs_for_testing<R: Rng>(
        rng: &mut R,
        supported_num_vars: usize,
    ) -> Result<Self, PCSError> {
        if supported_num_vars == 0 {
            return Err(PCSError::InvalidParameters(
                "constant polynomial not supported".to_string(),
            ));
        }

        let max_degree = 1 << supported_num_vars;
        let n = max_degree;

        let _t_total = ScopedTimer::new(
            BACKEND,
            supported_num_vars,
            n,
            "srs_gen_total",
            max_degree + 1,
            "total",
        );

        let _t0 = ScopedTimer::new(
            BACKEND,
            supported_num_vars,
            n,
            "srs_gen_sample",
            1,
            "random-x-g1-g2",
        );
        let x = E::ScalarField::rand(rng);
        let g = E::G1::rand(rng);
        let h = E::G2::rand(rng);
        drop(_t0);

        let _t1 = ScopedTimer::new(
            BACKEND,
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

        // G1 powers via FixedBase MSM
        let _t2 = ScopedTimer::new(
            BACKEND,
            supported_num_vars,
            n,
            "samaritan_srs_g1_powers",
            max_degree + 1,
            "G1-fixed-base-msm",
        );
        let scalar_bits = E::ScalarField::MODULUS_BIT_SIZE as usize;
        let window_size = FixedBase::get_mul_window_size(max_degree + 1);
        let g1_table = FixedBase::get_window_table(scalar_bits, window_size, g);
        let g1_projective = FixedBase::msm(scalar_bits, window_size, &g1_table, &x_pows);
        let g1_powers: Vec<E::G1Affine> = E::G1::normalize_batch(&g1_projective);
        drop(_t2);

        // G2 powers via FixedBase MSM
        let _t3 = ScopedTimer::new(
            BACKEND,
            supported_num_vars,
            n,
            "samaritan_srs_g2_powers",
            max_degree + 1,
            "G2-fixed-base-msm",
        );
        let g2_window_size = FixedBase::get_mul_window_size(max_degree + 1);
        let g2_table = FixedBase::get_window_table(scalar_bits, g2_window_size, h);
        let g2_projective = FixedBase::msm(scalar_bits, g2_window_size, &g2_table, &x_pows);
        let g2_powers: Vec<E::G2Affine> = E::G2::normalize_batch(&g2_projective);
        drop(_t3);

        let pp = SamaritanProverParam {
            g1_powers,
            g2_powers,
            max_degree,
            max_num_vars: supported_num_vars,
        };
        let vp = SamaritanVerifierParam {
            g: pp.g1_powers[0],
            h: pp.g2_powers[0],
            h_x: pp.g2_powers[1],
            h_x_higher: pp.g2_powers[max_degree],
            max_degree,
            max_num_vars: supported_num_vars,
        };

        drop(_t_total);

        Ok(SamaritanUniversalParams {
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
    fn test_samaritan_srs_gen() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in [2, 4, 6] {
            let srs = SamaritanUniversalParams::<Bls12_381>::gen_srs_for_testing(&mut rng, nv)?;
            let n = 1 << nv;
            let (ck, vk) = srs.trim(n)?;
            assert_eq!(ck.g1_powers[0], vk.g, "g1_powers[0] != g");
            assert_eq!(ck.g1_powers.len(), n + 1);
            assert_eq!(ck.g2_powers.len(), n + 1);
            assert_eq!(vk.h_x_higher, ck.g2_powers[n]);
        }
        Ok(())
    }
}
