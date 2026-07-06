//! Structured Reference String for Zeromorph PCS.
//!
//! Zeromorph uses two univariate KZG SRS slices from a single universal SRS:
//! - commit_pp: for committing polynomials and q_i/q_hat
//! - open_pp: for the final KZG opening proof (shifted by offset)
//!
//! The verifier uses s_offset_g2 = powers_of_s_g2[offset] for the pairing
//! check.
//!
//! offset = monomial_g1.len() - poly_size. When gen_srs_for_testing produces
//! 2N powers and trim is called with poly_size=N, offset = N.
//! When gen_srs_for_testing produces >2N powers and trim uses smaller
//! poly_size, offset adjusts accordingly.
//!
//! Reference: han0110/plonkish (MIT-licensed).

use crate::pcs::{prelude::PCSError, profile::ScopedTimer, StructuredReferenceString};
use ark_ec::{
    pairing::Pairing,
    scalar_mul::{fixed_base::FixedBase, variable_base::VariableBaseMSM},
    CurveGroup,
};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{format, rand::Rng, vec::Vec, One, UniformRand};

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct ZeromorphUniversalParams<E: Pairing> {
    pub monomial_g1: Vec<E::G1Affine>,
    pub powers_of_s_g2: Vec<E::G2Affine>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct ZeromorphProverParam<E: Pairing> {
    pub commit_powers: Vec<E::G1Affine>,
    pub open_powers: Vec<E::G1Affine>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct ZeromorphVerifierParam<E: Pairing> {
    pub g: E::G1Affine,
    pub g2: E::G2Affine,
    pub s_g2: E::G2Affine,
    pub s_offset_g2: E::G2Affine,
}

impl<E: Pairing> ZeromorphProverParam<E> {
    pub fn commit_with(&self, powers: &[E::G1Affine], coeffs: &[E::ScalarField]) -> E::G1Affine {
        assert!(coeffs.len() <= powers.len());
        E::G1::msm_unchecked(&powers[..coeffs.len()], coeffs).into_affine()
    }
    pub fn commit_commit(&self, coeffs: &[E::ScalarField]) -> E::G1Affine {
        self.commit_with(&self.commit_powers, coeffs)
    }
    pub fn commit_open(&self, coeffs: &[E::ScalarField]) -> E::G1Affine {
        self.commit_with(&self.open_powers, coeffs)
    }
}

impl<E: Pairing> StructuredReferenceString<E> for ZeromorphUniversalParams<E> {
    type ProverParam = ZeromorphProverParam<E>;
    type VerifierParam = ZeromorphVerifierParam<E>;

    fn extract_prover_param(&self, _size: usize) -> Self::ProverParam {
        unimplemented!()
    }
    fn extract_verifier_param(&self, _size: usize) -> Self::VerifierParam {
        unimplemented!()
    }

    fn trim(&self, poly_size: usize) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        if poly_size > self.monomial_g1.len() {
            return Err(PCSError::InvalidParameters(format!(
                "poly_size {} > SRS max {}",
                poly_size,
                self.monomial_g1.len()
            )));
        }
        let offset = self.monomial_g1.len() - poly_size;

        let commit_powers = self.monomial_g1[..poly_size].to_vec();
        let open_powers = self.monomial_g1[offset..offset + poly_size].to_vec();
        let s_offset_g2 = self.powers_of_s_g2[offset];

        let g_first = commit_powers[0];
        Ok((
            ZeromorphProverParam {
                commit_powers,
                open_powers,
            },
            ZeromorphVerifierParam {
                g: g_first,
                g2: self.powers_of_s_g2[0],
                s_g2: self.powers_of_s_g2[1],
                s_offset_g2,
            },
        ))
    }

    fn gen_srs_for_testing<R: Rng>(
        rng: &mut R,
        supported_num_vars: usize,
    ) -> Result<Self, PCSError> {
        if supported_num_vars == 0 {
            return Err(PCSError::InvalidParameters(
                "num_vars must be > 0".to_string(),
            ));
        }
        let poly_size = 1 << supported_num_vars;
        let total_len = 2 * poly_size;

        let x = E::ScalarField::rand(rng);
        let g = E::G1::rand(rng);
        let h = E::G2::rand(rng);

        // Field powers
        let mut x_pows = Vec::with_capacity(total_len);
        let mut acc = E::ScalarField::one();
        for _ in 0..total_len {
            x_pows.push(acc);
            acc *= x;
        }

        let scalar_bits = E::ScalarField::MODULUS_BIT_SIZE as usize;

        // G1 powers via FixedBase MSM
        let _t_g1 = ScopedTimer::new(
            "Zeromorph",
            supported_num_vars,
            poly_size,
            "zeromorph_srs_g1_powers",
            total_len,
            "G1-fixed-base-msm",
        );
        let window_size = FixedBase::get_mul_window_size(total_len);
        let g_table = FixedBase::get_window_table(scalar_bits, window_size, g);
        let g_proj = FixedBase::msm(scalar_bits, window_size, &g_table, &x_pows);
        let monomial_g1 = E::G1::normalize_batch(&g_proj);
        drop(_t_g1);

        // G2 powers via FixedBase MSM (reuses same x_pows as G1)
        let _t_g2 = ScopedTimer::new(
            "Zeromorph",
            supported_num_vars,
            poly_size,
            "zeromorph_srs_g2_powers",
            total_len,
            "G2-fixed-base-msm",
        );
        let h_window = FixedBase::get_mul_window_size(total_len);
        let h_table = FixedBase::get_window_table(scalar_bits, h_window, h);
        let h_proj = FixedBase::msm(scalar_bits, h_window, &h_table, &x_pows);
        let powers_of_s_g2 = E::G2::normalize_batch(&h_proj);
        drop(_t_g2);

        Ok(ZeromorphUniversalParams {
            monomial_g1,
            powers_of_s_g2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_std::test_rng;

    #[test]
    fn test_zeromorph_srs_gen() -> Result<(), PCSError> {
        let mut rng = test_rng();
        for nv in 4..8 {
            let srs = ZeromorphUniversalParams::<Bls12_381>::gen_srs_for_testing(&mut rng, nv)?;
            let n = 1 << nv;
            let (ck, vk) = srs.trim(n)?;
            assert_eq!(ck.commit_powers[0], vk.g);
            assert_eq!(ck.commit_powers.len(), n);
            assert_eq!(ck.open_powers.len(), n);
            assert_eq!(vk.s_g2, srs.powers_of_s_g2[1]);
            let offset = srs.monomial_g1.len() - n;
            assert_eq!(vk.s_offset_g2, srs.powers_of_s_g2[offset]);
            assert_eq!(ck.open_powers[0], srs.monomial_g1[offset]);
        }
        Ok(())
    }
}
