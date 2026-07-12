//! HyperPlonk + NestedGridKzgPCS backend correctness tests.
//!
//! Run with:
//!   cargo test -p hyperplonk --test nested_grid_kzg_backend -- --nocapture

#[cfg(test)]
mod tests {
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_std::test_rng;
    use hyperplonk::{
        prelude::{CustomizedGates, HyperPlonkErrors, MockCircuit},
        HyperPlonkSNARK,
    };
    use subroutines::{
        pcs::{
            prelude::{
                GeminiPCS, MultilinearKzgPCS, NestedGridKzgPCS, ReciPCS, SamaritanPCS, ZeromorphPCS,
            },
            PolynomialCommitmentScheme,
        },
        poly_iop::PolyIOP,
    };

    type E = Bls12_381;
    type FrType = Fr;

    #[test]
    fn test_hyperplonk_nested_grid_kzg_e2e() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let pcs_srs = NestedGridKzgPCS::<E>::gen_srs_for_testing(&mut rng, 12)?;

        for nv in [4usize, 5, 6] {
            let size = 1 << nv;
            let circuit = MockCircuit::<FrType>::new(size, &gates);
            assert!(circuit.is_satisfied(), "circuit not satisfied at nv={nv}");

            let (pk, vk) =
                <PolyIOP<FrType> as HyperPlonkSNARK<E, NestedGridKzgPCS<E>>>::preprocess(
                    &circuit.index,
                    &pcs_srs,
                )?;
            let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, NestedGridKzgPCS<E>>>::prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
            )?;
            let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, NestedGridKzgPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?;
            assert!(ok, "HyperPlonk+NestedGridKZG verify failed at nv={nv}");
        }
        Ok(())
    }

    #[test]
    fn test_hyperplonk_nested_grid_kzg_rejects_tampered_public_input(
    ) -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let pcs_srs = NestedGridKzgPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;
        let nv = 5;
        let size = 1 << nv;
        let circuit = MockCircuit::<FrType>::new(size, &gates);
        let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, NestedGridKzgPCS<E>>>::preprocess(
            &circuit.index,
            &pcs_srs,
        )?;
        let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, NestedGridKzgPCS<E>>>::prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        let mut bad_inputs = circuit.public_inputs.clone();
        if !bad_inputs.is_empty() {
            bad_inputs[0] += Fr::from(1u64);
            let res = <PolyIOP<FrType> as HyperPlonkSNARK<E, NestedGridKzgPCS<E>>>::verify(
                &vk,
                &bad_inputs,
                &proof,
            );
            assert!(
                res.is_err() || !res.unwrap(),
                "tampered public input must be rejected"
            );
        }
        Ok(())
    }

    // Cross-backend correctness: the same circuit is provable and verifiable
    // under every PCS backend in the tree, giving independent confidence in
    // the NestedGridKZG integration.
    #[test]
    fn test_hyperplonk_cross_backend_all() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let nv = 5;
        let size = 1 << nv;
        let circuit = MockCircuit::<FrType>::new(size, &gates);
        assert!(circuit.is_satisfied());

        macro_rules! run_backend {
            ($pcs:ty, $label:literal) => {{
                let srs = <$pcs>::gen_srs_for_testing(&mut rng, 12)?;
                let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, $pcs>>::preprocess(
                    &circuit.index,
                    &srs,
                )?;
                let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, $pcs>>::prove(
                    &pk,
                    &circuit.public_inputs,
                    &circuit.witnesses,
                )?;
                let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, $pcs>>::verify(
                    &vk,
                    &circuit.public_inputs,
                    &proof,
                )?;
                assert!(ok, "cross-backend verify failed for {}", $label);
            }};
        }

        run_backend!(MultilinearKzgPCS<E>, "mKZG");
        run_backend!(ZeromorphPCS<E>, "Zeromorph");
        run_backend!(SamaritanPCS<E>, "Samaritan");
        run_backend!(GeminiPCS<E>, "Gemini");
        run_backend!(ReciPCS<E>, "ReciPCS");
        run_backend!(NestedGridKzgPCS<E>, "NestedGridKZG");
        Ok(())
    }
}
