//! HyperPlonk + Chopin backend correctness tests.
//!
//! Run with: cargo test -p hyperplonk --test chopin_backend -- --nocapture

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
                CarinaPCS, ChopinPCS, GeminiPCS, MercuryPCS, MultilinearKzgPCS, SamaritanPCS,
                VelaPCS, ZeromorphPCS,
            },
            PolynomialCommitmentScheme,
        },
        poly_iop::PolyIOP,
    };

    type E = Bls12_381;
    type FrType = Fr;

    #[test]
    fn test_hyperplonk_chopin_e2e() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let pcs_srs = ChopinPCS::<E>::gen_srs_for_testing(&mut rng, 15)?;

        for nv in [4usize, 5, 6] {
            let size = 1 << nv;
            let circuit = MockCircuit::<FrType>::new(size, &gates);
            assert!(circuit.is_satisfied(), "circuit not satisfied at nv={nv}");

            let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, ChopinPCS<E>>>::preprocess(
                &circuit.index,
                &pcs_srs,
            )?;
            let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, ChopinPCS<E>>>::prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
            )?;
            let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, ChopinPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?;
            assert!(ok, "HyperPlonk+Chopin verify failed at nv={nv}");
        }
        Ok(())
    }

    #[test]
    fn test_hyperplonk_chopin_rejects_tampered_public_input() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let pcs_srs = ChopinPCS::<E>::gen_srs_for_testing(&mut rng, 13)?;
        let nv = 5;
        let size = 1 << nv;
        let circuit = MockCircuit::<FrType>::new(size, &gates);
        let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, ChopinPCS<E>>>::preprocess(
            &circuit.index,
            &pcs_srs,
        )?;
        let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, ChopinPCS<E>>>::prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        let mut bad_inputs = circuit.public_inputs.clone();
        if !bad_inputs.is_empty() {
            bad_inputs[0] += Fr::from(1u64);
            let res = <PolyIOP<FrType> as HyperPlonkSNARK<E, ChopinPCS<E>>>::verify(
                &vk,
                &bad_inputs,
                &proof,
            );
            assert!(
                res.is_err() || !res.unwrap(),
                "tampered PI must be rejected"
            );
        }
        Ok(())
    }

    #[test]
    fn test_hyperplonk_cross_backend_all() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let nv = 5;
        let size = 1 << nv;
        let circuit = MockCircuit::<FrType>::new(size, &gates);
        assert!(circuit.is_satisfied());

        macro_rules! run_backend {
            ($pcs:ty, $name:expr) => {{
                let srs = <$pcs>::gen_srs_for_testing(&mut rng, 13)?;
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
                assert!(ok, "cross-backend {} failed", $name);
            }};
        }

        run_backend!(MultilinearKzgPCS<E>, "mKZG");
        run_backend!(GeminiPCS<E>, "Gemini");
        run_backend!(VelaPCS<E>, "Vela");
        run_backend!(ZeromorphPCS<E>, "Zeromorph");
        run_backend!(SamaritanPCS<E>, "Samaritan");
        run_backend!(CarinaPCS<E>, "Carina");
        run_backend!(MercuryPCS<E>, "Mercury");
        run_backend!(ChopinPCS<E>, "Chopin");
        Ok(())
    }
}
