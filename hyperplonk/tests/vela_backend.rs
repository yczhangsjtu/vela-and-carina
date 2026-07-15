//! HyperPlonk + Vela backend correctness tests.
//!
//! Run with: cargo test -p hyperplonk vela_backend -- --nocapture

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
            prelude::{MultilinearKzgPCS, VelaPCS},
            PolynomialCommitmentScheme,
        },
        poly_iop::PolyIOP,
    };

    type E = Bls12_381;
    type FrType = Fr;

    #[test]
    fn test_hyperplonk_vela_e2e() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let pcs_srs = VelaPCS::<E>::gen_srs_for_testing(&mut rng, 12)?;

        for nv in [4usize, 5, 6] {
            let size = 1 << nv;
            let circuit = MockCircuit::<FrType>::new(size, &gates);
            assert!(circuit.is_satisfied(), "circuit not satisfied at nv={nv}");

            let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, VelaPCS<E>>>::preprocess(
                &circuit.index,
                &pcs_srs,
            )?;
            let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, VelaPCS<E>>>::prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
            )?;
            let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, VelaPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?;
            assert!(ok, "HyperPlonk+Vela verify failed at nv={nv}");
        }
        Ok(())
    }

    #[test]
    fn test_hyperplonk_vela_rejects_tampered_public_input() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let pcs_srs = VelaPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;
        let nv = 5;
        let size = 1 << nv;
        let circuit = MockCircuit::<FrType>::new(size, &gates);
        let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, VelaPCS<E>>>::preprocess(
            &circuit.index,
            &pcs_srs,
        )?;
        let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, VelaPCS<E>>>::prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        let mut bad_inputs = circuit.public_inputs.clone();
        if !bad_inputs.is_empty() {
            bad_inputs[0] += Fr::from(1u64);
            let res = <PolyIOP<FrType> as HyperPlonkSNARK<E, VelaPCS<E>>>::verify(
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

    // Cross-backend: the same circuit is provable and verifiable under both mKZG
    // and Vela, giving independent confidence in the Vela integration.
    #[test]
    fn test_hyperplonk_cross_backend_vela_vs_mkzg() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let nv = 5;
        let size = 1 << nv;
        let circuit = MockCircuit::<FrType>::new(size, &gates);
        assert!(circuit.is_satisfied());

        let srs_vela = VelaPCS::<E>::gen_srs_for_testing(&mut rng, 12)?;
        let (pk_r, vk_r) = <PolyIOP<FrType> as HyperPlonkSNARK<E, VelaPCS<E>>>::preprocess(
            &circuit.index,
            &srs_vela,
        )?;
        let proof_r = <PolyIOP<FrType> as HyperPlonkSNARK<E, VelaPCS<E>>>::prove(
            &pk_r,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        assert!(<PolyIOP<FrType> as HyperPlonkSNARK<E, VelaPCS<E>>>::verify(
            &vk_r,
            &circuit.public_inputs,
            &proof_r
        )?);

        let srs_m = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 12)?;
        let (pk_m, vk_m) =
            <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::preprocess(
                &circuit.index,
                &srs_m,
            )?;
        let proof_m = <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::prove(
            &pk_m,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        assert!(<PolyIOP<FrType> as HyperPlonkSNARK<
            E,
            MultilinearKzgPCS<E>,
        >>::verify(&vk_m, &circuit.public_inputs, &proof_m)?);
        Ok(())
    }
}
