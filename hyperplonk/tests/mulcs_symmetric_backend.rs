//! HyperPlonk + MulcsSymmetricPCS backend correctness tests.
//!
//! Run with: cargo test -p hyperplonk mulcs_symmetric_backend -- --nocapture

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
            prelude::{MulcsPCS, MulcsSymmetricPCS, MultilinearKzgPCS},
            PolynomialCommitmentScheme,
        },
        poly_iop::PolyIOP,
    };

    type E = Bls12_381;
    type FrType = Fr;

    #[test]
    fn test_hyperplonk_mulcs_symmetric_e2e() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();

        let pcs_srs = MulcsSymmetricPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;

        for nv in [4usize, 5, 6] {
            let size = 1 << nv;
            let circuit = MockCircuit::<FrType>::new(size, &gates);
            assert!(circuit.is_satisfied(), "circuit not satisfied at nv={nv}");

            let (pk, vk) =
                <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsSymmetricPCS<E>>>::preprocess(
                    &circuit.index,
                    &pcs_srs,
                )?;

            let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsSymmetricPCS<E>>>::prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
            )?;

            let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsSymmetricPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?;

            assert!(ok, "HyperPlonk+MulcsSymmetric verify failed at nv={nv}");
        }

        Ok(())
    }

    #[test]
    fn test_hyperplonk_cross_backend_mulcs_symmetric() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let nv = 5;
        let size = 1 << nv;

        let mkzg_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;
        let claymore_srs = MulcsPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;
        let sym_srs = MulcsSymmetricPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;

        let circuit = MockCircuit::<FrType>::new(size, &gates);
        assert!(circuit.is_satisfied());

        // mKZG
        let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::preprocess(
            &circuit.index,
            &mkzg_srs,
        )?;
        let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        assert!(
            <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof
            )?,
            "mKZG backend verify failed"
        );

        // Mulcs Claymore
        let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::preprocess(
            &circuit.index,
            &claymore_srs,
        )?;
        let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        assert!(
            <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?,
            "Mulcs Claymore backend verify failed"
        );

        // Mulcs Symmetric
        let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsSymmetricPCS<E>>>::preprocess(
            &circuit.index,
            &sym_srs,
        )?;
        let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsSymmetricPCS<E>>>::prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        assert!(
            <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsSymmetricPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?,
            "Mulcs Symmetric backend verify failed"
        );

        Ok(())
    }
}
