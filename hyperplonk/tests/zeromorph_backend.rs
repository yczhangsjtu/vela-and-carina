//! HyperPlonk + ZeromorphPCS backend correctness tests.

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
            prelude::{MulcsPCS, MultilinearKzgPCS, ZeromorphPCS},
            PolynomialCommitmentScheme,
        },
        poly_iop::PolyIOP,
    };

    type E = Bls12_381;
    type FrType = Fr;

    #[test]
    fn test_hyperplonk_zeromorph_e2e() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let pcs_srs = ZeromorphPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;

        for nv in [4usize, 5, 6] {
            let size = 1 << nv;
            let circuit = MockCircuit::<FrType>::new(size, &gates);
            assert!(circuit.is_satisfied(), "circuit not satisfied at nv={nv}");

            let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, ZeromorphPCS<E>>>::preprocess(
                &circuit.index,
                &pcs_srs,
            )?;
            let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, ZeromorphPCS<E>>>::prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
            )?;
            let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, ZeromorphPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?;
            assert!(ok, "HyperPlonk+Zeromorph verify failed at nv={nv}");
        }
        Ok(())
    }

    #[test]
    fn test_hyperplonk_cross_backend() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let nv = 5;
        let size = 1 << nv;

        let mkzg_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;
        let mulcs_srs = MulcsPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;
        let zm_srs = ZeromorphPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;

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
        assert!(<PolyIOP<FrType> as HyperPlonkSNARK<
            E,
            MultilinearKzgPCS<E>,
        >>::verify(&vk, &circuit.public_inputs, &proof,)?);

        // Mulcs
        let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::preprocess(
            &circuit.index,
            &mulcs_srs,
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
            )?
        );

        // Zeromorph
        let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, ZeromorphPCS<E>>>::preprocess(
            &circuit.index,
            &zm_srs,
        )?;
        let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, ZeromorphPCS<E>>>::prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        assert!(
            <PolyIOP<FrType> as HyperPlonkSNARK<E, ZeromorphPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?
        );

        Ok(())
    }
}
