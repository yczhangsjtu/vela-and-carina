//! HyperPlonk + GeminiPCS backend correctness tests.
//!
//! Run with: cargo test -p hyperplonk gemini_backend -- --nocapture

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
            prelude::{GeminiPCS, MultilinearKzgPCS, ReciPCS, SamaritanPCS, ZeromorphPCS},
            PolynomialCommitmentScheme,
        },
        poly_iop::PolyIOP,
    };

    type E = Bls12_381;
    type FrType = Fr;

    #[test]
    fn test_hyperplonk_gemini_e2e() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let pcs_srs = GeminiPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;

        for nv in [4usize, 5, 6] {
            let size = 1 << nv;
            let circuit = MockCircuit::<FrType>::new(size, &gates);
            assert!(circuit.is_satisfied(), "circuit not satisfied at nv={nv}");

            let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, GeminiPCS<E>>>::preprocess(
                &circuit.index,
                &pcs_srs,
            )?;
            let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, GeminiPCS<E>>>::prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
            )?;
            let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, GeminiPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?;
            assert!(ok, "HyperPlonk+Gemini verify failed at nv={nv}");
        }
        Ok(())
    }

    #[test]
    fn test_hyperplonk_cross_backend_gemini() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();
        let nv = 5;
        let size = 1 << nv;
        let mkzg_srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;
        let reci_srs = ReciPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;
        let zm_srs = ZeromorphPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;
        let sam_srs = SamaritanPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;
        let gem_srs = GeminiPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;

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
        >>::verify(&vk, &circuit.public_inputs, &proof)?);

        // ReciPCS (the canonical symmetric reciprocal construction)
        let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, ReciPCS<E>>>::preprocess(
            &circuit.index,
            &reci_srs,
        )?;
        let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, ReciPCS<E>>>::prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        assert!(<PolyIOP<FrType> as HyperPlonkSNARK<E, ReciPCS<E>>>::verify(
            &vk,
            &circuit.public_inputs,
            &proof
        )?);

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
                &proof
            )?
        );

        // Samaritan
        let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, SamaritanPCS<E>>>::preprocess(
            &circuit.index,
            &sam_srs,
        )?;
        let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, SamaritanPCS<E>>>::prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        assert!(
            <PolyIOP<FrType> as HyperPlonkSNARK<E, SamaritanPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof
            )?
        );

        // Gemini
        let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, GeminiPCS<E>>>::preprocess(
            &circuit.index,
            &gem_srs,
        )?;
        let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, GeminiPCS<E>>>::prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        assert!(
            <PolyIOP<FrType> as HyperPlonkSNARK<E, GeminiPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof
            )?
        );

        Ok(())
    }
}
