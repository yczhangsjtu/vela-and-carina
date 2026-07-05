//! HyperPlonk + MulcsPCS backend correctness tests.
//!
//! Tests small-parameter mock circuits with both mKZG and Mulcs backends
//! to verify correctness and cross-backend consistency.
//!
//! Run with: cargo test -p hyperplonk mulcs_backend -- --nocapture

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
            prelude::{MulcsPCS, MultilinearKzgPCS},
            PolynomialCommitmentScheme,
        },
        poly_iop::PolyIOP,
    };

    type E = Bls12_381;
    type FrType = Fr;

    #[test]
    fn test_hyperplonk_mulcs_e2e() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let gates = CustomizedGates::vanilla_plonk_gate();

        let pcs_srs = MulcsPCS::<E>::gen_srs_for_testing(&mut rng, 10)?;

        for nv in [4usize, 5, 6] {
            let size = 1 << nv;
            let circuit = MockCircuit::<FrType>::new(size, &gates);
            assert!(circuit.is_satisfied(), "circuit not satisfied at nv={nv}");

            let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::preprocess(
                &circuit.index,
                &pcs_srs,
            )?;

            let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
            )?;

            let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?;

            assert!(ok, "HyperPlonk+Mulcs verify failed at nv={nv}");
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

        let circuit = MockCircuit::<FrType>::new(size, &gates);
        assert!(circuit.is_satisfied());

        let (mkzg_pk, mkzg_vk) =
            <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::preprocess(
                &circuit.index,
                &mkzg_srs,
            )?;
        let mkzg_proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::prove(
            &mkzg_pk,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        assert!(
            <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::verify(
                &mkzg_vk,
                &circuit.public_inputs,
                &mkzg_proof,
            )?,
            "mKZG backend verify failed"
        );

        let (mulcs_pk, mulcs_vk) =
            <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::preprocess(
                &circuit.index,
                &mulcs_srs,
            )?;
        let mulcs_proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::prove(
            &mulcs_pk,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
        assert!(
            <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::verify(
                &mulcs_vk,
                &circuit.public_inputs,
                &mulcs_proof,
            )?,
            "Mulcs backend verify failed"
        );

        Ok(())
    }
}
