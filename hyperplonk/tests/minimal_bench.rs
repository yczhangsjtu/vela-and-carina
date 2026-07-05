//! Minimal HyperPlonk benchmark — vanilla Plonk gate, single repetition.
//! Run with: cargo test --release hp_vanilla -- --nocapture

#[cfg(test)]
mod tests {
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_std::test_rng;
    use hyperplonk::{
        prelude::{CustomizedGates, HyperPlonkErrors, MockCircuit},
        HyperPlonkSNARK,
    };
    use std::time::Instant;
    use subroutines::{
        pcs::{prelude::MultilinearKzgPCS, PolynomialCommitmentScheme},
        poly_iop::PolyIOP,
    };

    #[test]
    fn hp_vanilla() -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let srs = MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, 16)?;
        let gate = CustomizedGates::vanilla_plonk_gate();

        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║  HyperPlonk Vanilla Plonk (BLS12-381, 1 thread)   ║");
        println!("╠════╤═══════╤══════════╤══════════╣");
        println!("║  μ │     N │ Prove (s) │ Vrfy(ms) ║");
        println!("╠════╪═══════╪══════════╪══════════╣");

        for nv in 8..=16 {
            let n = 1 << nv;
            let circuit = MockCircuit::<Fr>::new(n, &gate);
            assert!(circuit.is_satisfied());
            let index = circuit.index;

            let (pk, vk) = <PolyIOP<Fr> as HyperPlonkSNARK<
                Bls12_381,
                MultilinearKzgPCS<Bls12_381>,
            >>::preprocess(&index, &srs)?;

            let t0 = Instant::now();
            let proof = <PolyIOP<Fr> as HyperPlonkSNARK<
                Bls12_381,
                MultilinearKzgPCS<Bls12_381>,
            >>::prove(&pk, &circuit.public_inputs, &circuit.witnesses)?;
            let t_prove = t0.elapsed();

            let t0 = Instant::now();
            let ok =
                <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::verify(
                    &vk,
                    &circuit.public_inputs,
                    &proof,
                )?;
            let t_verify = t0.elapsed();
            assert!(ok);

            println!(
                "║ {:>2} │ {:>5} │ {:>8.3} │ {:>8.1} │               ║",
                nv,
                n,
                t_prove.as_secs_f64(),
                t_verify.as_secs_f64() * 1000.0
            );
        }
        println!("╚════╧═══════╧══════════╧══════════╧═══════════════╝");
        Ok(())
    }
}
