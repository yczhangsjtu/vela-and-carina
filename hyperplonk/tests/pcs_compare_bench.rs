//! HyperPlonk PCS comparison benchmark: mKZG vs Mulcs
//!
//! **These tests are `#[ignore]` and do not run in default `cargo test`.**
//!
//! Run with:
//!   cargo test -p hyperplonk --release bench_hyperplonk_pcs_compare --
//! --ignored --nocapture
//!
//! Parameters: vanilla Plonk gate, BLS12-381, nv = 4,5,6 by default (small for
//! quick iteration). Set the NV_RANGE env var to override: e.g. NV_RANGE=6,8,10

#[cfg(test)]
mod tests {
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_std::test_rng;
    use hyperplonk::{
        prelude::{CustomizedGates, HyperPlonkErrors, MockCircuit},
        HyperPlonkSNARK,
    };
    use std::{env, time::Instant};
    use subroutines::{
        pcs::{
            prelude::{MulcsPCS, MultilinearKzgPCS},
            PolynomialCommitmentScheme,
        },
        poly_iop::PolyIOP,
    };

    fn default_nv_range() -> Vec<usize> {
        if let Ok(val) = env::var("NV_RANGE") {
            val.split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect()
        } else {
            vec![4, 5, 6]
        }
    }

    fn bench_backend<B: AsRef<str>>(backend: B, nv: usize) -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let size = 1 << nv;
        let gate = CustomizedGates::vanilla_plonk_gate();
        let circuit = MockCircuit::<Fr>::new(size, &gate);
        assert!(circuit.is_satisfied());
        let name = backend.as_ref();

        if name == "mKZG" {
            let t0 = Instant::now();
            let srs = MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, nv)?;
            let srs_time = t0.elapsed();

            let t0 = Instant::now();
            let (pk, vk) = <PolyIOP<Fr> as HyperPlonkSNARK<
                Bls12_381,
                MultilinearKzgPCS<Bls12_381>,
            >>::preprocess(&circuit.index, &srs)?;
            let prep_time = t0.elapsed();

            let t0 = Instant::now();
            let proof = <PolyIOP<Fr> as HyperPlonkSNARK<
                Bls12_381,
                MultilinearKzgPCS<Bls12_381>,
            >>::prove(&pk, &circuit.public_inputs, &circuit.witnesses)?;
            let prove_time = t0.elapsed();

            let t0 = Instant::now();
            let ok =
                <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::verify(
                    &vk,
                    &circuit.public_inputs,
                    &proof,
                )?;
            let verify_time = t0.elapsed();

            println!(
                "{:>6} │ {:>3} │ {:>6} │ {:>8.2?} │ {:>10.2?} │ {:>10.2?} │ {:>10.2?} │ {:>5} │ unavailable",
                name, nv, size, srs_time, prep_time, prove_time, verify_time, ok
            );
        } else if name == "Mulcs" {
            let t0 = Instant::now();
            let srs = MulcsPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, nv)?;
            let srs_time = t0.elapsed();

            let t0 = Instant::now();
            let (pk, vk) =
                <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, MulcsPCS<Bls12_381>>>::preprocess(
                    &circuit.index,
                    &srs,
                )?;
            let prep_time = t0.elapsed();

            let t0 = Instant::now();
            let proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, MulcsPCS<Bls12_381>>>::prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
            )?;
            let prove_time = t0.elapsed();

            let t0 = Instant::now();
            let ok = <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, MulcsPCS<Bls12_381>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?;
            let verify_time = t0.elapsed();

            println!(
                "{:>6} │ {:>3} │ {:>6} │ {:>8.2?} │ {:>10.2?} │ {:>10.2?} │ {:>10.2?} │ {:>5} │ unavailable",
                name, nv, size, srs_time, prep_time, prove_time, verify_time, ok
            );
        }

        Ok(())
    }

    #[test]
    #[ignore = "benchmark: run with cargo test -p hyperplonk --release bench_hyperplonk_pcs_compare -- --ignored --nocapture"]
    fn bench_hyperplonk_pcs_compare() -> Result<(), HyperPlonkErrors> {
        let nvs = default_nv_range();
        let threads = rayon::current_num_threads();
        println!(
            "\n╔════════════════════════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║  HyperPlonk PCS Comparison — BLS12-381, vanilla Plonk, {} thread(s)               ║",
            threads
        );
        println!(
            "╠════════╤═════╤════════╤══════════╤════════════╤════════════╤════════════╤═══════╤═══════════╣"
        );
        println!(
            "║ Backend│ nv  │ N      │ SRS gen  │ Preprocess │ Prove      │ Verify     │ Pass  │ Proof size║"
        );
        println!(
            "╠════════╪═════╪════════╪══════════╪════════════╪════════════╪════════════╪═══════╪═══════════╣"
        );

        for &nv in &nvs {
            bench_backend("mKZG", nv)?;
            bench_backend("Mulcs", nv)?;
            if nv != *nvs.last().unwrap() {
                println!(
                    "╟────────┼─────┼────────┼──────────┼────────────┼────────────┼────────────┼───────┼───────────╢"
                );
            }
        }

        println!(
            "╚════════╧═════╧════════╧══════════╧════════════╧════════════╧════════════╧═══════╧═══════════╝"
        );
        Ok(())
    }
}
