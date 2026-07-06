//! HyperPlonk PCS comparison benchmark: mKZG vs Mulcs vs Zeromorph
//!
//! **These tests are `#[ignore]` and do not run in default `cargo test`.**
//!
//! Run with:
//!   NV_RANGE=4,6 cargo test -p hyperplonk --release
//! bench_hyperplonk_pcs_compare -- --ignored --nocapture   NV_RANGE=8,10,12
//! cargo test -p hyperplonk --release bench_hyperplonk_pcs_compare -- --ignored
//! --nocapture
//!
//! Parameters: vanilla Plonk gate, BLS12-381. Set NV_RANGE env var to override.

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
            prelude::{MulcsPCS, MultilinearKzgPCS, ZeromorphPCS},
            PolynomialCommitmentScheme,
        },
        poly_iop::PolyIOP,
    };

    type E = Bls12_381;
    type FrType = Fr;

    fn default_nv_range() -> Vec<usize> {
        if let Ok(val) = env::var("NV_RANGE") {
            val.split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect()
        } else {
            vec![4, 5, 6]
        }
    }

    fn bench_backend(backend: &str, nv: usize) -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let size = 1 << nv;
        let gate = CustomizedGates::vanilla_plonk_gate();
        let circuit = MockCircuit::<FrType>::new(size, &gate);
        assert!(circuit.is_satisfied());

        match backend {
            "mKZG" => {
                let t0 = Instant::now();
                let srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
                let srs_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let (pk, vk) =
                    <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::preprocess(
                        &circuit.index,
                        &srs,
                    )?;
                let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::prove(
                    &pk,
                    &circuit.public_inputs,
                    &circuit.witnesses,
                )?;
                let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::verify(
                    &vk,
                    &circuit.public_inputs,
                    &proof,
                )?;
                let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;

                println!("top_level,{backend},{nv},{size},0,srs_gen,{srs_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,preprocess,{prep_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,prove,{prove_ms:.6},1,");
                println!(
                    "top_level,{backend},{nv},{size},0,verify,{verify_ms:.6},1,{}",
                    if ok { "pass" } else { "FAIL" }
                );
                assert!(ok);
            },
            "Mulcs" => {
                let t0 = Instant::now();
                let srs = MulcsPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
                let srs_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::preprocess(
                    &circuit.index,
                    &srs,
                )?;
                let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::prove(
                    &pk,
                    &circuit.public_inputs,
                    &circuit.witnesses,
                )?;
                let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::verify(
                    &vk,
                    &circuit.public_inputs,
                    &proof,
                )?;
                let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;

                println!("top_level,{backend},{nv},{size},0,srs_gen,{srs_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,preprocess,{prep_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,prove,{prove_ms:.6},1,");
                println!(
                    "top_level,{backend},{nv},{size},0,verify,{verify_ms:.6},1,{}",
                    if ok { "pass" } else { "FAIL" }
                );
                assert!(ok);
            },
            "Zeromorph" => {
                let t0 = Instant::now();
                let srs = ZeromorphPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
                let srs_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let (pk, vk) =
                    <PolyIOP<FrType> as HyperPlonkSNARK<E, ZeromorphPCS<E>>>::preprocess(
                        &circuit.index,
                        &srs,
                    )?;
                let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, ZeromorphPCS<E>>>::prove(
                    &pk,
                    &circuit.public_inputs,
                    &circuit.witnesses,
                )?;
                let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, ZeromorphPCS<E>>>::verify(
                    &vk,
                    &circuit.public_inputs,
                    &proof,
                )?;
                let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;

                println!("top_level,{backend},{nv},{size},0,srs_gen,{srs_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,preprocess,{prep_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,prove,{prove_ms:.6},1,");
                println!(
                    "top_level,{backend},{nv},{size},0,verify,{verify_ms:.6},1,{}",
                    if ok { "pass" } else { "FAIL" }
                );
                assert!(ok);
            },
            _ => panic!("unknown backend: {backend}"),
        }
        Ok(())
    }

    #[test]
    #[ignore = "benchmark: run with NV_RANGE=4,6 cargo test -p hyperplonk --release bench_hyperplonk_pcs_compare -- --ignored --nocapture"]
    fn bench_hyperplonk_pcs_compare() -> Result<(), HyperPlonkErrors> {
        let nvs = default_nv_range();
        println!("source,backend,nv,N,repeat,phase,elapsed_ms,count,notes");
        for &nv in &nvs {
            bench_backend("mKZG", nv)?;
            bench_backend("Mulcs", nv)?;
            bench_backend("Zeromorph", nv)?;
        }
        Ok(())
    }
}
