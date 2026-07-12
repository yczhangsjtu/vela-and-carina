//! HyperPlonk PCS comparison benchmark.
//!
//! Backends: mKZG, Mulcs, ReciPCS, Zeromorph, Samaritan, Gemini,
//! NestedGridKZG. Gemini uses naive separate-KZG openings (NOT Shplonk).
//!
//! **These tests are `#[ignore]` and do not run in default `cargo test`.**
//!
//! Env vars:
//!   NV_RANGE=4,6        (default: 4,5,6)
//!   BACKEND=nrg          (default: all)
//!   BACKEND=all          (runs all backends)
//! Supported BACKEND values: mKZG, Mulcs, ReciPCS, Zeromorph,
//!                            Samaritan, Gemini, NestedGridKZG (nrg), all
//!
//! Examples:
//!   NV_RANGE=4 BACKEND=nrg cargo test -p hyperplonk --release \
//!     --test pcs_compare_bench -- --ignored --nocapture
//!
//!   NV_RANGE=4,6 BACKEND=all cargo test -p hyperplonk --release \
//!     --test pcs_compare_bench -- --ignored --nocapture

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
            prelude::{
                GeminiPCS, MulcsPCS, MultilinearKzgPCS, NestedGridKzgPCS, ReciPCS, SamaritanPCS,
                ZeromorphPCS,
            },
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

    const ALL_BACKENDS: &[&str] = &[
        "mKZG",
        "Mulcs",
        "ReciPCS",
        "Zeromorph",
        "Samaritan",
        "Gemini",
        "NestedGridKZG",
    ];

    fn selected_backends() -> Vec<&'static str> {
        let raw = match env::var("BACKEND") {
            Ok(v) => v,
            Err(_) => return ALL_BACKENDS.to_vec(),
        };
        let key = raw.trim().to_lowercase();
        match key.as_str() {
            "all" => ALL_BACKENDS.to_vec(),
            "mkzg" => vec!["mKZG"],
            "mulcs" => vec!["Mulcs"],
            "recipcs" | "mulcssymmetric" | "symmetric" | "mulcs-symmetric" => vec!["ReciPCS"],
            "zeromorph" => vec!["Zeromorph"],
            "samaritan" => vec!["Samaritan"],
            "gemini" => vec!["Gemini"],
            "nrg" | "nestedgrid" | "nested-grid-kzg" | "nested_grid_kzg" => vec!["NestedGridKZG"],
            _ => panic!(
                "unknown BACKEND '{raw}'. Supported: mKZG, Mulcs, ReciPCS, Zeromorph, Samaritan, Gemini, NestedGridKZG (nrg), all"
            ),
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
                assert!(ok);
                println!("top_level,{backend},{nv},{size},0,srs_gen,{srs_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,preprocess,{prep_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,prove,{prove_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,verify,{verify_ms:.6},1,pass");
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
                assert!(ok);
                println!("top_level,{backend},{nv},{size},0,srs_gen,{srs_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,preprocess,{prep_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,prove,{prove_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,verify,{verify_ms:.6},1,pass");
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
                assert!(ok);
                println!("top_level,{backend},{nv},{size},0,srs_gen,{srs_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,preprocess,{prep_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,prove,{prove_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,verify,{verify_ms:.6},1,pass");
            },
            "ReciPCS" => {
                let t0 = Instant::now();
                let srs = ReciPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
                let srs_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, ReciPCS<E>>>::preprocess(
                    &circuit.index,
                    &srs,
                )?;
                let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, ReciPCS<E>>>::prove(
                    &pk,
                    &circuit.public_inputs,
                    &circuit.witnesses,
                )?;
                let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, ReciPCS<E>>>::verify(
                    &vk,
                    &circuit.public_inputs,
                    &proof,
                )?;
                let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;
                assert!(ok);
                println!("top_level,{backend},{nv},{size},0,srs_gen,{srs_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,preprocess,{prep_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,prove,{prove_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,verify,{verify_ms:.6},1,pass");
            },
            "Samaritan" => {
                let t0 = Instant::now();
                let srs = SamaritanPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
                let srs_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let (pk, vk) =
                    <PolyIOP<FrType> as HyperPlonkSNARK<E, SamaritanPCS<E>>>::preprocess(
                        &circuit.index,
                        &srs,
                    )?;
                let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, SamaritanPCS<E>>>::prove(
                    &pk,
                    &circuit.public_inputs,
                    &circuit.witnesses,
                )?;
                let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, SamaritanPCS<E>>>::verify(
                    &vk,
                    &circuit.public_inputs,
                    &proof,
                )?;
                let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;
                assert!(ok);
                println!("top_level,{backend},{nv},{size},0,srs_gen,{srs_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,preprocess,{prep_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,prove,{prove_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,verify,{verify_ms:.6},1,pass");
            },
            "Gemini" => {
                let t0 = Instant::now();
                let srs = GeminiPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
                let srs_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, GeminiPCS<E>>>::preprocess(
                    &circuit.index,
                    &srs,
                )?;
                let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, GeminiPCS<E>>>::prove(
                    &pk,
                    &circuit.public_inputs,
                    &circuit.witnesses,
                )?;
                let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, GeminiPCS<E>>>::verify(
                    &vk,
                    &circuit.public_inputs,
                    &proof,
                )?;
                let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;
                assert!(ok);
                println!("top_level,{backend},{nv},{size},0,srs_gen,{srs_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,preprocess,{prep_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,prove,{prove_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,verify,{verify_ms:.6},1,pass");
            },
            "NestedGridKZG" => {
                let t0 = Instant::now();
                let srs = NestedGridKzgPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
                let srs_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let (pk, vk) =
                    <PolyIOP<FrType> as HyperPlonkSNARK<E, NestedGridKzgPCS<E>>>::preprocess(
                        &circuit.index,
                        &srs,
                    )?;
                let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, NestedGridKzgPCS<E>>>::prove(
                    &pk,
                    &circuit.public_inputs,
                    &circuit.witnesses,
                )?;
                let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = Instant::now();
                let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, NestedGridKzgPCS<E>>>::verify(
                    &vk,
                    &circuit.public_inputs,
                    &proof,
                )?;
                let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;
                assert!(ok);
                println!("top_level,{backend},{nv},{size},0,srs_gen,{srs_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,preprocess,{prep_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,prove,{prove_ms:.6},1,");
                println!("top_level,{backend},{nv},{size},0,verify,{verify_ms:.6},1,pass");
            },
            _ => panic!("unknown backend: {backend}"),
        }
        Ok(())
    }

    #[test]
    #[ignore = "benchmark: set NV_RANGE and BACKEND, then run with --release -- --ignored --nocapture"]
    fn bench_hyperplonk_pcs_compare() -> Result<(), HyperPlonkErrors> {
        let nvs = default_nv_range();
        let backends = selected_backends();
        println!("source,backend,nv,N,repeat,phase,elapsed_ms,count,notes");
        for &nv in &nvs {
            for b in &backends {
                bench_backend(b, nv)?;
            }
        }
        Ok(())
    }
}
