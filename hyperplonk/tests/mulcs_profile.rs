//! HyperPlonk Mulcs PCS profiling test
//!
//! **#[ignore] — does not run in default cargo test.**
//!
//! Run with:
//!   MULCS_PROFILE=1 NV_RANGE=8,10,12 BACKEND=mulcs cargo test -p hyperplonk
//! --release --test mulcs_profile -- --ignored --nocapture   MULCS_PROFILE=1
//! NV_RANGE=8,10,12,14 BACKEND=both cargo test -p hyperplonk --release --test
//! mulcs_profile -- --ignored --nocapture
//!
//! Output: unified 9-column CSV to stdout.

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

    type E = Bls12_381;
    type FrType = Fr;

    const VALID_BACKENDS: &[&str] = &["mulcs", "mkzg", "both"];

    fn default_nv_range() -> Vec<usize> {
        let val = env::var("NV_RANGE").unwrap_or_else(|_| "8,10,12".to_string());
        let nvs: Vec<usize> = val
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        if nvs.is_empty() {
            panic!("NV_RANGE={val} parsed to empty list — provide comma-separated non-empty values like 8,10,12");
        }
        nvs
    }

    fn default_backend() -> String {
        let val = env::var("BACKEND").unwrap_or_else(|_| "mulcs".to_string());
        if !VALID_BACKENDS.contains(&val.as_str()) {
            panic!(
                "BACKEND={val} invalid — must be one of {:?}",
                VALID_BACKENDS
            );
        }
        val
    }

    fn default_repeat() -> usize {
        let val = env::var("REPEAT").unwrap_or_else(|_| "1".to_string());
        let r: usize = val.parse().unwrap_or(1);
        if r == 0 {
            println!("# WARNING: REPEAT=0, defaulting to 1");
            1
        } else {
            r
        }
    }

    fn print_csv_row(
        source: &str,
        backend: &str,
        nv: usize,
        n: usize,
        repeat: usize,
        phase: &str,
        ms: f64,
        count: usize,
        notes: &str,
    ) {
        println!("{source},{backend},{nv},{n},{repeat},{phase},{ms:.6},{count},{notes}");
    }

    fn bench_mulcs(nv: usize, repeat: usize) -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let n = 1 << nv;
        let gate = CustomizedGates::vanilla_plonk_gate();
        let circuit = MockCircuit::<FrType>::new(n, &gate);
        assert!(circuit.is_satisfied());

        for r in 0..repeat {
            let t0 = Instant::now();
            let srs = MulcsPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
            let srs_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row("top_level", "Mulcs", nv, n, r, "srs_gen", srs_ms, 1, "");

            let t0 = Instant::now();
            let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::preprocess(
                &circuit.index,
                &srs,
            )?;
            let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row("top_level", "Mulcs", nv, n, r, "preprocess", prep_ms, 1, "");

            let t0 = Instant::now();
            let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
            )?;
            let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row("top_level", "Mulcs", nv, n, r, "prove", prove_ms, 1, "");

            let t0 = Instant::now();
            let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?;
            let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row(
                "top_level",
                "Mulcs",
                nv,
                n,
                r,
                "verify",
                verify_ms,
                1,
                if ok { "pass" } else { "FAIL" },
            );
            assert!(ok, "Mulcs verify failed at nv={nv} r={r}");
        }
        Ok(())
    }

    fn bench_mkzg(nv: usize, repeat: usize) -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let n = 1 << nv;
        let gate = CustomizedGates::vanilla_plonk_gate();
        let circuit = MockCircuit::<FrType>::new(n, &gate);
        assert!(circuit.is_satisfied());

        for r in 0..repeat {
            let t0 = Instant::now();
            let srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
            let srs_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row("top_level", "mKZG", nv, n, r, "srs_gen", srs_ms, 1, "");

            let t0 = Instant::now();
            let (pk, vk) =
                <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::preprocess(
                    &circuit.index,
                    &srs,
                )?;
            let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row("top_level", "mKZG", nv, n, r, "preprocess", prep_ms, 1, "");

            let t0 = Instant::now();
            let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
            )?;
            let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row("top_level", "mKZG", nv, n, r, "prove", prove_ms, 1, "");

            let t0 = Instant::now();
            let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?;
            let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row(
                "top_level",
                "mKZG",
                nv,
                n,
                r,
                "verify",
                verify_ms,
                1,
                if ok { "pass" } else { "FAIL" },
            );
            assert!(ok, "mKZG verify failed at nv={nv} r={r}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "profile: run with MULCS_PROFILE=1 NV_RANGE=8,10,12 BACKEND=mulcs cargo test -p hyperplonk --release --test mulcs_profile -- --ignored --nocapture"]
    fn bench_mulcs_profile() -> Result<(), HyperPlonkErrors> {
        let nvs = default_nv_range();
        let backend = default_backend();
        let repeat = default_repeat();
        let threads = rayon::current_num_threads();

        println!(
            "# HyperPlonk PCS Profile — BLS12-381, vanilla Plonk, {} thread(s)",
            threads
        );
        println!("# NV_RANGE={:?} BACKEND={} REPEAT={}", nvs, backend, repeat);

        // Unified CSV header — printed once
        println!("source,backend,nv,N,repeat,phase,elapsed_ms,count,notes");

        for &nv in &nvs {
            let n = 1 << nv;
            if backend == "mulcs" || backend == "both" {
                println!("# --- Mulcs nv={nv} N={n} ---");
                bench_mulcs(nv, repeat)?;
            }
            if backend == "mkzg" || backend == "both" {
                println!("# --- mKZG nv={nv} N={n} ---");
                bench_mkzg(nv, repeat)?;
            }
        }
        Ok(())
    }
}
