//! HyperPlonk Mulcs PCS profiling test
//!
//! **#[ignore] — does not run in default cargo test.**
//!
//! Run with:
//!   MULCS_PROFILE=1 NV_RANGE=8,10,12 BACKEND=mulcs cargo test -p hyperplonk
//! --release --test mulcs_profile -- --ignored --nocapture   NV_RANGE=8,10,12,
//! 14,16 BACKEND=both cargo test -p hyperplonk --release --test mulcs_profile
//! -- --ignored --nocapture
//!
//! Output: CSV lines to stderr (when MULCS_PROFILE=1), plus summary table to
//! stdout.

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

    fn default_nv_range() -> Vec<usize> {
        if let Ok(val) = env::var("NV_RANGE") {
            val.split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect()
        } else {
            vec![8, 10, 12]
        }
    }

    fn default_backend() -> String {
        env::var("BACKEND").unwrap_or_else(|_| "mulcs".to_string())
    }

    fn default_repeat() -> usize {
        env::var("REPEAT")
            .unwrap_or_else(|_| "1".to_string())
            .parse()
            .unwrap_or(1)
    }

    fn print_csv_row(
        backend: &str,
        nv: usize,
        n: usize,
        repeat: usize,
        phase: &str,
        ms: f64,
        notes: &str,
    ) {
        println!("csv,{backend},{nv},{n},{repeat},{phase},{ms:.6},{notes}");
    }

    fn bench_mulcs(nv: usize, repeat: usize) -> Result<(), HyperPlonkErrors> {
        let mut rng = test_rng();
        let n = 1 << nv;
        let gate = CustomizedGates::vanilla_plonk_gate();
        let circuit = MockCircuit::<FrType>::new(n, &gate);
        assert!(circuit.is_satisfied());

        for r in 0..repeat {
            // SRS gen
            let t0 = Instant::now();
            let srs = MulcsPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
            let srs_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row("Mulcs", nv, n, r, "srs_gen", srs_ms, "");

            // Preprocess
            let t0 = Instant::now();
            let (pk, vk) = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::preprocess(
                &circuit.index,
                &srs,
            )?;
            let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row("Mulcs", nv, n, r, "preprocess", prep_ms, "");

            // Prove
            let t0 = Instant::now();
            let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
            )?;
            let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row("Mulcs", nv, n, r, "prove", prove_ms, "");

            // Verify
            let t0 = Instant::now();
            let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, MulcsPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?;
            let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row(
                "Mulcs",
                nv,
                n,
                r,
                "verify",
                verify_ms,
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
            print_csv_row("mKZG", nv, n, r, "srs_gen", srs_ms, "");

            let t0 = Instant::now();
            let (pk, vk) =
                <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::preprocess(
                    &circuit.index,
                    &srs,
                )?;
            let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row("mKZG", nv, n, r, "preprocess", prep_ms, "");

            let t0 = Instant::now();
            let proof = <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
            )?;
            let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row("mKZG", nv, n, r, "prove", prove_ms, "");

            let t0 = Instant::now();
            let ok = <PolyIOP<FrType> as HyperPlonkSNARK<E, MultilinearKzgPCS<E>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?;
            let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;
            print_csv_row(
                "mKZG",
                nv,
                n,
                r,
                "verify",
                verify_ms,
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

        eprintln!(
            "# HyperPlonk PCS Profile — BLS12-381, vanilla Plonk, {} thread(s)",
            threads
        );
        eprintln!("# NV_RANGE={:?} BACKEND={} REPEAT={}", nvs, backend, repeat);
        println!(
            "# HyperPlonk PCS Profile — threads={}, nvs={:?}",
            threads, nvs
        );
        println!("backend,nv,N,repeat,phase,elapsed_ms,notes");

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
