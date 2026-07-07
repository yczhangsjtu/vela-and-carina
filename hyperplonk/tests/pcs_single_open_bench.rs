//! Single-opening PCS benchmark: mKZG vs Mulcs vs Zeromorph.
//!
//! This benchmark measures only the PCS API path:
//! setup = gen_srs_for_testing + trim, then commit, open, verify.
//! It does not include HyperPlonk preprocessing, proving, or batch opening.
//!
//! Run examples:
//!   NV_RANGE=8,10,12 BACKEND=all REPEAT=1 cargo test -p hyperplonk --release \
//!     bench_pcs_single_open -- --ignored --nocapture
//!   NV_RANGE=8,9,10,11,12,13,14,15,16,17,18,19,20 cargo test -p hyperplonk \
//!     --release bench_pcs_single_open -- --ignored --nocapture

#[cfg(test)]
mod tests {
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_poly::DenseMultilinearExtension;
    use ark_std::{rand::Rng, test_rng, UniformRand};
    use std::{env, sync::Arc, time::Instant};
    use subroutines::pcs::{
        prelude::{MulcsPCS, MultilinearKzgPCS, PCSError, ZeromorphPCS},
        PolynomialCommitmentScheme,
    };

    type E = Bls12_381;

    fn parse_nv_range() -> Vec<usize> {
        if let Ok(raw) = env::var("NV_RANGE") {
            let nvs = raw
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| {
                    s.parse::<usize>()
                        .unwrap_or_else(|_| panic!("invalid NV_RANGE entry: {s}"))
                })
                .collect::<Vec<_>>();
            assert!(!nvs.is_empty(), "NV_RANGE must not be empty");
            nvs
        } else {
            (8..=20).collect()
        }
    }

    fn parse_repeat() -> usize {
        env::var("REPEAT")
            .ok()
            .map(|raw| {
                raw.parse::<usize>()
                    .unwrap_or_else(|_| panic!("invalid REPEAT: {raw}"))
            })
            .filter(|&repeat| repeat > 0)
            .unwrap_or(1)
    }

    fn selected_backends() -> Vec<&'static str> {
        match env::var("BACKEND")
            .unwrap_or_else(|_| "all".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "all" => vec!["mKZG", "Mulcs", "Zeromorph"],
            "mkzg" | "multilinear_kzg" => vec!["mKZG"],
            "mulcs" => vec!["Mulcs"],
            "zeromorph" | "zm" => vec!["Zeromorph"],
            other => panic!("unknown BACKEND: {other}"),
        }
    }

    fn random_poly_and_point<R: Rng>(
        rng: &mut R,
        nv: usize,
    ) -> (Arc<DenseMultilinearExtension<Fr>>, Vec<Fr>) {
        let n = 1 << nv;
        let evals = (0..n).map(|_| Fr::rand(rng)).collect();
        let point = (0..nv).map(|_| Fr::rand(rng)).collect();
        (
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, evals)),
            point,
        )
    }

    fn time_ms<T, F>(f: F) -> Result<(T, f64), PCSError>
    where
        F: FnOnce() -> Result<T, PCSError>,
    {
        let start = Instant::now();
        let value = f()?;
        Ok((value, start.elapsed().as_secs_f64() * 1000.0))
    }

    fn bench_one<PCS, R>(
        backend: &str,
        nv: usize,
        repeat: usize,
        rng: &mut R,
        poly: &Arc<DenseMultilinearExtension<Fr>>,
        point: &Vec<Fr>,
    ) -> Result<(), PCSError>
    where
        PCS: PolynomialCommitmentScheme<
            E,
            Polynomial = Arc<DenseMultilinearExtension<Fr>>,
            Point = Vec<Fr>,
            Evaluation = Fr,
        >,
        R: Rng,
    {
        let n = 1 << nv;

        let ((pp, vp), setup_ms) = time_ms(|| {
            let srs = PCS::gen_srs_for_testing(rng, nv)?;
            PCS::trim(&srs, None, Some(nv))
        })?;

        let (commitment, commit_ms) = time_ms(|| PCS::commit(&pp, poly))?;
        let ((proof, value), prover_ms) = time_ms(|| PCS::open(&pp, poly, point))?;
        let (ok, verifier_ms) = time_ms(|| PCS::verify(&vp, &commitment, point, &value, &proof))?;

        assert!(ok, "{backend} verify failed at nv={nv}, repeat={repeat}");

        println!("single_open,{backend},{nv},{n},{repeat},setup,{setup_ms:.6},1,gen_srs_plus_trim");
        println!("single_open,{backend},{nv},{n},{repeat},commit,{commit_ms:.6},1,");
        println!("single_open,{backend},{nv},{n},{repeat},prover,{prover_ms:.6},1,open");
        println!("single_open,{backend},{nv},{n},{repeat},verifier,{verifier_ms:.6},1,verify");
        Ok(())
    }

    #[test]
    #[ignore = "benchmark: run with cargo test -p hyperplonk --release bench_pcs_single_open -- --ignored --nocapture"]
    fn bench_pcs_single_open() -> Result<(), PCSError> {
        let nvs = parse_nv_range();
        let repeats = parse_repeat();
        let backends = selected_backends();
        let mut rng = test_rng();

        println!("source,backend,nv,N,repeat,phase,elapsed_ms,count,notes");
        for &nv in &nvs {
            for repeat in 0..repeats {
                let (poly, point) = random_poly_and_point(&mut rng, nv);
                for &backend in &backends {
                    match backend {
                        "mKZG" => bench_one::<MultilinearKzgPCS<E>, _>(
                            backend, nv, repeat, &mut rng, &poly, &point,
                        )?,
                        "Mulcs" => bench_one::<MulcsPCS<E>, _>(
                            backend, nv, repeat, &mut rng, &poly, &point,
                        )?,
                        "Zeromorph" => bench_one::<ZeromorphPCS<E>, _>(
                            backend, nv, repeat, &mut rng, &poly, &point,
                        )?,
                        _ => unreachable!(),
                    }
                }
            }
        }
        Ok(())
    }
}
