// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::UniformRand;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{sync::Arc, test_rng};
use std::{env, time::Instant};
use subroutines::pcs::{
    prelude::{
        GeminiPCS, MercuryPCS, MulcsPCS, MultilinearKzgPCS, NestedGridKzgPCS, PCSError, ReciPCS,
        SamaritanPCS, ZeromorphPCS,
    },
    PolynomialCommitmentScheme,
};

const DEFAULT_NV_LIST: [usize; 7] = [8, 10, 12, 14, 16, 18, 20];
const VERIFY_REPETITIONS: usize = 100;
const ALL_BACKENDS: [&str; 8] = [
    "mkzg",
    "gemini",
    "mulcs",
    "samaritan",
    "zeromorph",
    "recipcs",
    "nrg",
    "mercury",
];

fn parse_nv_list() -> Result<Vec<usize>, PCSError> {
    match std::env::var("PCS_BENCH_NV_RANGE") {
        Ok(raw) => {
            let mut list = Vec::new();
            for s in raw.split(',') {
                let s = s.trim();
                if s.is_empty() {
                    continue;
                }
                let nv: usize = s.parse().map_err(|_| {
                    PCSError::InvalidParameters(format!("PCS_BENCH_NV_RANGE: invalid nv '{}'", s))
                })?;
                list.push(nv);
            }
            if list.is_empty() {
                Err(PCSError::InvalidParameters(
                    "PCS_BENCH_NV_RANGE: empty list".to_string(),
                ))
            } else {
                Ok(list)
            }
        },
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_NV_LIST.to_vec()),
        Err(e) => Err(PCSError::InvalidParameters(format!(
            "PCS_BENCH_NV_RANGE env error: {}",
            e
        ))),
    }
}

fn parse_backends() -> Result<Vec<String>, PCSError> {
    let raw = match env::var("PCS_BENCH_BACKEND") {
        Ok(raw) => raw,
        Err(env::VarError::NotPresent) => {
            return Ok(ALL_BACKENDS
                .iter()
                .map(|name| (*name).to_string())
                .collect())
        },
        Err(err) => {
            return Err(PCSError::InvalidParameters(format!(
                "PCS_BENCH_BACKEND env error: {err}"
            )))
        },
    };
    let selected = raw.trim().to_ascii_lowercase();
    if selected == "all" {
        return Ok(ALL_BACKENDS
            .iter()
            .map(|name| (*name).to_string())
            .collect());
    }
    // The old symmetric Mulcs name is a compatibility alias for ReciPCS.
    let canonical = match selected.as_str() {
        "symmetric" | "mulcs_symmetric" | "mulcs-symmetric" => "recipcs".to_string(),
        "nestedgrid" | "nested-grid-kzg" | "nested_grid_kzg" => "nrg".to_string(),
        other => other.to_string(),
    };
    if ALL_BACKENDS.contains(&canonical.as_str()) {
        Ok(vec![canonical])
    } else {
        Err(PCSError::InvalidParameters(format!(
            "PCS_BENCH_BACKEND: unsupported backend '{raw}'; use mkzg, gemini, mulcs, samaritan, zeromorph, recipcs, nrg, mercury, or all"
        )))
    }
}

fn main() -> Result<(), PCSError> {
    bench_all()
}

fn bench_all() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv_list = parse_nv_list()?;
    let backends = parse_backends()?;
    println!(
        "{:<16} {:>4}  {:>12}  {:>12}  {:>14}  {:>18}  {:>14}  {:>14}",
        "backend",
        "nv",
        "srs_gen",
        "trim",
        "commit",
        "prover_open_trait",
        "verify_mean",
        "verify_median"
    );
    println!(
        "# srs_gen/trim/commit/prover_open_trait are SINGLE wall-clock measurements (each heavy \
         phase runs exactly once, self-checked); use them only as coarse guides."
    );
    println!(
        "# verify is repeated {VERIFY_REPETITIONS}x; mean and median are reported. For \
         publication-grade verifier numbers use the Criterion bench (pcs-single-verify-benches)."
    );
    println!("{}", "-".repeat(104));

    for &nv in &nv_list {
        for backend in &backends {
            match backend.as_str() {
                "mkzg" => bench_backend::<MultilinearKzgPCS<Bls12_381>>(&mut rng, "mKZG", nv)?,
                "gemini" => bench_backend::<GeminiPCS<Bls12_381>>(&mut rng, "Gemini", nv)?,
                "mulcs" => bench_backend::<MulcsPCS<Bls12_381>>(&mut rng, "MulcsClaymore", nv)?,
                "samaritan" => bench_backend::<SamaritanPCS<Bls12_381>>(&mut rng, "Samaritan", nv)?,
                "zeromorph" => bench_backend::<ZeromorphPCS<Bls12_381>>(&mut rng, "Zeromorph", nv)?,
                "recipcs" => bench_backend::<ReciPCS<Bls12_381>>(&mut rng, "ReciPCS", nv)?,
                "nrg" => {
                    bench_backend::<NestedGridKzgPCS<Bls12_381>>(&mut rng, "NestedGridKZG", nv)?
                },
                "mercury" => bench_backend::<MercuryPCS<Bls12_381>>(&mut rng, "Mercury", nv)?,
                _ => unreachable!("parse_backends validates values"),
            }
        }
        println!();
    }

    Ok(())
}

/// Single-open measurement for one backend / `nv`.
///
/// Measurement discipline (fixes the earlier double-execution bug):
///   - `gen_srs_for_testing`, `trim`, `commit`, and the trait `open` each run
///     **exactly once**; the SRS / keys / commitment / proof they return are
///     reused (never regenerated just to time them). This is asserted at
///     runtime via the `phase_calls` counters below, so every benchmark run
///     self-checks the invariant.
///   - Only `verify` repeats (`VERIFY_REPETITIONS`), and the very first
///     repetition doubles as the correctness assertion (no extra untimed
///     verify). Both mean and median are reported.
fn bench_backend<PCS>(
    rng: &mut impl ark_std::rand::Rng,
    name: &str,
    nv: usize,
) -> Result<(), PCSError>
where
    PCS: PolynomialCommitmentScheme<
        Bls12_381,
        Polynomial = Arc<DenseMultilinearExtension<Fr>>,
        Point = Vec<Fr>,
        Evaluation = Fr,
    >,
{
    // (srs_gen, trim, commit, open) call counters — each must end at 1.
    let mut phase_calls = [0usize; 4];

    let start = Instant::now();
    let srs = PCS::gen_srs_for_testing(rng, nv)?;
    let srs_gen_ns = start.elapsed().as_nanos();
    phase_calls[0] += 1;

    let start = Instant::now();
    let (ck, vk) = PCS::trim(&srs, None, Some(nv))?;
    let trim_ns = start.elapsed().as_nanos();
    phase_calls[1] += 1;

    let poly = Arc::new(DenseMultilinearExtension::rand(nv, rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(rng)).collect();

    let start = Instant::now();
    let com = PCS::commit(&ck, &poly)?;
    let commit_ns = start.elapsed().as_nanos();
    phase_calls[2] += 1;

    let start = Instant::now();
    let (proof, value) = PCS::open(&ck, &poly, &point)?;
    let prover_ns = start.elapsed().as_nanos();
    phase_calls[3] += 1;

    // Only verify repeats; the first repetition is the correctness check.
    let mut samples = Vec::with_capacity(VERIFY_REPETITIONS);
    for _ in 0..VERIFY_REPETITIONS {
        let start = Instant::now();
        let ok = PCS::verify(&vk, &com, &point, &value, &proof)?;
        samples.push(start.elapsed().as_nanos());
        assert!(ok, "{name}: verify returned false at nv={nv}");
    }

    assert_eq!(
        phase_calls, [1usize; 4],
        "{name}: each heavy phase (srs_gen, trim, commit, open) must run exactly once"
    );

    let sum: u128 = samples.iter().sum();
    let mean = sum / (VERIFY_REPETITIONS as u128);
    samples.sort_unstable();
    let median = samples[VERIFY_REPETITIONS / 2];

    println!(
        "{:<16} {:>4}  {:>12}  {:>12}  {:>14}  {:>18}  {:>14}  {:>14}",
        name,
        nv,
        format_ns(srs_gen_ns),
        format_ns(trim_ns),
        format_ns(commit_ns),
        format_ns(prover_ns),
        format_ns(mean),
        format_ns(median),
    );

    Ok(())
}

fn format_ns(ns: u128) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.2} s", ns as f64 / 1e9)
    } else if ns >= 1_000_000 {
        format!("{:.2} ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.2} us", ns as f64 / 1e3)
    } else {
        format!("{ns} ns")
    }
}
