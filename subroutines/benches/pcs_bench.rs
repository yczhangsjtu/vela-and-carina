// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::UniformRand;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{sync::Arc, test_rng};
use std::{env, time::Instant};
use subroutines::pcs::{
    prelude::{
        GeminiPCS, MulcsPCS, MulcsSymmetricPCS, MultilinearKzgPCS, PCSError, SamaritanPCS,
        ZeromorphPCS,
    },
    PolynomialCommitmentScheme,
};

const DEFAULT_NV_LIST: [usize; 7] = [8, 10, 12, 14, 16, 18, 20];
const VERIFY_REPETITIONS: usize = 100;
const ALL_BACKENDS: [&str; 6] = [
    "mkzg",
    "gemini",
    "mulcs",
    "symmetric",
    "samaritan",
    "zeromorph",
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
    if ALL_BACKENDS.contains(&selected.as_str()) {
        Ok(vec![selected])
    } else {
        Err(PCSError::InvalidParameters(format!(
            "PCS_BENCH_BACKEND: unsupported backend '{raw}'; use mkzg, gemini, mulcs, symmetric, samaritan, zeromorph, or all"
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
        "{:<16} {:>4}  {:>12}  {:>12}  {:>14}  {:>14}  {:>14}",
        "backend", "nv", "srs_gen", "trim", "commit", "prover(open)", "verify"
    );
    println!("# verify is the mean of {VERIFY_REPETITIONS} repetitions; all other phases run once");
    println!("{}", "-".repeat(92));

    for &nv in &nv_list {
        for backend in &backends {
            match backend.as_str() {
                "mkzg" => bench_backend::<MultilinearKzgPCS<Bls12_381>>(&mut rng, "mKZG", nv)?,
                "gemini" => bench_backend::<GeminiPCS<Bls12_381>>(&mut rng, "Gemini", nv)?,
                "mulcs" => bench_backend::<MulcsPCS<Bls12_381>>(&mut rng, "MulcsClaymore", nv)?,
                "symmetric" => {
                    bench_backend::<MulcsSymmetricPCS<Bls12_381>>(&mut rng, "MulcsSymmetric", nv)?
                },
                "samaritan" => bench_backend::<SamaritanPCS<Bls12_381>>(&mut rng, "Samaritan", nv)?,
                "zeromorph" => bench_backend::<ZeromorphPCS<Bls12_381>>(&mut rng, "Zeromorph", nv)?,
                _ => unreachable!("parse_backends validates values"),
            }
        }
        println!();
    }

    Ok(())
}

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
    let srs_gen_ns = {
        let start = Instant::now();
        let _ = PCS::gen_srs_for_testing(rng, nv)?;
        start.elapsed().as_nanos()
    };

    let srs = PCS::gen_srs_for_testing(rng, nv)?;

    let trim_ns = {
        let start = Instant::now();
        let _ = PCS::trim(&srs, None, Some(nv))?;
        start.elapsed().as_nanos()
    };

    let (ck, vk) = PCS::trim(&srs, None, Some(nv))?;
    let poly = Arc::new(DenseMultilinearExtension::rand(nv, rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(rng)).collect();
    let com = PCS::commit(&ck, &poly)?;

    let commit_ns = {
        let start = Instant::now();
        let _ = PCS::commit(&ck, &poly)?;
        start.elapsed().as_nanos()
    };

    let prover_ns = {
        let start = Instant::now();
        let _ = PCS::open(&ck, &poly, &point)?;
        start.elapsed().as_nanos()
    };

    let (proof, value) = PCS::open(&ck, &poly, &point)?;
    assert!(PCS::verify(&vk, &com, &point, &value, &proof)?);

    let verify_ns = {
        let start = Instant::now();
        for _ in 0..VERIFY_REPETITIONS {
            assert!(PCS::verify(&vk, &com, &point, &value, &proof)?);
        }
        start.elapsed().as_nanos() / (VERIFY_REPETITIONS as u128)
    };

    println!(
        "{:<16} {:>4}  {:>12}  {:>12}  {:>14}  {:>14}  {:>14}",
        name,
        nv,
        format_ns(srs_gen_ns),
        format_ns(trim_ns),
        format_ns(commit_ns),
        format_ns(prover_ns),
        format_ns(verify_ns),
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
