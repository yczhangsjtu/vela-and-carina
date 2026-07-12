//! Dedicated NestedGridKZG (NRG-KZG) benchmark.
//!
//! Reports, per `nv`: dimensions, one-shot heavy phases, repeated verify, and
//! proof sizes. Heavy phases (SRS gen, trim, commit, core/trait open) run once;
//! only verify is repeated.
//!
//! Env vars:
//!   NRG_BENCH_NV_RANGE     default: 8,10,12,14,16,18,20
//!   NRG_VERIFY_REPETITIONS default: 100
//!
//! Examples:
//!   NRG_BENCH_NV_RANGE=8 cargo bench -p subroutines --bench
//! nested-grid-kzg-benches   NRG_BENCH_NV_RANGE=8,10,12,14,16,18,20 \
//!     cargo bench -p subroutines --bench nested-grid-kzg-benches

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::UniformRand;
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::{sync::Arc, test_rng};
use std::time::Instant;
use subroutines::pcs::{
    prelude::{NestedGridKzgPCS, PCSError},
    PolynomialCommitmentScheme,
};

type E = Bls12_381;

const DEFAULT_NV_LIST: [usize; 7] = [8, 10, 12, 14, 16, 18, 20];
const DEFAULT_VERIFY_REPS: usize = 100;

fn parse_nv_list() -> Result<Vec<usize>, PCSError> {
    match std::env::var("NRG_BENCH_NV_RANGE") {
        Ok(raw) => {
            let mut list = Vec::new();
            for s in raw.split(',') {
                let s = s.trim();
                if s.is_empty() {
                    continue;
                }
                let nv: usize = s.parse().map_err(|_| {
                    PCSError::InvalidParameters(format!("NRG_BENCH_NV_RANGE: invalid nv '{}'", s))
                })?;
                list.push(nv);
            }
            if list.is_empty() {
                Err(PCSError::InvalidParameters(
                    "NRG_BENCH_NV_RANGE: empty list".to_string(),
                ))
            } else {
                Ok(list)
            }
        },
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_NV_LIST.to_vec()),
        Err(e) => Err(PCSError::InvalidParameters(format!(
            "NRG_BENCH_NV_RANGE env error: {}",
            e
        ))),
    }
}

fn parse_verify_reps() -> usize {
    std::env::var("NRG_VERIFY_REPETITIONS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_VERIFY_REPS)
}

fn format_ns(ns: u128) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.2}s", ns as f64 / 1e9)
    } else if ns >= 1_000_000 {
        format!("{:.2}ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.2}us", ns as f64 / 1e3)
    } else {
        format!("{ns}ns")
    }
}

fn main() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv_list = parse_nv_list()?;
    let reps = parse_verify_reps();

    println!("# NestedGridKZG dedicated benchmark");
    println!("# heavy phases run once; verify_mean over {reps} repetitions");
    println!(
        "{:>3} {:>10} {:>7} {:>7} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10} {:>10} {:>10} {:>12}",
        "nv",
        "N",
        "M_L",
        "M_R",
        "srs_gen",
        "trim",
        "commit",
        "core_open",
        "trait_open",
        "verify",
        "proofB",
        "payloadB",
        "srs_g1_elts",
    );

    for &nv in &nv_list {
        let m_left = nv.div_ceil(2);
        let m_right = nv / 2;
        let big_ml = 1usize << m_left;
        let big_mr = 1usize << m_right;
        let n = big_ml * big_mr;

        let srs_gen_ns = {
            let start = Instant::now();
            let _ = NestedGridKzgPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
            start.elapsed().as_nanos()
        };
        let srs = NestedGridKzgPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;

        let trim_ns = {
            let start = Instant::now();
            let _ = NestedGridKzgPCS::<E>::trim(&srs, None, Some(nv))?;
            start.elapsed().as_nanos()
        };
        let (ck, vk) = NestedGridKzgPCS::<E>::trim(&srs, None, Some(nv))?;

        let evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, evals));
        let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

        let com = NestedGridKzgPCS::<E>::commit(&ck, &poly)?;
        let commit_ns = {
            let start = Instant::now();
            let _ = NestedGridKzgPCS::<E>::commit(&ck, &poly)?;
            start.elapsed().as_nanos()
        };

        // core_open: uses the already-computed commitment (no recommitment).
        let core_open_ns = {
            let start = Instant::now();
            let _ = NestedGridKzgPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
            start.elapsed().as_nanos()
        };
        // trait_open_total: trait `open`, which recomputes C_f for the transcript.
        let trait_open_ns = {
            let start = Instant::now();
            let _ = NestedGridKzgPCS::<E>::open(&ck, &poly, &point)?;
            start.elapsed().as_nanos()
        };

        let (proof, value) = NestedGridKzgPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
        assert!(NestedGridKzgPCS::<E>::verify(
            &vk, &com, &point, &value, &proof
        )?);

        let verify_ns = {
            let start = Instant::now();
            for _ in 0..reps {
                assert!(NestedGridKzgPCS::<E>::verify(
                    &vk, &com, &point, &value, &proof
                )?);
            }
            start.elapsed().as_nanos() / reps as u128
        };

        let proof_bytes = proof.serialized_size(Compress::Yes);
        let payload_bytes = proof.cryptographic_payload_bytes();

        println!(
            "{:>3} {:>10} {:>7} {:>7} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10} {:>10} {:>10} {:>12}",
            nv,
            n,
            big_ml,
            big_mr,
            format_ns(srs_gen_ns),
            format_ns(trim_ns),
            format_ns(commit_ns),
            format_ns(core_open_ns),
            format_ns(trait_open_ns),
            format_ns(verify_ns),
            proof_bytes,
            payload_bytes,
            n, // G1 SRS is exactly N elements; G2 material is exactly 5.
        );
    }
    Ok(())
}
