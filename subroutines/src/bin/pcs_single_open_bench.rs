//! PCS single-open benchmark — one backend, one nv, one heavy phase per
//! process.
//!
//! Env vars:
//!   PCS_BENCH_BACKEND   required — single backend name
//!   PCS_BENCH_NV        required — single nv integer
//!   PCS_BENCH_THREADS   optional — rayon thread count, default num_cpus
//!   PCS_PROFILE         optional — existing profile flag
//!
//! Output: CSV lines with columns
//!   backend,nv,N,phase,scope,elapsed_ms,heavy_invocations,verify_iterations,
//!   threads,proof_bytes,status,api_used

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::UniformRand;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{sync::Arc, test_rng};
use std::{env, time::Instant};
use subroutines::pcs::{
    prelude::{
        ChopinPCS, GeminiPCS, MercuryPCS, MulcsPCS, MultilinearKzgPCS, NestedGridKzgPCS, PCSError,
        ReciPCS, SamaritanPCS, ZeromorphPCS,
    },
    PolynomialCommitmentScheme,
};

type E = Bls12_381;
const VERIFY_REPS: usize = 100;

fn require_env(key: &str) -> Result<String, PCSError> {
    env::var(key).map_err(|_| PCSError::InvalidParameters(format!("{key} is required")))
}

fn parse_nv() -> Result<usize, PCSError> {
    let raw = require_env("PCS_BENCH_NV")?;
    raw.trim()
        .parse::<usize>()
        .map_err(|_| PCSError::InvalidParameters(format!("PCS_BENCH_NV invalid: '{raw}'")))
}

fn parse_threads() -> usize {
    env::var("PCS_BENCH_THREADS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
        })
}

const CSV_HDR: &str =
    "backend,nv,N,phase,scope,elapsed_ms,heavy_invocations,verify_iterations,threads,proof_bytes,status,api_used";

fn csv(values: &[String]) {
    println!("{}", values.join(","));
}

// ── verify loop shared by all backends ──

fn verify_loop<PCS>(
    name: &str,
    nv: usize,
    n: usize,
    threads: usize,
    vk: &PCS::VerifierParam,
    com: &PCS::Commitment,
    point: &PCS::Point,
    eval: &PCS::Evaluation,
    proof: &PCS::Proof,
) -> Result<(), PCSError>
where
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<Fr>>,
        Point = Vec<Fr>,
        Evaluation = Fr,
    >,
{
    let mut buf = Vec::new();
    proof
        .serialize_compressed(&mut buf)
        .map_err(PCSError::from)?;
    let pb = buf.len();
    let mut samples = Vec::with_capacity(VERIFY_REPS);
    for _ in 0..VERIFY_REPS {
        let p2 = PCS::Proof::deserialize_compressed(buf.as_slice()).map_err(PCSError::from)?;
        let start = Instant::now();
        let ok = PCS::verify(vk, com, point, eval, &p2)?;
        assert!(ok, "{name}: verify returned false");
        samples.push(start.elapsed().as_nanos());
    }
    let sum: u128 = samples.iter().sum();
    let mean_ms = (sum / VERIFY_REPS as u128) as f64 / 1e6;
    samples.sort_unstable();
    let median_ms = samples[VERIFY_REPS / 2] as f64 / 1e6;
    csv(&[
        name.to_string(),
        nv.to_string(),
        n.to_string(),
        "verify_mean".into(),
        "verify_once".into(),
        format!("{mean_ms}"),
        "0".into(),
        VERIFY_REPS.to_string(),
        threads.to_string(),
        pb.to_string(),
        "pass".into(),
        "-".into(),
    ]);
    csv(&[
        name.to_string(),
        nv.to_string(),
        n.to_string(),
        "verify_median".into(),
        "verify_once".into(),
        format!("{median_ms}"),
        "0".into(),
        VERIFY_REPS.to_string(),
        threads.to_string(),
        pb.to_string(),
        "pass".into(),
        "-".into(),
    ]);
    Ok(())
}

fn commit_plus_open<PCS>(
    name: &str,
    nv: usize,
    n: usize,
    threads: usize,
    ck: &PCS::ProverParam,
    rng: &mut impl ark_std::rand::Rng,
) -> Result<(), PCSError>
where
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<Fr>>,
        Point = Vec<Fr>,
        Evaluation = Fr,
    >,
{
    let poly = Arc::new(DenseMultilinearExtension::rand(nv, rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(rng)).collect();
    let start = Instant::now();
    let _com = PCS::commit(ck, &poly)?;
    let (_proof, _eval) = PCS::open(ck, &poly, &point)?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    csv(&[
        name.to_string(),
        nv.to_string(),
        n.to_string(),
        "commit_open".into(),
        "commit_plus_open".into(),
        format!("{ms}"),
        "1".into(),
        "0".into(),
        threads.to_string(),
        "0".into(),
        "pass".into(),
        "trait_open".into(),
    ]);
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// Per-backend benchmarks
// ═══════════════════════════════════════════════════════════════════════

fn bench_mkzg(name: &str, nv: usize, n: usize, threads: usize) -> Result<(), PCSError> {
    type PCS = MultilinearKzgPCS<E>;
    let mut rng = test_rng();
    let srs = {
        let t = Instant::now();
        let s = PCS::gen_srs_for_testing(&mut rng, nv)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "srs".into(),
            "setup".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        s
    };
    let (ck, vk) = {
        let t = Instant::now();
        let (c, v) = PCS::trim(&srs, None, Some(nv))?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "trim".into(),
            "trim".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        (c, v)
    };
    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();
    let com = {
        let t = Instant::now();
        let c = PCS::commit(&ck, &poly)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "commit".into(),
            "commit".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        c
    };
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = PCS::open(&ck, &poly, &point)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_precommitted".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "trait_open_no_recommit".into(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    commit_plus_open::<PCS>(name, nv, n, threads, &ck, &mut rng)?;
    Ok(())
}

fn bench_mulcs(name: &str, nv: usize, n: usize, threads: usize) -> Result<(), PCSError> {
    type PCS = MulcsPCS<E>;
    let mut rng = test_rng();
    let srs = {
        let t = Instant::now();
        let s = PCS::gen_srs_for_testing(&mut rng, nv)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "srs".into(),
            "setup".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        s
    };
    let (ck, vk) = {
        let t = Instant::now();
        let (c, v) = PCS::trim(&srs, None, Some(nv))?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "trim".into(),
            "trim".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        (c, v)
    };
    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();
    let com = {
        let t = Instant::now();
        let c = PCS::commit(&ck, &poly)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "commit".into(),
            "commit".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        c
    };
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = PCS::open(&ck, &poly, &point)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_precommitted".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "trait_open_no_recommit".into(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    commit_plus_open::<PCS>(name, nv, n, threads, &ck, &mut rng)?;
    Ok(())
}

fn bench_recipcs(name: &str, nv: usize, n: usize, threads: usize) -> Result<(), PCSError> {
    type PCS = ReciPCS<E>;
    let mut rng = test_rng();
    let srs = {
        let t = Instant::now();
        let s = PCS::gen_srs_for_testing(&mut rng, nv)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "srs".into(),
            "setup".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        s
    };
    let (ck, vk) = {
        let t = Instant::now();
        let (c, v) = PCS::trim(&srs, None, Some(nv))?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "trim".into(),
            "trim".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        (c, v)
    };
    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();
    let com = {
        let t = Instant::now();
        let c = PCS::commit(&ck, &poly)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "commit".into(),
            "commit".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        c
    };
    let val = poly.evaluate(&point).unwrap();
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = ReciPCS::<E>::open_with_commitment(&ck, &poly, &point, val, &com)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_precommitted".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    commit_plus_open::<PCS>(name, nv, n, threads, &ck, &mut rng)?;
    Ok(())
}

fn bench_gemini(name: &str, nv: usize, n: usize, threads: usize) -> Result<(), PCSError> {
    type PCS = GeminiPCS<E>;
    let mut rng = test_rng();
    let srs = {
        let t = Instant::now();
        let s = PCS::gen_srs_for_testing(&mut rng, nv)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "srs".into(),
            "setup".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        s
    };
    let (ck, vk) = {
        let t = Instant::now();
        let (c, v) = PCS::trim(&srs, None, Some(nv))?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "trim".into(),
            "trim".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        (c, v)
    };
    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();
    let com = {
        let t = Instant::now();
        let c = PCS::commit(&ck, &poly)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "commit".into(),
            "commit".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        c
    };
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = GeminiPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_precommitted".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    commit_plus_open::<PCS>(name, nv, n, threads, &ck, &mut rng)?;
    Ok(())
}

fn bench_samaritan(name: &str, nv: usize, n: usize, threads: usize) -> Result<(), PCSError> {
    type PCS = SamaritanPCS<E>;
    let mut rng = test_rng();
    let srs = {
        let t = Instant::now();
        let s = PCS::gen_srs_for_testing(&mut rng, nv)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "srs".into(),
            "setup".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        s
    };
    let (ck, vk) = {
        let t = Instant::now();
        let (c, v) = PCS::trim(&srs, None, Some(nv))?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "trim".into(),
            "trim".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        (c, v)
    };
    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();
    let com = {
        let t = Instant::now();
        let c = PCS::commit(&ck, &poly)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "commit".into(),
            "commit".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        c
    };
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = SamaritanPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_precommitted".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    commit_plus_open::<PCS>(name, nv, n, threads, &ck, &mut rng)?;
    Ok(())
}

fn bench_zeromorph(name: &str, nv: usize, n: usize, threads: usize) -> Result<(), PCSError> {
    type PCS = ZeromorphPCS<E>;
    let mut rng = test_rng();
    let srs = {
        let t = Instant::now();
        let s = PCS::gen_srs_for_testing(&mut rng, nv)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "srs".into(),
            "setup".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        s
    };
    let (ck, vk) = {
        let t = Instant::now();
        let (c, v) = PCS::trim(&srs, None, Some(nv))?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "trim".into(),
            "trim".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        (c, v)
    };
    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();
    let com = {
        let t = Instant::now();
        let c = PCS::commit(&ck, &poly)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "commit".into(),
            "commit".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        c
    };
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = ZeromorphPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_precommitted".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    commit_plus_open::<PCS>(name, nv, n, threads, &ck, &mut rng)?;
    Ok(())
}

fn bench_nrg(name: &str, nv: usize, n: usize, threads: usize) -> Result<(), PCSError> {
    type PCS = NestedGridKzgPCS<E>;
    let mut rng = test_rng();
    let srs = {
        let t = Instant::now();
        let s = PCS::gen_srs_for_testing(&mut rng, nv)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "srs".into(),
            "setup".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        s
    };
    let (ck, vk) = {
        let t = Instant::now();
        let (c, v) = PCS::trim(&srs, None, Some(nv))?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "trim".into(),
            "trim".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        (c, v)
    };
    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();
    let com = {
        let t = Instant::now();
        let c = PCS::commit(&ck, &poly)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "commit".into(),
            "commit".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        c
    };
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = NestedGridKzgPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_precommitted".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    commit_plus_open::<PCS>(name, nv, n, threads, &ck, &mut rng)?;
    Ok(())
}

fn bench_mercury(name: &str, nv: usize, n: usize, threads: usize) -> Result<(), PCSError> {
    type PCS = MercuryPCS<E>;
    let mut rng = test_rng();
    let srs = {
        let t = Instant::now();
        let s = PCS::gen_srs_for_testing(&mut rng, nv)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "srs".into(),
            "setup".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        s
    };
    let (ck, vk) = {
        let t = Instant::now();
        let (c, v) = PCS::trim(&srs, None, Some(nv))?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "trim".into(),
            "trim".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        (c, v)
    };
    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();
    let com = {
        let t = Instant::now();
        let c = PCS::commit(&ck, &poly)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "commit".into(),
            "commit".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        c
    };
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = MercuryPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_precommitted".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    commit_plus_open::<PCS>(name, nv, n, threads, &ck, &mut rng)?;
    Ok(())
}

fn bench_chopin(name: &str, nv: usize, n: usize, threads: usize) -> Result<(), PCSError> {
    type PCS = ChopinPCS<E>;
    let mut rng = test_rng();
    let srs = {
        let t = Instant::now();
        let s = PCS::gen_srs_for_testing(&mut rng, nv)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "srs".into(),
            "setup".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        s
    };
    let (ck, vk) = {
        let t = Instant::now();
        let (c, v) = PCS::trim(&srs, None, Some(nv))?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "trim".into(),
            "trim".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        (c, v)
    };
    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();
    let com = {
        let t = Instant::now();
        let c = PCS::commit(&ck, &poly)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "commit".into(),
            "commit".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
        ]);
        c
    };
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = ChopinPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
        csv(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_precommitted".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    commit_plus_open::<PCS>(name, nv, n, threads, &ck, &mut rng)?;
    Ok(())
}

fn main() -> Result<(), PCSError> {
    let backend = require_env("PCS_BENCH_BACKEND")?;
    let nv = parse_nv()?;
    let n = 1usize << nv;
    let threads = parse_threads();
    println!("{CSV_HDR}");
    let run = || -> Result<(), PCSError> {
        match backend.trim().to_ascii_lowercase().as_str() {
            "mkzg" => bench_mkzg("mKZG", nv, n, threads),
            "gemini" => bench_gemini("Gemini", nv, n, threads),
            "mulcs" => bench_mulcs("MulcsClaymore", nv, n, threads),
            "samaritan" => bench_samaritan("Samaritan", nv, n, threads),
            "zeromorph" => bench_zeromorph("Zeromorph", nv, n, threads),
            "recipcs" => bench_recipcs("ReciPCS", nv, n, threads),
            "nrg" => bench_nrg("NestedGridKZG", nv, n, threads),
            "mercury" => bench_mercury("Mercury", nv, n, threads),
            "chopin" => bench_chopin("Chopin", nv, n, threads),
            other => Err(PCSError::InvalidParameters(format!(
                "unsupported '{other}'"
            ))),
        }
    };
    if let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
        pool.install(run)
    } else {
        env::set_var("RAYON_NUM_THREADS", threads.to_string());
        run()
    }
}
