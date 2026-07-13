//! PCS single-open benchmark — one backend, one nv, one heavy phase per
//! process.
//!
//! Env vars:
//!   PCS_BENCH_BACKEND   required — single backend name
//!   PCS_BENCH_NV        required — single nv integer
//!   PCS_BENCH_SEED      optional — u64 seed for deterministic (poly, point)
//! generation   PCS_BENCH_THREADS   optional — rayon thread count, default
//! num_cpus   PCS_PROFILE         optional — if set (non-empty, non-zero),
//! binary refuses to run                                and returns an error.
//! Profiling CSV must not be intermixed                                with
//! benchmark output.
//!
//! Output: CSV lines with columns
//!   backend,nv,N,phase,scope,elapsed_ms,heavy_invocations,verify_iterations,
//!   threads,proof_bytes,status,api_used,peak_rss_bytes

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::UniformRand;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{sync::Arc, test_rng};
use std::{env, process, time::Instant};
use subroutines::pcs::{
    prelude::{
        ChopinPCS, GeminiPCS, MercuryPCS, MulcsPCS, MultilinearKzgPCS, NestedGridKzgPCS, PCSError,
        ReciPCS, SamaritanPCS, ZeromorphPCS,
    },
    PolynomialCommitmentScheme,
};

type E = Bls12_381;
const VERIFY_REPS: usize = 100;
const MAX_NV: usize = 24;

// ── fixed CSV schema (13 columns) ──

const CSV_HDR: &str =
    "backend,nv,N,phase,scope,elapsed_ms,heavy_invocations,verify_iterations,threads,proof_bytes,status,api_used,peak_rss_bytes";

fn emit(values: &[String]) {
    assert_eq!(
        values.len(),
        CSV_HDR.split(',').count(),
        "CSV row column count mismatch"
    );
    println!("{}", values.join(","));
}

/// Derive a stable input seed from the master seed, variable count, and domain
/// separator.  The backend name is deliberately absent: every PCS at the same
/// `nv` must receive identical benchmark inputs.
fn input_seed(master: &[u8; 32], nv: usize, domain: u8) -> [u8; 32] {
    let mut seed = *master;
    for (i, byte) in nv.to_le_bytes().iter().enumerate() {
        seed[16 + i] ^= *byte;
    }
    seed[31] ^= domain;
    seed
}

fn require_env(key: &str) -> Result<String, PCSError> {
    env::var(key).map_err(|_| PCSError::InvalidParameters(format!("{key} is required")))
}

fn parse_nv() -> Result<usize, PCSError> {
    let raw = require_env("PCS_BENCH_NV")?;
    let nv: usize = raw
        .trim()
        .parse()
        .map_err(|_| PCSError::InvalidParameters(format!("PCS_BENCH_NV invalid: '{raw}'")))?;
    if nv == 0 {
        return Err(PCSError::InvalidParameters("nv must be >= 1".into()));
    }
    if nv >= usize::BITS as usize {
        return Err(PCSError::InvalidParameters(format!(
            "nv={nv} >= usize::BITS"
        )));
    }
    if nv > MAX_NV {
        return Err(PCSError::InvalidParameters(format!(
            "nv={nv} > MAX_NV={MAX_NV}"
        )));
    }
    Ok(nv)
}

fn parse_threads() -> usize {
    env::var("PCS_BENCH_THREADS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&t| t > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
        })
}

/// The legacy trait-open measurement is an audit-only metric.  It is disabled
/// by default so it cannot add avoidable heat or runtime to paper-data runs.
fn include_legacy_trait_open() -> bool {
    matches!(
        env::var("PCS_BENCH_INCLUDE_LEGACY").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

fn parse_seed() -> [u8; 32] {
    let seed = env::var("PCS_BENCH_SEED")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0xdead_beef_cafe_babe);
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&seed.to_le_bytes());
    bytes
}

/// Reject PCS_PROFILE=1 in benchmark mode — profiling CSV must not intermix.
fn assert_no_profile() -> Result<(), PCSError> {
    if let Ok(val) = env::var("PCS_PROFILE") {
        let v = val.trim();
        if !v.is_empty() && v != "0" {
            return Err(PCSError::InvalidParameters(
                "PCS_PROFILE is set; benchmark binary refuses to run with profiling enabled (profiling CSV would intermix with main output)".into(),
            ));
        }
    }
    Ok(())
}

// ── RSS ──

fn peak_rss_string() -> String {
    // RSS is captured externally by the runner via /usr/bin/time -l.
    // The binary emits "unavailable"; the runner post-processes the CSV
    // to fill in the actual peak RSS from time's stderr output.
    "unavailable".to_string()
}

// ── deterministic poly/point generation ──

/// Create a deterministic RNG from seed bytes.
fn seeded_rng(seed: &[u8; 32]) -> impl ark_std::rand::Rng {
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
    ChaCha20Rng::from_seed(*seed)
}

/// Generate a random multilinear polynomial with deterministic seed.
fn seeded_poly(seed: &[u8; 32], nv: usize) -> Arc<DenseMultilinearExtension<Fr>> {
    let mut rng = seeded_rng(seed);
    let n = 1usize << nv;
    let evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, evals))
}

/// Generate a random evaluation point with deterministic seed.
fn seeded_point(seed: &[u8; 32], nv: usize) -> Vec<Fr> {
    let mut rng = seeded_rng(seed);
    (0..nv).map(|_| Fr::rand(&mut rng)).collect()
}

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

    // verify_core: proof is deserialized OUTSIDE the timing loop.
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
    let rss = peak_rss_string();
    emit(&[
        name.into(),
        nv.to_string(),
        n.to_string(),
        "verify_core_mean".into(),
        "verify_once".into(),
        format!("{mean_ms}"),
        "0".into(),
        VERIFY_REPS.to_string(),
        threads.to_string(),
        pb.to_string(),
        "pass".into(),
        "-".into(),
        rss.clone(),
    ]);
    emit(&[
        name.into(),
        nv.to_string(),
        n.to_string(),
        "verify_core_median".into(),
        "verify_once".into(),
        format!("{median_ms}"),
        "0".into(),
        VERIFY_REPS.to_string(),
        threads.to_string(),
        pb.to_string(),
        "pass".into(),
        "-".into(),
        rss,
    ]);
    Ok(())
}

// ── legacy trait open (audit reference only) ──

fn legacy_trait_open<PCS>(
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
    emit(&[
        name.into(),
        nv.to_string(),
        n.to_string(),
        "legacy_trait_open".into(),
        "legacy_trait_open".into(),
        format!("{ms}"),
        "1".into(),
        "0".into(),
        threads.to_string(),
        "0".into(),
        "pass".into(),
        "trait_open_may_recommit".into(),
        peak_rss_string(),
    ]);
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// Per-backend benchmarks (refactored with macro for shared structure)
// ═══════════════════════════════════════════════════════════════════════

macro_rules! bench_common_phases {
    ($PCS:ty, $name:expr, $nv:expr, $n:expr, $threads:expr, $rng:expr) => {{
        let rng_shared = &mut test_rng(); // for SRS only
        let srs = {
            let t = Instant::now();
            let s = <$PCS>::gen_srs_for_testing(rng_shared, $nv)?;
            emit(&[
                $name.into(),
                ($nv).to_string(),
                ($n).to_string(),
                "srs".into(),
                "setup".into(),
                format!("{}", t.elapsed().as_secs_f64() * 1000.0),
                "1".into(),
                "0".into(),
                ($threads).to_string(),
                "0".into(),
                "pass".into(),
                "-".into(),
                peak_rss_string(),
            ]);
            s
        };
        let (ck, vk) = {
            let t = Instant::now();
            let (c, v) = <$PCS>::trim(&srs, None, Some($nv))?;
            emit(&[
                $name.into(),
                ($nv).to_string(),
                ($n).to_string(),
                "trim".into(),
                "trim".into(),
                format!("{}", t.elapsed().as_secs_f64() * 1000.0),
                "1".into(),
                "0".into(),
                ($threads).to_string(),
                "0".into(),
                "pass".into(),
                "-".into(),
                peak_rss_string(),
            ]);
            (c, v)
        };
        (ck, vk)
    }};
}

macro_rules! bench_commit_phase {
    ($PCS:ty, $name:expr, $nv:expr, $n:expr, $threads:expr, $ck:expr, $poly:expr) => {{
        let t = Instant::now();
        let c = <$PCS>::commit(&$ck, &$poly)?;
        emit(&[
            $name.into(),
            ($nv).to_string(),
            ($n).to_string(),
            "commit".into(),
            "commit".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            ($threads).to_string(),
            "0".into(),
            "pass".into(),
            "-".into(),
            peak_rss_string(),
        ]);
        c
    }};
}

// ── backends using trait open (audit-clean: no C_f recommit) ──

fn bench_mkzg(
    name: &str,
    nv: usize,
    n: usize,
    threads: usize,
    master_seed: &[u8; 32],
) -> Result<(), PCSError> {
    type PCS = MultilinearKzgPCS<E>;
    let poly_seed = input_seed(master_seed, nv, b'p');
    let point_seed = input_seed(master_seed, nv, b'r');
    let (ck, vk) = bench_common_phases!(PCS, name, nv, n, threads, ());
    let poly = seeded_poly(&poly_seed, nv);
    let point = seeded_point(&point_seed, nv);
    let com = bench_commit_phase!(PCS, name, nv, n, threads, ck, poly);
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = PCS::open(&ck, &poly, &point)?;
        emit(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_prebound".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "trait_open_no_recommit".into(),
            peak_rss_string(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    if include_legacy_trait_open() {
        legacy_trait_open::<PCS>(name, nv, n, threads, &ck, &mut test_rng())?;
    }
    Ok(())
}

fn bench_mulcs(
    name: &str,
    nv: usize,
    n: usize,
    threads: usize,
    master_seed: &[u8; 32],
) -> Result<(), PCSError> {
    type PCS = MulcsPCS<E>;
    let poly_seed = input_seed(master_seed, nv, b'p');
    let point_seed = input_seed(master_seed, nv, b'r');
    let (ck, vk) = bench_common_phases!(PCS, name, nv, n, threads, ());
    let poly = seeded_poly(&poly_seed, nv);
    let point = seeded_point(&point_seed, nv);
    let com = bench_commit_phase!(PCS, name, nv, n, threads, ck, poly);
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = PCS::open(&ck, &poly, &point)?;
        emit(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_prebound".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "trait_open_no_recommit".into(),
            peak_rss_string(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    if include_legacy_trait_open() {
        legacy_trait_open::<PCS>(name, nv, n, threads, &ck, &mut test_rng())?;
    }
    Ok(())
}

// ── backends using open_with_commitment ──

fn bench_recipcs(
    name: &str,
    nv: usize,
    n: usize,
    threads: usize,
    master_seed: &[u8; 32],
) -> Result<(), PCSError> {
    type PCS = ReciPCS<E>;
    let poly_seed = input_seed(master_seed, nv, b'p');
    let point_seed = input_seed(master_seed, nv, b'r');
    let (ck, vk) = bench_common_phases!(PCS, name, nv, n, threads, ());
    let poly = seeded_poly(&poly_seed, nv);
    let point = seeded_point(&point_seed, nv);
    let com = bench_commit_phase!(PCS, name, nv, n, threads, ck, poly);
    let (proof, eval) = {
        let t = Instant::now();
        // eval part of protocol-internal computation; include in timing.
        let val = poly.evaluate(&point).unwrap();
        let (p, e) = ReciPCS::<E>::open_with_commitment(&ck, &poly, &point, val, &com)?;
        emit(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_prebound".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
            peak_rss_string(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    if include_legacy_trait_open() {
        legacy_trait_open::<PCS>(name, nv, n, threads, &ck, &mut test_rng())?;
    }
    Ok(())
}

fn bench_gemini(
    name: &str,
    nv: usize,
    n: usize,
    threads: usize,
    master_seed: &[u8; 32],
) -> Result<(), PCSError> {
    type PCS = GeminiPCS<E>;
    let poly_seed = input_seed(master_seed, nv, b'p');
    let point_seed = input_seed(master_seed, nv, b'r');
    let (ck, vk) = bench_common_phases!(PCS, name, nv, n, threads, ());
    let poly = seeded_poly(&poly_seed, nv);
    let point = seeded_point(&point_seed, nv);
    let com = bench_commit_phase!(PCS, name, nv, n, threads, ck, poly);
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = GeminiPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
        emit(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_prebound".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
            peak_rss_string(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    if include_legacy_trait_open() {
        legacy_trait_open::<PCS>(name, nv, n, threads, &ck, &mut test_rng())?;
    }
    Ok(())
}

fn bench_samaritan(
    name: &str,
    nv: usize,
    n: usize,
    threads: usize,
    master_seed: &[u8; 32],
) -> Result<(), PCSError> {
    type PCS = SamaritanPCS<E>;
    let poly_seed = input_seed(master_seed, nv, b'p');
    let point_seed = input_seed(master_seed, nv, b'r');
    let (ck, vk) = bench_common_phases!(PCS, name, nv, n, threads, ());
    let poly = seeded_poly(&poly_seed, nv);
    let point = seeded_point(&point_seed, nv);
    let com = bench_commit_phase!(PCS, name, nv, n, threads, ck, poly);
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = SamaritanPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
        emit(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_prebound".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
            peak_rss_string(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    if include_legacy_trait_open() {
        legacy_trait_open::<PCS>(name, nv, n, threads, &ck, &mut test_rng())?;
    }
    Ok(())
}

fn bench_zeromorph(
    name: &str,
    nv: usize,
    n: usize,
    threads: usize,
    master_seed: &[u8; 32],
) -> Result<(), PCSError> {
    type PCS = ZeromorphPCS<E>;
    let poly_seed = input_seed(master_seed, nv, b'p');
    let point_seed = input_seed(master_seed, nv, b'r');
    let (ck, vk) = bench_common_phases!(PCS, name, nv, n, threads, ());
    let poly = seeded_poly(&poly_seed, nv);
    let point = seeded_point(&point_seed, nv);
    let com = bench_commit_phase!(PCS, name, nv, n, threads, ck, poly);
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = ZeromorphPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
        emit(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_prebound".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
            peak_rss_string(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    if include_legacy_trait_open() {
        legacy_trait_open::<PCS>(name, nv, n, threads, &ck, &mut test_rng())?;
    }
    Ok(())
}

fn bench_nrg(
    name: &str,
    nv: usize,
    n: usize,
    threads: usize,
    master_seed: &[u8; 32],
) -> Result<(), PCSError> {
    type PCS = NestedGridKzgPCS<E>;
    let poly_seed = input_seed(master_seed, nv, b'p');
    let point_seed = input_seed(master_seed, nv, b'r');
    let (ck, vk) = bench_common_phases!(PCS, name, nv, n, threads, ());
    let poly = seeded_poly(&poly_seed, nv);
    let point = seeded_point(&point_seed, nv);
    let com = bench_commit_phase!(PCS, name, nv, n, threads, ck, poly);
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = NestedGridKzgPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
        emit(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_prebound".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
            peak_rss_string(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    if include_legacy_trait_open() {
        legacy_trait_open::<PCS>(name, nv, n, threads, &ck, &mut test_rng())?;
    }
    Ok(())
}

fn bench_mercury(
    name: &str,
    nv: usize,
    n: usize,
    threads: usize,
    master_seed: &[u8; 32],
) -> Result<(), PCSError> {
    type PCS = MercuryPCS<E>;
    let poly_seed = input_seed(master_seed, nv, b'p');
    let point_seed = input_seed(master_seed, nv, b'r');
    let (ck, vk) = bench_common_phases!(PCS, name, nv, n, threads, ());
    let poly = seeded_poly(&poly_seed, nv);
    let point = seeded_point(&point_seed, nv);
    let com = bench_commit_phase!(PCS, name, nv, n, threads, ck, poly);
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = MercuryPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
        emit(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_prebound".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
            peak_rss_string(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    if include_legacy_trait_open() {
        legacy_trait_open::<PCS>(name, nv, n, threads, &ck, &mut test_rng())?;
    }
    Ok(())
}

fn bench_chopin(
    name: &str,
    nv: usize,
    n: usize,
    threads: usize,
    master_seed: &[u8; 32],
) -> Result<(), PCSError> {
    type PCS = ChopinPCS<E>;
    let poly_seed = input_seed(master_seed, nv, b'p');
    let point_seed = input_seed(master_seed, nv, b'r');
    let (ck, vk) = bench_common_phases!(PCS, name, nv, n, threads, ());
    let poly = seeded_poly(&poly_seed, nv);
    let point = seeded_point(&point_seed, nv);
    let com = bench_commit_phase!(PCS, name, nv, n, threads, ck, poly);
    let (proof, eval) = {
        let t = Instant::now();
        let (p, e) = ChopinPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
        emit(&[
            name.into(),
            nv.to_string(),
            n.to_string(),
            "core_open".into(),
            "core_open_prebound".into(),
            format!("{}", t.elapsed().as_secs_f64() * 1000.0),
            "1".into(),
            "0".into(),
            threads.to_string(),
            "0".into(),
            "pass".into(),
            "open_with_commitment".into(),
            peak_rss_string(),
        ]);
        (p, e)
    };
    verify_loop::<PCS>(name, nv, n, threads, &vk, &com, &point, &eval, &proof)?;
    if include_legacy_trait_open() {
        legacy_trait_open::<PCS>(name, nv, n, threads, &ck, &mut test_rng())?;
    }
    Ok(())
}

fn main() {
    if let Err(e) = try_main() {
        eprintln!("FATAL: {e:?}");
        process::exit(1);
    }
}

fn try_main() -> Result<(), PCSError> {
    let backend = require_env("PCS_BENCH_BACKEND")?;
    // Validate nv early, before allocating anything based on nv
    let nv = parse_nv()?;
    assert_no_profile()?;
    let n = 1usize << nv;
    let threads = parse_threads();
    let master_seed = parse_seed();

    // Only print header once, before any data rows
    println!("{CSV_HDR}");

    let run = || -> Result<(), PCSError> {
        match backend.trim().to_ascii_lowercase().as_str() {
            "mkzg" => bench_mkzg("mKZG", nv, n, threads, &master_seed),
            "gemini" => bench_gemini("Gemini", nv, n, threads, &master_seed),
            "mulcs" => bench_mulcs("MulcsClaymore", nv, n, threads, &master_seed),
            "samaritan" => bench_samaritan("Samaritan", nv, n, threads, &master_seed),
            "zeromorph" => bench_zeromorph("Zeromorph", nv, n, threads, &master_seed),
            "recipcs" => bench_recipcs("ReciPCS", nv, n, threads, &master_seed),
            "nrg" => bench_nrg("NestedGridKZG", nv, n, threads, &master_seed),
            "mercury" => bench_mercury("Mercury", nv, n, threads, &master_seed),
            "chopin" => bench_chopin("Chopin", nv, n, threads, &master_seed),
            other => Err(PCSError::InvalidParameters(format!(
                "unsupported backend '{other}'"
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
