// Dedicated Mercury single-open benchmark.
//
// Two modes (MERCURY_BENCH_MODE):
//
//   comparison (default): one consistent CSV table across the selected
//     single-open backends. For each backend/nv, srs_gen / trim / commit /
//     trait_open each run EXACTLY ONCE (self-checked; the returned objects are
//     reused), and only verify repeats. Reports both mean and median verify,
//     the compressed proof bytes, and the compressed verifier-key bytes (setup
//     storage the verifier keeps). Mercury appears once as a normal backend
//     using its trait `open` (which includes the C_f recommit). The canonical
//     backends are mkzg, gemini, recipcs, samaritan, zeromorph, nrg, mercury;
//     the Claymore `mulcs` row is LEGACY and only appears when explicitly
//     selected (BACKEND=mulcs).
//
//   detail (MERCURY_BENCH_MODE=detail): Mercury ONLY. Additionally reports
//     `core_open` (`open_with_commitment`, no C_f recommit) vs `trait_open`
//     (with recommit), the tight SRS G1/G2 element counts, prover-key and
//     verifier-key bytes, and the two dominant N-sized prover MSM lengths plus
//     the analytic core-prover scalar total `2N + O(sqrt N)`. Only `core_open`
//     may be used to reason about the paper's `2N + O(sqrt N)` MSM count;
//     `trait_open` is NOT valid for that because it adds an N-MSM recommit.
//
// Env vars (all strictly parsed; invalid input is a hard error, never silently
// dropped):
//   MERCURY_BENCH_NV_RANGE       comma separated positive ints; default
//                                8,10,12,14,16,18,20. Empty tokens, non-ints,
//                                nv<1, out-of-range, and duplicates are errors.
//   MERCURY_BENCH_BACKEND        mkzg|gemini|mulcs|recipcs|samaritan|zeromorph|
//                                nrg|mercury|all (+ symmetric/nestedgrid
//                                aliases). Unknown values are errors.
//   MERCURY_BENCH_MODE           comparison|detail (default comparison).
//   MERCURY_VERIFY_REPETITIONS   default 100.
//
// For exact per-MSM scalar counts set PCS_PROFILE=1 (the internal ScopedTimers
// then emit real `count` columns). Run each large nv in its OWN process, e.g.
//   MERCURY_BENCH_NV_RANGE=20 MERCURY_BENCH_BACKEND=mercury \
//     cargo bench -p subroutines --bench mercury-benches

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::UniformRand;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::CanonicalSerialize;
use ark_std::{sync::Arc, test_rng};
use std::{collections::BTreeSet, env, time::Instant};
use subroutines::pcs::{
    prelude::{
        GeminiPCS, MercuryPCS, MulcsPCS, MultilinearKzgPCS, NestedGridKzgPCS, PCSError, ReciPCS,
        SamaritanPCS, ZeromorphPCS,
    },
    PolynomialCommitmentScheme,
};

type E = Bls12_381;

const DEFAULT_NV: [usize; 7] = [8, 10, 12, 14, 16, 18, 20];
const CANONICAL: [&str; 7] = [
    "mkzg",
    "gemini",
    "recipcs",
    "samaritan",
    "zeromorph",
    "nrg",
    "mercury",
];

#[derive(PartialEq, Eq)]
enum Mode {
    Comparison,
    Detail,
}

fn parse_nv() -> Result<Vec<usize>, PCSError> {
    let raw = match env::var("MERCURY_BENCH_NV_RANGE") {
        Ok(v) => v,
        Err(env::VarError::NotPresent) => return Ok(DEFAULT_NV.to_vec()),
        Err(e) => {
            return Err(PCSError::InvalidParameters(format!(
                "MERCURY_BENCH_NV_RANGE env error: {e}"
            )))
        },
    };
    let mut list = Vec::new();
    let mut seen = BTreeSet::new();
    for tok in raw.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            return Err(PCSError::InvalidParameters(format!(
                "MERCURY_BENCH_NV_RANGE: empty token in '{raw}'"
            )));
        }
        let nv: usize = tok.parse().map_err(|_| {
            PCSError::InvalidParameters(format!("MERCURY_BENCH_NV_RANGE: invalid nv '{tok}'"))
        })?;
        if nv < 1 {
            return Err(PCSError::InvalidParameters(format!(
                "MERCURY_BENCH_NV_RANGE: nv must be >= 1, got {nv}"
            )));
        }
        if nv >= usize::BITS as usize {
            return Err(PCSError::InvalidParameters(format!(
                "MERCURY_BENCH_NV_RANGE: nv {nv} exceeds platform word size"
            )));
        }
        if !seen.insert(nv) {
            return Err(PCSError::InvalidParameters(format!(
                "MERCURY_BENCH_NV_RANGE: duplicate nv {nv}"
            )));
        }
        list.push(nv);
    }
    if list.is_empty() {
        return Err(PCSError::InvalidParameters(
            "MERCURY_BENCH_NV_RANGE: empty list".to_string(),
        ));
    }
    Ok(list)
}

fn parse_backends() -> Result<Vec<String>, PCSError> {
    let raw = match env::var("MERCURY_BENCH_BACKEND") {
        Ok(v) => v,
        Err(env::VarError::NotPresent) => {
            return Ok(CANONICAL.iter().map(|s| s.to_string()).collect())
        },
        Err(e) => {
            return Err(PCSError::InvalidParameters(format!(
                "MERCURY_BENCH_BACKEND env error: {e}"
            )))
        },
    };
    let sel = raw.trim().to_ascii_lowercase();
    if sel == "all" {
        return Ok(CANONICAL.iter().map(|s| s.to_string()).collect());
    }
    let canonical = match sel.as_str() {
        "symmetric" | "mulcs_symmetric" | "mulcs-symmetric" => "recipcs",
        "nestedgrid" | "nested-grid-kzg" | "nested_grid_kzg" => "nrg",
        "mulcs" | "mulcsclaymore" | "claymore" => "mulcs",
        other => other,
    };
    if CANONICAL.contains(&canonical) || canonical == "mulcs" {
        Ok(vec![canonical.to_string()])
    } else {
        Err(PCSError::InvalidParameters(format!(
            "MERCURY_BENCH_BACKEND: unknown backend '{raw}'; use mkzg, gemini, mulcs, recipcs, \
             samaritan, zeromorph, nrg, mercury, or all"
        )))
    }
}

fn parse_mode() -> Result<Mode, PCSError> {
    match env::var("MERCURY_BENCH_MODE") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "comparison" | "compare" | "" => Ok(Mode::Comparison),
            "detail" => Ok(Mode::Detail),
            other => Err(PCSError::InvalidParameters(format!(
                "MERCURY_BENCH_MODE: unknown '{other}'; use comparison or detail"
            ))),
        },
        Err(env::VarError::NotPresent) => Ok(Mode::Comparison),
        Err(e) => Err(PCSError::InvalidParameters(format!(
            "MERCURY_BENCH_MODE env error: {e}"
        ))),
    }
}

fn verify_reps() -> Result<usize, PCSError> {
    match env::var("MERCURY_VERIFY_REPETITIONS") {
        Ok(v) => {
            let r: usize = v.trim().parse().map_err(|_| {
                PCSError::InvalidParameters(format!("MERCURY_VERIFY_REPETITIONS: invalid '{v}'"))
            })?;
            if r == 0 {
                return Err(PCSError::InvalidParameters(
                    "MERCURY_VERIFY_REPETITIONS must be >= 1".to_string(),
                ));
            }
            Ok(r)
        },
        Err(env::VarError::NotPresent) => Ok(100),
        Err(e) => Err(PCSError::InvalidParameters(format!(
            "MERCURY_VERIFY_REPETITIONS env error: {e}"
        ))),
    }
}

fn main() -> Result<(), PCSError> {
    let nv_list = parse_nv()?;
    let reps = verify_reps()?;
    let mode = parse_mode()?;
    let backends = parse_backends()?;

    if nv_list.iter().any(|&nv| nv >= 18) {
        eprintln!(
            "# NOTE: large nv requested; run each nv/backend in its OWN process to bound peak RSS:"
        );
        eprintln!(
            "#   MERCURY_BENCH_NV_RANGE=20 MERCURY_BENCH_BACKEND=mercury cargo bench -p \
             subroutines --bench mercury-benches"
        );
    }

    match mode {
        Mode::Comparison => run_comparison(&nv_list, &backends, reps),
        Mode::Detail => {
            // detail mode is Mercury-only; refuse an explicit non-mercury backend.
            let explicit_non_mercury = env::var("MERCURY_BENCH_BACKEND").is_ok()
                && backends != vec!["mercury".to_string()]
                && backends.len() != CANONICAL.len();
            if explicit_non_mercury {
                return Err(PCSError::InvalidParameters(
                    "MERCURY_BENCH_MODE=detail is Mercury-only; unset MERCURY_BENCH_BACKEND or set \
                     it to mercury/all"
                        .to_string(),
                ));
            }
            run_detail(&nv_list, reps)
        },
    }
}

fn run_comparison(nv_list: &[usize], backends: &[String], reps: usize) -> Result<(), PCSError> {
    println!("# Mercury comparison table (BLS12-381)");
    println!(
        "# srs_gen/trim/commit/trait_open: single wall-clock each (run once, reused). verify: \
         mean & median over {reps} reps. vk_bytes = compressed verifier-key (setup storage)."
    );
    println!(
        "backend,nv,srs_gen_ms,trim_ms,commit_ms,trait_open_ms,verify_us_mean,verify_us_median,\
         proof_bytes,vk_bytes,tag"
    );
    for &nv in nv_list {
        for b in backends {
            match b.as_str() {
                "mkzg" => compare_row::<MultilinearKzgPCS<E>>("mKZG", false, nv, reps)?,
                "gemini" => compare_row::<GeminiPCS<E>>("Gemini", false, nv, reps)?,
                "recipcs" => compare_row::<ReciPCS<E>>("ReciPCS", false, nv, reps)?,
                "samaritan" => compare_row::<SamaritanPCS<E>>("Samaritan", false, nv, reps)?,
                "zeromorph" => compare_row::<ZeromorphPCS<E>>("Zeromorph", false, nv, reps)?,
                "nrg" => compare_row::<NestedGridKzgPCS<E>>("NestedGridKZG", false, nv, reps)?,
                "mercury" => compare_row::<MercuryPCS<E>>("Mercury", false, nv, reps)?,
                "mulcs" => compare_row::<MulcsPCS<E>>("MulcsClaymore", true, nv, reps)?,
                other => {
                    return Err(PCSError::InvalidParameters(format!(
                        "unreachable backend {other}"
                    )))
                },
            }
        }
    }
    Ok(())
}

fn mean_median(mut samples: Vec<f64>) -> (f64, f64) {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    (mean, median)
}

fn compressed_len<T: CanonicalSerialize>(x: &T) -> usize {
    let mut b = Vec::new();
    x.serialize_compressed(&mut b).unwrap();
    b.len()
}

fn compare_row<PCS>(name: &str, legacy: bool, nv: usize, reps: usize) -> Result<(), PCSError>
where
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<Fr>>,
        Point = Vec<Fr>,
        Evaluation = Fr,
    >,
    PCS::Proof: CanonicalSerialize,
    PCS::VerifierParam: CanonicalSerialize,
{
    let mut rng = test_rng();
    let mut calls = [0usize; 4]; // srs, trim, commit, trait_open — each must be 1.

    let t = Instant::now();
    let srs = PCS::gen_srs_for_testing(&mut rng, nv)?;
    let srs_ms = t.elapsed().as_secs_f64() * 1e3;
    calls[0] += 1;

    let t = Instant::now();
    let (ck, vk) = PCS::trim(&srs, None, Some(nv))?;
    let trim_ms = t.elapsed().as_secs_f64() * 1e3;
    calls[1] += 1;

    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

    let t = Instant::now();
    let com = PCS::commit(&ck, &poly)?;
    let commit_ms = t.elapsed().as_secs_f64() * 1e3;
    calls[2] += 1;

    let t = Instant::now();
    let (proof, value) = PCS::open(&ck, &poly, &point)?;
    let open_ms = t.elapsed().as_secs_f64() * 1e3;
    calls[3] += 1;

    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let ok = PCS::verify(&vk, &com, &point, &value, &proof)?;
        samples.push(t.elapsed().as_secs_f64() * 1e6);
        assert!(ok, "{name}: verify returned false at nv={nv}");
    }
    assert_eq!(
        calls, [1usize; 4],
        "{name}: each heavy phase must run exactly once"
    );
    let (mean, median) = mean_median(samples);
    let tag = if legacy { "legacy" } else { "" };
    println!(
        "{name},{nv},{srs_ms:.3},{trim_ms:.3},{commit_ms:.3},{open_ms:.3},{mean:.3},{median:.3},{},{},{tag}",
        compressed_len(&proof),
        compressed_len(&vk)
    );
    Ok(())
}

fn run_detail(nv_list: &[usize], reps: usize) -> Result<(), PCSError> {
    println!(
        "# Mercury detail (Mercury-only): core_open (no C_f recommit) vs trait_open (recommit)"
    );
    println!(
        "# Only core_open is valid for the paper's 2N+O(sqrt N) MSM claim; trait_open adds an \
         N-MSM recommit."
    );
    println!(
        "nv,N,b,b_row,srs_g1,srs_g2,pk_bytes,vk_bytes,commit_ms,core_open_ms,trait_open_ms,\
         verify_us_mean,verify_us_median,proof_bytes,msm_q_len,msm_quot_f_len,core_scalar_total"
    );
    for &nv in nv_list {
        detail_row(nv, reps)?;
    }
    Ok(())
}

fn detail_row(nv: usize, reps: usize) -> Result<(), PCSError> {
    let mut rng = test_rng();
    let n = 1usize << nv;
    let t = nv.div_ceil(2);
    let b = 1usize << t;
    let b_row = 1usize << (nv - t);

    let mut calls = [0usize; 5]; // srs, trim, commit, core_open, trait_open.

    let srs = MercuryPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
    calls[0] += 1;
    let (ck, vk) = MercuryPCS::<E>::trim(&srs, None, Some(nv))?;
    calls[1] += 1;
    let srs_g1 = ck.g1_powers.len();
    let srs_g2 = 2usize; // exactly [1]_2, [tau]_2
    let pk_bytes = compressed_len(&ck);
    let vk_bytes = compressed_len(&vk);

    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

    let t0 = Instant::now();
    let com = MercuryPCS::<E>::commit(&ck, &poly)?;
    let commit_ms = t0.elapsed().as_secs_f64() * 1e3;
    calls[2] += 1;

    // core_open: no statement recommit (caller supplies C_f).
    let t0 = Instant::now();
    let (proof, value) = MercuryPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
    let core_open_ms = t0.elapsed().as_secs_f64() * 1e3;
    calls[3] += 1;

    // trait_open: includes the extra N-MSM recommit of C_f.
    let t0 = Instant::now();
    let _ = MercuryPCS::<E>::open(&ck, &poly, &point)?;
    let trait_open_ms = t0.elapsed().as_secs_f64() * 1e3;
    calls[4] += 1;

    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let ok = MercuryPCS::<E>::verify(&vk, &com, &point, &value, &proof)?;
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
        assert!(ok, "Mercury detail: verify returned false at nv={nv}");
    }
    assert_eq!(
        calls, [1usize; 5],
        "each heavy phase (srs, trim, commit, core_open, trait_open) must run exactly once"
    );
    let (mean, median) = mean_median(samples);

    // The two dominant N-sized prover MSMs and the analytic core-prover total.
    let msm_q_len = b * b_row.saturating_sub(1);
    let msm_quot_f_len = n - 1;
    let core_scalar_total = msm_q_len + msm_quot_f_len + 3 * b + (b - 1) + 2 * (b - 1);

    println!(
        "{nv},{n},{b},{b_row},{srs_g1},{srs_g2},{pk_bytes},{vk_bytes},{commit_ms:.3},\
         {core_open_ms:.3},{trait_open_ms:.3},{mean:.3},{median:.3},{},{msm_q_len},\
         {msm_quot_f_len},{core_scalar_total}",
        compressed_len(&proof)
    );
    Ok(())
}
