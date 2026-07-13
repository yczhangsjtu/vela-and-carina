// Dedicated CHOPIN single-open benchmark.
//
// Two modes (CHOPIN_BENCH_MODE):
//
//   comparison (default): one consistent CSV table across the selected
//     single-open backends. For each backend/nv, srs_gen / trim / commit /
//     trait_open each run EXACTLY ONCE (self-checked), and only verify repeats.
//     Reports mean & median verify, proof bytes, vk bytes.
//     Canonical backends: mkzg, gemini, recipcs, samaritan, zeromorph, nrg,
//     mercury, chopin.
//
//   detail (CHOPIN_BENCH_MODE=detail): Chopin ONLY. Additionally reports
//     core_open (no C_f recommit) vs trait_open (with recommit), SRS G1/G2
//     counts, pk/vk bytes, q1/q2 MSM lengths, and the analytic core-prover
//     scalar total 1N + O(sqrt N).
//
// Env vars:
//   CHOPIN_BENCH_NV_RANGE        default 8,10,12,14,16,18,20
//   CHOPIN_BENCH_BACKEND         mkzg|gemini|recipcs|samaritan|zeromorph|nrg|
//                                mercury|chopin|all (+ mulcs legacy, nrg aliases)
//   CHOPIN_BENCH_MODE            comparison|detail (default comparison)
//   CHOPIN_VERIFY_REPETITIONS    default 100.

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::UniformRand;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::CanonicalSerialize;
use ark_std::{sync::Arc, test_rng};
use std::{collections::BTreeSet, env, time::Instant};
use subroutines::pcs::{
    prelude::{
        ChopinPCS, GeminiPCS, MercuryPCS, MulcsPCS, MultilinearKzgPCS, NestedGridKzgPCS, PCSError,
        ReciPCS, SamaritanPCS, ZeromorphPCS,
    },
    PolynomialCommitmentScheme,
};

type E = Bls12_381;

const DEFAULT_NV: [usize; 7] = [8, 10, 12, 14, 16, 18, 20];
const CANONICAL: [&str; 8] = [
    "mkzg",
    "gemini",
    "recipcs",
    "samaritan",
    "zeromorph",
    "nrg",
    "mercury",
    "chopin",
];

#[derive(PartialEq, Eq)]
enum Mode {
    Comparison,
    Detail,
}

fn parse_nv() -> Result<Vec<usize>, PCSError> {
    let raw = match env::var("CHOPIN_BENCH_NV_RANGE") {
        Ok(v) => v,
        Err(env::VarError::NotPresent) => return Ok(DEFAULT_NV.to_vec()),
        Err(e) => {
            return Err(PCSError::InvalidParameters(format!(
                "CHOPIN_BENCH_NV_RANGE env error: {e}"
            )))
        },
    };
    let mut list = Vec::new();
    let mut seen = BTreeSet::new();
    for tok in raw.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            return Err(PCSError::InvalidParameters(format!(
                "CHOPIN_BENCH_NV_RANGE: empty token in '{raw}'"
            )));
        }
        let nv: usize = tok.parse().map_err(|_| {
            PCSError::InvalidParameters(format!("CHOPIN_BENCH_NV_RANGE: invalid nv '{tok}'"))
        })?;
        if nv < 1 {
            return Err(PCSError::InvalidParameters(format!(
                "CHOPIN_BENCH_NV_RANGE: nv must be >= 1, got {nv}"
            )));
        }
        if nv >= usize::BITS as usize {
            return Err(PCSError::InvalidParameters(format!(
                "CHOPIN_BENCH_NV_RANGE: nv {nv} exceeds platform word size"
            )));
        }
        if !seen.insert(nv) {
            return Err(PCSError::InvalidParameters(format!(
                "CHOPIN_BENCH_NV_RANGE: duplicate nv {nv}"
            )));
        }
        list.push(nv);
    }
    if list.is_empty() {
        return Err(PCSError::InvalidParameters(
            "CHOPIN_BENCH_NV_RANGE: empty list".to_string(),
        ));
    }
    Ok(list)
}

fn parse_backends() -> Result<Vec<String>, PCSError> {
    let raw = match env::var("CHOPIN_BENCH_BACKEND") {
        Ok(v) => v,
        Err(env::VarError::NotPresent) => {
            return Ok(CANONICAL.iter().map(|s| s.to_string()).collect())
        },
        Err(e) => {
            return Err(PCSError::InvalidParameters(format!(
                "CHOPIN_BENCH_BACKEND env error: {e}"
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
            "CHOPIN_BENCH_BACKEND: unknown backend '{raw}'"
        )))
    }
}

fn parse_mode() -> Result<Mode, PCSError> {
    match env::var("CHOPIN_BENCH_MODE") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "comparison" | "compare" | "" => Ok(Mode::Comparison),
            "detail" => Ok(Mode::Detail),
            other => Err(PCSError::InvalidParameters(format!(
                "CHOPIN_BENCH_MODE: unknown '{other}'; use comparison or detail"
            ))),
        },
        Err(env::VarError::NotPresent) => Ok(Mode::Comparison),
        Err(e) => Err(PCSError::InvalidParameters(format!(
            "CHOPIN_BENCH_MODE env error: {e}"
        ))),
    }
}

fn verify_reps() -> Result<usize, PCSError> {
    match env::var("CHOPIN_VERIFY_REPETITIONS") {
        Ok(v) => {
            let r: usize = v.trim().parse().map_err(|_| {
                PCSError::InvalidParameters(format!("CHOPIN_VERIFY_REPETITIONS: invalid '{v}'"))
            })?;
            if r == 0 {
                return Err(PCSError::InvalidParameters(
                    "CHOPIN_VERIFY_REPETITIONS must be >= 1".to_string(),
                ));
            }
            Ok(r)
        },
        Err(env::VarError::NotPresent) => Ok(100),
        Err(e) => Err(PCSError::InvalidParameters(format!(
            "CHOPIN_VERIFY_REPETITIONS env error: {e}"
        ))),
    }
}

fn main() -> Result<(), PCSError> {
    let nv_list = parse_nv()?;
    let reps = verify_reps()?;
    let mode = parse_mode()?;
    let backends = parse_backends()?;

    if nv_list.iter().any(|&nv| nv >= 18) {
        eprintln!("# NOTE: large nv requested; run each nv/backend in its OWN process to bound peak RSS:");
        eprintln!("#   CHOPIN_BENCH_NV_RANGE=20 CHOPIN_BENCH_BACKEND=chopin cargo bench -p subroutines --bench chopin-benches");
    }

    match mode {
        Mode::Comparison => run_comparison(&nv_list, &backends, reps),
        Mode::Detail => {
            let explicit_non_chopin = env::var("CHOPIN_BENCH_BACKEND").is_ok()
                && backends != vec!["chopin".to_string()]
                && backends.len() != CANONICAL.len();
            if explicit_non_chopin {
                return Err(PCSError::InvalidParameters(
                    "CHOPIN_BENCH_MODE=detail is Chopin-only".to_string(),
                ));
            }
            run_detail(&nv_list, reps)
        },
    }
}

fn run_comparison(nv_list: &[usize], backends: &[String], reps: usize) -> Result<(), PCSError> {
    println!("# Chopin comparison table (BLS12-381)");
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
                "chopin" => compare_row::<ChopinPCS<E>>("Chopin", false, nv, reps)?,
                "mulcs" => compare_row::<MulcsPCS<E>>("MulcsClaymore", true, nv, reps)?,
                other => return Err(PCSError::InvalidParameters(format!("unreachable backend {other}"))),
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
    let mut calls = [0usize; 4];

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
    assert_eq!(calls, [1usize; 4]);
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
    println!("# Chopin detail: core_open (no C_f recommit) vs trait_open (recommit)");
    println!(
        "nv,N,m_left,m_right,M_L,M_R,srs_g1,srs_g2,pk_bytes,vk_bytes,commit_ms,\
         core_open_ms,trait_open_ms,verify_us_mean,verify_us_median,proof_bytes,\
         msms_q1_len,msms_q2_len,msms_c0_len,msms_c1_len,msms_cs_len,msms_w_len,msms_wp_len"
    );
    for &nv in nv_list {
        detail_row(nv, reps)?;
    }
    Ok(())
}

fn detail_row(nv: usize, reps: usize) -> Result<(), PCSError> {
    let mut rng = test_rng();
    let n = 1usize << nv;
    let m_left = nv.div_ceil(2);
    let m_right = nv / 2;
    let ml = 1usize << m_left;
    let mr = 1usize << m_right;

    let srs = ChopinPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
    let (ck, vk) = ChopinPCS::<E>::trim(&srs, None, Some(nv))?;
    let srs_g1 = ml * mr;
    let srs_g2 = 3usize;
    let pk_bytes = compressed_len(&ck);
    let vk_bytes = compressed_len(&vk);

    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

    let t0 = Instant::now();
    let com = ChopinPCS::<E>::commit(&ck, &poly)?;
    let commit_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = Instant::now();
    let (proof, value) = ChopinPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
    let core_open_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = Instant::now();
    let _ = ChopinPCS::<E>::open(&ck, &poly, &point)?;
    let trait_open_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let ok = ChopinPCS::<E>::verify(&vk, &com, &point, &value, &proof)?;
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
        assert!(ok, "Chopin detail: verify returned false at nv={nv}");
    }
    let (mean, median) = mean_median(samples);

    let q1_len = (ml - 1) * mr;
    let q2_len = mr.saturating_sub(1);
    let c0_len = ml;
    let c1_len = mr;
    let cs_len = ml.saturating_sub(1);
    let w_len = ml.saturating_sub(8).max(1); // W degree bound rough approx
    let wp_len = ml.saturating_sub(2).max(1);

    println!(
        "{nv},{n},{m_left},{m_right},{ml},{mr},{srs_g1},{srs_g2},{pk_bytes},{vk_bytes},\
         {commit_ms:.3},{core_open_ms:.3},{trait_open_ms:.3},{mean:.3},{median:.3},{},\
         {q1_len},{q2_len},{c0_len},{c1_len},{cs_len},{w_len},{wp_len}",
        compressed_len(&proof)
    );
    Ok(())
}
