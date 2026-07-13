// Dedicated Mercury single-open benchmark.
//
// Reports, per `nv`, for every single-open backend (mKZG, Gemini, ReciPCS,
// Samaritan, Zeromorph, NestedGridKZG, Mercury; MulcsClaymore as a legacy
// engineering row): srs_gen, trim, commit, open (trait), verify, and compressed
// proof bytes. For Mercury it additionally reports `core_open`
// (`open_with_commitment`, no statement recommit) vs `trait_open`, the tight
// SRS G1/G2 element counts, and the two dominant N-sized prover MSM lengths
// plus the analytic `2N + O(sqrt N)` core-prover scalar total.
//
// Measurement discipline: setup/trim/commit/core_open/trait_open are each timed
// ONCE and their objects reused; only verify is repeated
// (MERCURY_VERIFY_REPETITIONS, default 100). For exact per-MSM scalar counts
// set PCS_PROFILE=1 (the internal ScopedTimers then emit real `count` columns).
//
// Env vars:
//   MERCURY_BENCH_NV_RANGE   comma separated, default 8,10,12,14,16,18,20
//   MERCURY_VERIFY_REPETITIONS   default 100
//
// Run each large nv in its OWN process to avoid accumulating SRS peak memory.

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::UniformRand;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::CanonicalSerialize;
use ark_std::{sync::Arc, test_rng};
use std::{env, time::Instant};
use subroutines::pcs::{
    prelude::{
        GeminiPCS, MercuryPCS, MulcsPCS, MultilinearKzgPCS, NestedGridKzgPCS, PCSError, ReciPCS,
        SamaritanPCS, ZeromorphPCS,
    },
    PolynomialCommitmentScheme,
};

type E = Bls12_381;

const DEFAULT_NV: [usize; 7] = [8, 10, 12, 14, 16, 18, 20];

fn parse_nv() -> Vec<usize> {
    match env::var("MERCURY_BENCH_NV_RANGE") {
        Ok(raw) => raw
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect(),
        Err(_) => DEFAULT_NV.to_vec(),
    }
}

fn verify_reps() -> usize {
    env::var("MERCURY_VERIFY_REPETITIONS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(100)
}

fn main() -> Result<(), PCSError> {
    let nv_list = parse_nv();
    let reps = verify_reps();
    println!("# Mercury dedicated single-open benchmark (BLS12-381)");
    println!("# verify = mean over {reps} reps; all other phases timed once and reused");
    println!("# --- comparison table ---");
    println!("backend,nv,srs_gen_ms,trim_ms,commit_ms,open_ms,verify_us,proof_bytes");
    for &nv in &nv_list {
        run_compare::<MultilinearKzgPCS<E>>("mKZG", nv, reps)?;
        run_compare::<GeminiPCS<E>>("Gemini", nv, reps)?;
        run_compare::<ReciPCS<E>>("ReciPCS", nv, reps)?;
        run_compare::<SamaritanPCS<E>>("Samaritan", nv, reps)?;
        run_compare::<ZeromorphPCS<E>>("Zeromorph", nv, reps)?;
        run_compare::<NestedGridKzgPCS<E>>("NestedGridKZG", nv, reps)?;
        run_compare::<MercuryPCS<E>>("Mercury", nv, reps)?;
        // Legacy engineering row (not used in the paper's main comparison).
        run_compare::<MulcsPCS<E>>("MulcsClaymore", nv, reps)?;
    }

    println!("# --- mercury detail ---");
    println!(
        "nv,N,b,b_row,srs_g1,srs_g2,commit_ms,core_open_ms,trait_open_ms,verify_us,proof_bytes,\
         msm_q_len,msm_quot_f_len,core_scalar_total_2N_plus_sqrt"
    );
    for &nv in &nv_list {
        mercury_detail(nv, reps)?;
    }
    Ok(())
}

fn run_compare<PCS>(name: &str, nv: usize, reps: usize) -> Result<(), PCSError>
where
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<Fr>>,
        Point = Vec<Fr>,
        Evaluation = Fr,
    >,
    PCS::Proof: CanonicalSerialize,
{
    let mut rng = test_rng();
    let t = Instant::now();
    let srs = PCS::gen_srs_for_testing(&mut rng, nv)?;
    let srs_gen_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let (ck, vk) = PCS::trim(&srs, None, Some(nv))?;
    let trim_ms = t.elapsed().as_secs_f64() * 1e3;

    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

    let t = Instant::now();
    let com = PCS::commit(&ck, &poly)?;
    let commit_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let (proof, value) = PCS::open(&ck, &poly, &point)?;
    let open_ms = t.elapsed().as_secs_f64() * 1e3;

    assert!(PCS::verify(&vk, &com, &point, &value, &proof)?);

    let t = Instant::now();
    for _ in 0..reps {
        assert!(PCS::verify(&vk, &com, &point, &value, &proof)?);
    }
    let verify_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

    let mut bytes = Vec::new();
    proof.serialize_compressed(&mut bytes).unwrap();

    println!(
        "{name},{nv},{srs_gen_ms:.3},{trim_ms:.3},{commit_ms:.3},{open_ms:.3},{verify_us:.3},{}",
        bytes.len()
    );
    Ok(())
}

fn mercury_detail(nv: usize, reps: usize) -> Result<(), PCSError> {
    let mut rng = test_rng();
    let n = 1usize << nv;
    let t = nv.div_ceil(2);
    let b = 1usize << t;
    let b_row = 1usize << (nv - t);

    let srs = MercuryPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
    let (ck, vk) = MercuryPCS::<E>::trim(&srs, None, Some(nv))?;
    let srs_g1 = ck.g1_powers.len();
    let srs_g2 = 2usize; // exactly [1]_2, [tau]_2

    let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

    let t0 = Instant::now();
    let com = MercuryPCS::<E>::commit(&ck, &poly)?;
    let commit_ms = t0.elapsed().as_secs_f64() * 1e3;

    // core_open: no statement recommit (caller supplies C_f).
    let t0 = Instant::now();
    let (proof, value) = MercuryPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
    let core_open_ms = t0.elapsed().as_secs_f64() * 1e3;

    // trait_open: includes the extra N-MSM recommit of C_f.
    let t0 = Instant::now();
    let _ = MercuryPCS::<E>::open(&ck, &poly, &point)?;
    let trait_open_ms = t0.elapsed().as_secs_f64() * 1e3;

    assert!(MercuryPCS::<E>::verify(&vk, &com, &point, &value, &proof)?);
    let t0 = Instant::now();
    for _ in 0..reps {
        assert!(MercuryPCS::<E>::verify(&vk, &com, &point, &value, &proof)?);
    }
    let verify_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;

    let mut bytes = Vec::new();
    proof.serialize_compressed(&mut bytes).unwrap();

    // The two dominant N-sized prover MSMs.
    let msm_q_len = b * b_row.saturating_sub(1);
    let msm_quot_f_len = n - 1;
    // Core-prover scalar total: 2N-scale dominant MSMs + six O(sqrt N) MSMs.
    let core_scalar_total = msm_q_len + msm_quot_f_len + 3 * b + (b - 1) + 2 * (b - 1);

    println!(
        "{nv},{n},{b},{b_row},{srs_g1},{srs_g2},{commit_ms:.3},{core_open_ms:.3},\
         {trait_open_ms:.3},{verify_us:.3},{},{msm_q_len},{msm_quot_f_len},{core_scalar_total}",
        bytes.len()
    );
    Ok(())
}
