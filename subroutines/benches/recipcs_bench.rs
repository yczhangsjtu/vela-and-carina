// ReciPCS unified single-PCS benchmark.
//
// Compares mKZG, Gemini(+Shplonk), Zeromorph, Samaritan, and ReciPCS on a
// single commit/open/verify over BLS12-381. Each backend should be run in its
// OWN process (via PCS_BENCH_BACKEND) so that allocator state, cache, and SRS
// peak memory do not leak across backends.
//
// Measurement discipline:
//   - srs_gen / trim / commit / open : timed once each
//   - verify : timed as the mean over VERIFY_REPS repetitions, and the median
//   - proof size : compressed serialized bytes
// Output is CSV to stdout (one row per nv) plus a header describing the env.

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::UniformRand;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::CanonicalSerialize;
use ark_std::{sync::Arc, test_rng};
use std::{env, time::Instant};
use subroutines::pcs::{
    prelude::{GeminiPCS, MultilinearKzgPCS, PCSError, ReciPCS, SamaritanPCS, ZeromorphPCS},
    PolynomialCommitmentScheme,
};

const DEFAULT_NV: [usize; 7] = [8, 10, 12, 14, 16, 18, 20];
const VERIFY_REPS: usize = 100;
const BACKENDS: [&str; 5] = ["mkzg", "gemini", "zeromorph", "samaritan", "recipcs"];

fn parse_nv() -> Vec<usize> {
    match env::var("RECIPCS_NV") {
        Ok(raw) => raw
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect(),
        Err(_) => DEFAULT_NV.to_vec(),
    }
}

fn parse_backends() -> Vec<String> {
    match env::var("RECIPCS_BACKEND") {
        Ok(raw) => {
            let sel = raw.trim().to_ascii_lowercase();
            if sel == "all" {
                BACKENDS.iter().map(|s| s.to_string()).collect()
            } else {
                vec![sel]
            }
        },
        Err(_) => BACKENDS.iter().map(|s| s.to_string()).collect(),
    }
}

fn main() -> Result<(), PCSError> {
    let nv_list = parse_nv();
    let backends = parse_backends();
    println!("# ReciPCS unified single-PCS benchmark (BLS12-381)");
    println!("# verify = mean of {VERIFY_REPS} reps; other phases timed once");
    println!("backend,nv,srs_gen_ms,trim_ms,commit_ms,open_ms,verify_us_mean,verify_us_median,proof_bytes");
    for &nv in &nv_list {
        for b in &backends {
            match b.as_str() {
                "mkzg" => run::<MultilinearKzgPCS<Bls12_381>>("mkzg", nv)?,
                "gemini" => run::<GeminiPCS<Bls12_381>>("gemini", nv)?,
                "zeromorph" => run::<ZeromorphPCS<Bls12_381>>("zeromorph", nv)?,
                "samaritan" => run::<SamaritanPCS<Bls12_381>>("samaritan", nv)?,
                "recipcs" => run::<ReciPCS<Bls12_381>>("recipcs", nv)?,
                other => eprintln!("unknown backend {other}"),
            }
        }
    }
    Ok(())
}

fn run<PCS>(name: &str, nv: usize) -> Result<(), PCSError>
where
    PCS: PolynomialCommitmentScheme<
        Bls12_381,
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

    assert!(
        PCS::verify(&vk, &com, &point, &value, &proof)?,
        "{name}: honest proof failed to verify"
    );

    let mut samples = Vec::with_capacity(VERIFY_REPS);
    for _ in 0..VERIFY_REPS {
        let t = Instant::now();
        let ok = PCS::verify(&vk, &com, &point, &value, &proof)?;
        samples.push(t.elapsed().as_secs_f64() * 1e6);
        assert!(ok);
    }
    let mean = samples.iter().sum::<f64>() / VERIFY_REPS as f64;
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[VERIFY_REPS / 2];

    let mut proof_bytes = Vec::new();
    proof.serialize_compressed(&mut proof_bytes).unwrap();

    println!(
        "{name},{nv},{srs_gen_ms:.3},{trim_ms:.3},{commit_ms:.3},{open_ms:.3},{mean:.3},{median:.3},{}",
        proof_bytes.len()
    );
    Ok(())
}
