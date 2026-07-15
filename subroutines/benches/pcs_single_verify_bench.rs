// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

#[macro_use]
extern crate criterion;

use ark_bls12_381::{Bls12_381, Fr};
use ark_poly::DenseMultilinearExtension;
use ark_std::{rand::Rng, sync::Arc, test_rng, UniformRand};
use criterion::{
    black_box, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use std::{env, time::Duration};
use subroutines::pcs::{
    prelude::{
        CarinaPCS, ChopinPCS, GeminiPCS, MercuryPCS, MulcsPCS, MultilinearKzgPCS, PCSError,
        SamaritanPCS, VelaPCS, ZeromorphPCS,
    },
    PolynomialCommitmentScheme,
};

type E = Bls12_381;

const DEFAULT_NV_RANGE: [usize; 7] = [8, 10, 12, 14, 16, 18, 20];
const ALL_BACKENDS: [&str; 9] = [
    "mkzg",
    "gemini",
    "mulcs",
    "samaritan",
    "zeromorph",
    "vela",
    "carina",
    "mercury",
    "chopin",
];

/// Parse `PCS_VERIFY_NV_RANGE` (comma separated), default 8..=20 step 2.
fn parse_nv_range() -> Vec<usize> {
    match env::var("PCS_VERIFY_NV_RANGE") {
        Ok(raw) => raw
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect(),
        Err(_) => DEFAULT_NV_RANGE.to_vec(),
    }
}

/// Parse `PCS_VERIFY_BACKEND`, defaulting to all public backend names.
fn parse_backends() -> Vec<String> {
    let raw = match env::var("PCS_VERIFY_BACKEND") {
        Ok(v) => v,
        Err(_) => {
            return ALL_BACKENDS.iter().map(|s| s.to_string()).collect();
        },
    };
    let sel = raw.trim().to_ascii_lowercase();
    if sel == "all" {
        return ALL_BACKENDS.iter().map(|s| s.to_string()).collect();
    }
    let canonical = sel;
    if ALL_BACKENDS.contains(&canonical.as_str()) {
        vec![canonical]
    } else {
        panic!(
            "PCS_VERIFY_BACKEND: unsupported '{raw}'; use one of {:?} or all",
            ALL_BACKENDS
        );
    }
}

struct VerifyInstance<PCS>
where
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<Fr>>,
        Point = Vec<Fr>,
        Evaluation = Fr,
    >,
{
    verifier_param: PCS::VerifierParam,
    commitment: PCS::Commitment,
    point: Vec<Fr>,
    value: Fr,
    proof: PCS::Proof,
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

fn prepare_instance<PCS, R>(rng: &mut R, nv: usize) -> Result<VerifyInstance<PCS>, PCSError>
where
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<Fr>>,
        Point = Vec<Fr>,
        Evaluation = Fr,
    >,
    R: Rng,
{
    let (poly, point) = random_poly_and_point(rng, nv);
    let srs = PCS::gen_srs_for_testing(rng, nv)?;
    let (prover_param, verifier_param) = PCS::trim(&srs, None, Some(nv))?;
    let commitment = PCS::commit(&prover_param, &poly)?;
    let (proof, value) = PCS::open(&prover_param, &poly, &point)?;
    assert!(PCS::verify(
        &verifier_param,
        &commitment,
        &point,
        &value,
        &proof
    )?);
    Ok(VerifyInstance {
        verifier_param,
        commitment,
        point,
        value,
        proof,
    })
}

fn bench_backend<PCS>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    rng: &mut impl Rng,
    backend: &str,
    nv: usize,
) -> Result<(), PCSError>
where
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<Fr>>,
        Point = Vec<Fr>,
        Evaluation = Fr,
    >,
{
    // Setup and proof generation are OUTSIDE `b.iter`; only verify is timed.
    let instance = prepare_instance::<PCS, _>(rng, nv)?;
    let n = 1u64 << nv;
    group.throughput(Throughput::Elements(1));
    group.bench_with_input(BenchmarkId::new(backend, nv), &n, |b, _| {
        b.iter(|| {
            let ok = PCS::verify(
                black_box(&instance.verifier_param),
                black_box(&instance.commitment),
                black_box(&instance.point),
                black_box(&instance.value),
                black_box(&instance.proof),
            )
            .expect("PCS verify should not error");
            assert!(ok, "PCS verify returned false");
            black_box(ok)
        })
    });
    Ok(())
}

fn bench_pcs_single_verify(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut group = c.benchmark_group("pcs_single_verify");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let nv_range = parse_nv_range();
    let backends = parse_backends();

    // Only the selected backends are instantiated, so filtering to e.g. `carina`
    // avoids pre-generating every other backend's large (nv=20) SRS.
    for nv in nv_range {
        for backend in &backends {
            match backend.as_str() {
                "mkzg" => bench_backend::<MultilinearKzgPCS<E>>(&mut group, &mut rng, "mKZG", nv),
                "gemini" => bench_backend::<GeminiPCS<E>>(&mut group, &mut rng, "Gemini", nv),
                "mulcs" => bench_backend::<MulcsPCS<E>>(&mut group, &mut rng, "MulcsClaymore", nv),
                "samaritan" => {
                    bench_backend::<SamaritanPCS<E>>(&mut group, &mut rng, "Samaritan", nv)
                },
                "zeromorph" => {
                    bench_backend::<ZeromorphPCS<E>>(&mut group, &mut rng, "Zeromorph", nv)
                },
                "vela" => bench_backend::<VelaPCS<E>>(&mut group, &mut rng, "Vela", nv),
                "carina" => bench_backend::<CarinaPCS<E>>(&mut group, &mut rng, "Carina", nv),
                "mercury" => bench_backend::<MercuryPCS<E>>(&mut group, &mut rng, "Mercury", nv),
                "chopin" => bench_backend::<ChopinPCS<E>>(&mut group, &mut rng, "Chopin", nv),
                other => panic!("unreachable backend {other}"),
            }
            .unwrap_or_else(|e| {
                panic!("{backend} setup/proof generation failed at nv={nv}: {e:?}")
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_pcs_single_verify);
criterion_main!(benches);
