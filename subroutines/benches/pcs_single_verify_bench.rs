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
use std::time::Duration;
use subroutines::pcs::{
    prelude::{
        MulcsPCS, MulcsSymmetricPCS, MultilinearKzgPCS, PCSError, SamaritanPCS, ZeromorphPCS,
    },
    PolynomialCommitmentScheme,
};

type E = Bls12_381;

const NV_RANGE: [usize; 7] = [8, 10, 12, 14, 16, 18, 20];

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

    for nv in NV_RANGE {
        bench_backend::<MultilinearKzgPCS<E>>(&mut group, &mut rng, "mKZG", nv)
            .expect("mKZG setup/proof generation failed");
        bench_backend::<MulcsPCS<E>>(&mut group, &mut rng, "MulcsClaymore", nv)
            .expect("Mulcs setup/proof generation failed");
        bench_backend::<MulcsSymmetricPCS<E>>(&mut group, &mut rng, "MulcsSymmetric", nv)
            .expect("MulcsSymmetric setup/proof generation failed");
        bench_backend::<SamaritanPCS<E>>(&mut group, &mut rng, "Samaritan", nv)
            .expect("Samaritan setup/proof generation failed");
        bench_backend::<ZeromorphPCS<E>>(&mut group, &mut rng, "Zeromorph", nv)
            .expect("Zeromorph setup/proof generation failed");
    }

    group.finish();
}

criterion_group!(benches, bench_pcs_single_verify);
criterion_main!(benches);
