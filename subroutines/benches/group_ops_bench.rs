// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

#[macro_use]
extern crate criterion;

use ark_bls12_381::Bls12_381;
use ark_ec::{pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, CurveGroup};
use ark_ff::UniformRand;
use ark_std::test_rng;
use criterion::{black_box, BenchmarkId, Criterion};

type E = Bls12_381;
type Fr = <E as Pairing>::ScalarField;
type G1 = <E as Pairing>::G1;
type G1Affine = <E as Pairing>::G1Affine;
type G2 = <E as Pairing>::G2;

fn bench_small_g1_msm(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut group = c.benchmark_group("small_g1_msm");

    for size in 10usize..=25 {
        let bases = (0..size)
            .map(|_| G1::rand(&mut rng).into_affine())
            .collect::<Vec<G1Affine>>();
        let scalars = (0..size).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                black_box(G1::msm_unchecked(
                    black_box(bases.as_slice()),
                    black_box(scalars.as_slice()),
                ))
            })
        });
    }

    group.finish();
}

fn bench_g2_scalar_mul(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut group = c.benchmark_group("g2_scalar_mul");

    let base = G2::rand(&mut rng);
    let scalar = Fr::rand(&mut rng);
    group.bench_function("one", |b| {
        b.iter(|| black_box(black_box(base) * black_box(scalar)))
    });

    let base_a = G2::rand(&mut rng);
    let base_b = G2::rand(&mut rng);
    let scalar_a = Fr::rand(&mut rng);
    let scalar_b = Fr::rand(&mut rng);
    group.bench_function("two_plus_add", |b| {
        b.iter(|| {
            let a = black_box(base_a) * black_box(scalar_a);
            let b = black_box(base_b) * black_box(scalar_b);
            black_box(a + b)
        })
    });

    group.finish();
}

fn bench_group_ops(c: &mut Criterion) {
    bench_small_g1_msm(c);
    bench_g2_scalar_mul(c);
}

criterion_group!(benches, bench_group_ops);
criterion_main!(benches);
