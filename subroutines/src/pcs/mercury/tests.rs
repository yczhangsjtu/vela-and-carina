//! Mercury PCS correctness and negative tests.
//!
//! Covers:
//! - even/odd/min `nv`, random property tests;
//! - the paper polynomial identities (`f = (X^b-alpha)q + g`, structured `S` vs
//!   dense reference, the `S` symmetric-Laurent identity at `z`/`z^-1`);
//! - **coefficient-level BDFG20 algebra** (`m == Z_T * W`, `L == (X-z) * W'`,
//!   zero remainders, `W`/`W'` commitment equality, and the verifier's
//!   homomorphic reconstruction `lhs_1 == [tau W'(tau)]`, `lhs_2 ==
//!   [W'(tau)]`), plus BDFG20 negative cases (inconsistent evals break
//!   divisibility; a wrong challenge breaks the verifier reconstruction) —
//!   these do NOT rely on the end-to-end verify to hide a symbolic error;
//! - **odd-`nv` rectangular-split invariants** (`mu = 3,5,7`): matrix
//!   restriction vs multilinear evaluation, the fold identity, the `S`
//!   identity, and a differential check that the rectangular `g,h` equal the
//!   Nova-style zero-padded *square* layout on the original coefficients;
//! - every negative case (wrong value / point / commitment, tampering each
//!   proof field including the two BDFG20 witnesses, swapped evaluations),
//!   malformed `mu`/lengths, vk/pk capacity, statement binding, serialization
//!   roundtrip, proof-size assertion, and `catch_unwind` panic-freedom.
//!   Sumcheck batch adapter tests exercise `multi_open` / `batch_verify`.

use super::*;
use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::{One, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::{panic, test_rng, UniformRand};

type E = Bls12_381;

fn setup(nv: usize) -> (MercuryProverParam<E>, MercuryVerifierParam<E>) {
    let mut rng = test_rng();
    let srs = MercuryPCS::<E>::gen_srs_for_testing(&mut rng, nv).unwrap();
    MercuryPCS::<E>::trim(&srs, None, Some(nv)).unwrap()
}

fn rand_point(nv: usize, rng: &mut impl Rng) -> Vec<Fr> {
    (0..nv).map(|_| Fr::rand(rng)).collect()
}

fn rand_poly(nv: usize, rng: &mut impl Rng) -> Arc<DenseMultilinearExtension<Fr>> {
    Arc::new(DenseMultilinearExtension::rand(nv, rng))
}

// ════════════════════════════════════════════════════════════════════
// Positive: even / odd / minimum nv, random property tests
// ════════════════════════════════════════════════════════════════════

#[test]
fn open_verify_even_and_odd() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [2usize, 3, 4, 5, 6, 7, 8] {
        let (ck, vk) = setup(nv);
        let p = rand_poly(nv, &mut rng);
        let pt = rand_point(nv, &mut rng);
        let com = MercuryPCS::<E>::commit(&ck, &p)?;
        let (proof, val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
        assert_eq!(proof.mu, nv);
        assert!(
            MercuryPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?,
            "open/verify failed at nv={nv}"
        );
    }
    Ok(())
}

#[test]
fn open_verify_minimum_nv() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [1usize, 2] {
        let (ck, vk) = setup(nv);
        let p = rand_poly(nv, &mut rng);
        let pt = rand_point(nv, &mut rng);
        let com = MercuryPCS::<E>::commit(&ck, &p)?;
        let (proof, val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
        assert!(MercuryPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
    }
    Ok(())
}

#[test]
fn open_verify_random_property() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [4usize, 5, 6] {
        let (ck, vk) = setup(nv);
        for _ in 0..8 {
            let p = rand_poly(nv, &mut rng);
            let pt = rand_point(nv, &mut rng);
            let com = MercuryPCS::<E>::commit(&ck, &p)?;
            let (proof, val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
            assert_eq!(val, p.evaluate(&pt).unwrap());
            assert!(MercuryPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        }
    }
    Ok(())
}

#[test]
fn open_with_commitment_matches_trait_open() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [4usize, 5] {
        let (ck, vk) = setup(nv);
        let p = rand_poly(nv, &mut rng);
        let pt = rand_point(nv, &mut rng);
        let com = MercuryPCS::<E>::commit(&ck, &p)?;
        let (proof, val) = MercuryPCS::<E>::open_with_commitment(&ck, &p, &pt, &com)?;
        assert!(MercuryPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        // Same transcript => identical proof to the trait open.
        let (proof2, val2) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
        assert_eq!(val, val2);
        assert_eq!(proof, proof2);
    }
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Paper polynomial identities (coefficient-level)
// ════════════════════════════════════════════════════════════════════

#[test]
fn identity_divide_by_binomial() {
    let mut rng = test_rng();
    for mu in [2usize, 3, 4, 5, 6] {
        let (t, b, b_row, n) = mercury_dims(mu).unwrap();
        let coeffs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let alpha = Fr::rand(&mut rng);
        let (g, q) = divide_by_binomial(&coeffs, b, b_row, alpha);
        assert_eq!(g.len(), b);
        // f(r) == (r^b - alpha) q(r) + g(r) for random r
        for _ in 0..4 {
            let r = Fr::rand(&mut rng);
            let f_r = poly_eval(&coeffs, r);
            let q_r = poly_eval(&q, r);
            let g_r = poly_eval(&g, r);
            assert_eq!(f_r, (r.pow([b as u64]) - alpha) * q_r + g_r, "mu={mu}");
        }
        let _ = t;
    }
}

/// Dense O(b^2) reference for the symmetric-Laurent `S(X)`.
fn make_s_dense(g: &[Fr], h: &[Fr], u1: &[Fr], u2_full: &[Fr], b: usize, gamma: Fr) -> Vec<Fr> {
    let pu1 = build_eq_vec::<Fr>(u1, b);
    let pu2 = build_eq_vec::<Fr>(u2_full, b);
    let off = b - 1;
    let laurent = |coeffs: &[Fr], pu: &[Fr]| -> Vec<Fr> {
        let mut c = vec![Fr::zero(); 2 * b - 1];
        for (i, &ci) in coeffs.iter().enumerate().take(b) {
            for (j, &pj) in pu.iter().enumerate().take(b) {
                let idx = (off as isize + i as isize - j as isize) as usize;
                c[idx] += ci * pj;
            }
        }
        c
    };
    let c1 = laurent(g, &pu1);
    let c2 = laurent(h, &pu2);
    let mut s = vec![Fr::zero(); b - 1];
    for (k1, sk) in s.iter_mut().enumerate() {
        let k = k1 + 1;
        *sk = (c1[off + k] + c1[off - k]) + gamma * (c2[off + k] + c2[off - k]);
    }
    s
}

#[test]
fn identity_structured_s_matches_dense() {
    let mut rng = test_rng();
    for mu in [2usize, 4, 6, 8] {
        let (t, b, b_row, _n) = mercury_dims(mu).unwrap();
        let g: Vec<Fr> = (0..b).map(|_| Fr::rand(&mut rng)).collect();
        let mut h: Vec<Fr> = (0..b_row).map(|_| Fr::rand(&mut rng)).collect();
        h.resize(b, Fr::zero());
        let u1: Vec<Fr> = (0..t).map(|_| Fr::rand(&mut rng)).collect();
        let mut u2_full: Vec<Fr> = (0..(mu - t)).map(|_| Fr::rand(&mut rng)).collect();
        u2_full.resize(t, Fr::zero());
        let gamma = Fr::rand(&mut rng);
        let structured = make_s_polynomial_structured(&g, &h, &u1, &u2_full, t, b, gamma);
        let dense = make_s_dense(&g, &h, &u1, &u2_full, b, gamma);
        assert_eq!(structured, dense, "structured S != dense at mu={mu}");
    }
}

#[test]
fn identity_s_constant_coeff_is_ipa() {
    // The constant Laurent coefficient A_0/2 = <g,Pu1> + gamma <h,Pu2>.
    let mut rng = test_rng();
    let mu = 6;
    let (t, b, b_row, _n) = mercury_dims(mu).unwrap();
    let g: Vec<Fr> = (0..b).map(|_| Fr::rand(&mut rng)).collect();
    let mut h: Vec<Fr> = (0..b_row).map(|_| Fr::rand(&mut rng)).collect();
    h.resize(b, Fr::zero());
    let u1: Vec<Fr> = (0..t).map(|_| Fr::rand(&mut rng)).collect();
    let mut u2_full: Vec<Fr> = (0..(mu - t)).map(|_| Fr::rand(&mut rng)).collect();
    u2_full.resize(t, Fr::zero());
    let pu1 = build_eq_vec::<Fr>(&u1, b);
    let pu2 = build_eq_vec::<Fr>(&u2_full, b);
    let ip_g: Fr = g.iter().zip(pu1.iter()).map(|(a, c)| *a * *c).sum();
    let ip_h: Fr = h.iter().zip(pu2.iter()).map(|(a, c)| *a * *c).sum();
    let gamma = Fr::rand(&mut rng);
    // Reconstruct A_0 from the dense Laurent buffers.
    let dense_full = |coeffs: &[Fr], pu: &[Fr]| -> Fr {
        let mut acc = Fr::zero();
        for (i, &ci) in coeffs.iter().enumerate().take(b) {
            for (j, &pj) in pu.iter().enumerate().take(b) {
                if i == j {
                    acc += ci * pj;
                }
            }
        }
        acc
    };
    let a0_half = dense_full(&g, &pu1) + gamma * dense_full(&h, &pu2);
    assert_eq!(a0_half, ip_g + gamma * ip_h);
}

// ════════════════════════════════════════════════════════════════════
// Negative: wrong value / point / commitment
// ════════════════════════════════════════════════════════════════════

#[test]
fn reject_wrong_value() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [4usize, 5] {
        let (ck, vk) = setup(nv);
        let p = rand_poly(nv, &mut rng);
        let pt = rand_point(nv, &mut rng);
        let com = MercuryPCS::<E>::commit(&ck, &p)?;
        let (proof, _val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
        let bad = Fr::rand(&mut rng);
        assert!(!MercuryPCS::<E>::verify(&vk, &com, &pt, &bad, &proof)?);
    }
    Ok(())
}

#[test]
fn reject_wrong_point() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 5;
    let (ck, vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let com = MercuryPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
    let other = rand_point(nv, &mut rng);
    assert!(!MercuryPCS::<E>::verify(&vk, &com, &other, &val, &proof)?);
    Ok(())
}

#[test]
fn reject_wrong_commitment() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv);
    let p1 = rand_poly(nv, &mut rng);
    let p2 = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let com2 = MercuryPCS::<E>::commit(&ck, &p2)?;
    let (proof, val) = MercuryPCS::<E>::open(&ck, &p1, &pt)?;
    assert!(!MercuryPCS::<E>::verify(&vk, &com2, &pt, &val, &proof)?);
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Negative: tamper each proof field (attack regression)
// ════════════════════════════════════════════════════════════════════

fn tamper_g1(x: &<E as Pairing>::G1Affine) -> <E as Pairing>::G1Affine {
    (x.into_group() * Fr::from(2u64)).into_affine()
}

#[test]
fn reject_tampered_g1_fields() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let com = MercuryPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
    // one mutator per G1 field
    let mutators: Vec<(&str, fn(&mut MercuryProof<E>))> = vec![
        ("comm_h", |pr| pr.comm_h = tamper_g1(&pr.comm_h)),
        ("comm_g", |pr| pr.comm_g = tamper_g1(&pr.comm_g)),
        ("comm_q", |pr| pr.comm_q = tamper_g1(&pr.comm_q)),
        ("comm_s", |pr| pr.comm_s = tamper_g1(&pr.comm_s)),
        ("comm_d", |pr| pr.comm_d = tamper_g1(&pr.comm_d)),
        ("comm_quot_f", |pr| {
            pr.comm_quot_f = tamper_g1(&pr.comm_quot_f)
        }),
        ("comm_w", |pr| pr.comm_w = tamper_g1(&pr.comm_w)),
        ("comm_w_prime", |pr| {
            pr.comm_w_prime = tamper_g1(&pr.comm_w_prime)
        }),
    ];
    for (name, mutate) in mutators {
        let mut bad = proof.clone();
        mutate(&mut bad);
        let res = MercuryPCS::<E>::verify(&vk, &com, &pt, &val, &bad);
        assert!(
            matches!(res, Ok(false)) || res.is_err(),
            "tampering {name} must be rejected"
        );
    }
    Ok(())
}

#[test]
fn reject_tampered_scalar_fields() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let com = MercuryPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
    let mutators: Vec<(&str, fn(&mut MercuryProof<E>))> = vec![
        ("g_zeta", |pr| pr.g_zeta += Fr::one()),
        ("g_zeta_inv", |pr| pr.g_zeta_inv += Fr::one()),
        ("h_zeta", |pr| pr.h_zeta += Fr::one()),
        ("h_zeta_inv", |pr| pr.h_zeta_inv += Fr::one()),
        ("s_zeta", |pr| pr.s_zeta += Fr::one()),
        ("s_zeta_inv", |pr| pr.s_zeta_inv += Fr::one()),
    ];
    for (name, mutate) in mutators {
        let mut bad = proof.clone();
        mutate(&mut bad);
        let res = MercuryPCS::<E>::verify(&vk, &com, &pt, &val, &bad);
        assert!(
            matches!(res, Ok(false)) || res.is_err(),
            "tampering {name} must be rejected"
        );
    }
    Ok(())
}

#[test]
fn reject_swapped_evaluations() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let com = MercuryPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
    let mut bad = proof.clone();
    std::mem::swap(&mut bad.g_zeta, &mut bad.g_zeta_inv);
    let res = MercuryPCS::<E>::verify(&vk, &com, &pt, &val, &bad);
    assert!(matches!(res, Ok(false)) || res.is_err());
    let mut bad2 = proof;
    std::mem::swap(&mut bad2.h_zeta, &mut bad2.h_zeta_inv);
    let res2 = MercuryPCS::<E>::verify(&vk, &com, &pt, &val, &bad2);
    assert!(matches!(res2, Ok(false)) || res2.is_err());
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Negative: malformed mu / point / capacity (no panic)
// ════════════════════════════════════════════════════════════════════

#[test]
fn malformed_mu_no_panic() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let com = MercuryPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
    for bad_mu in [
        0usize,
        usize::BITS as usize,
        u32::MAX as usize,
        nv + 1,
        nv - 1,
    ] {
        let mut bad = proof.clone();
        bad.mu = bad_mu;
        let res = panic::catch_unwind(|| MercuryPCS::<E>::verify(&vk, &com, &pt, &val, &bad));
        assert!(res.is_ok(), "verify panicked for mu={bad_mu}");
        let inner = res.unwrap();
        assert!(
            inner.is_err() || !inner.unwrap(),
            "malformed mu={bad_mu} must be rejected"
        );
    }
    Ok(())
}

#[test]
fn malformed_point_length_no_panic() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let com = MercuryPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
    for bad_len in [0usize, nv - 1, nv + 1, nv + 5] {
        let bad_pt = rand_point(bad_len, &mut rng);
        let res = panic::catch_unwind(|| MercuryPCS::<E>::verify(&vk, &com, &bad_pt, &val, &proof));
        assert!(res.is_ok(), "verify panicked for point len {bad_len}");
        let inner = res.unwrap();
        assert!(inner.is_err() || !inner.unwrap());
    }
    Ok(())
}

#[test]
fn verifier_key_capacity_insufficient() -> Result<(), PCSError> {
    let mut rng = test_rng();
    // vk trimmed for nv=3 but proof claims nv=5.
    let (ck5, _vk5) = setup(5);
    let srs_small = MercuryPCS::<E>::gen_srs_for_testing(&mut rng, 3)?;
    let (_ck3, vk3) = MercuryPCS::<E>::trim(&srs_small, None, Some(3))?;
    let p = rand_poly(5, &mut rng);
    let pt = rand_point(5, &mut rng);
    let com = MercuryPCS::<E>::commit(&ck5, &p)?;
    let (proof, val) = MercuryPCS::<E>::open(&ck5, &p, &pt)?;
    let res = MercuryPCS::<E>::verify(&vk3, &com, &pt, &val, &proof);
    assert!(res.is_err(), "insufficient vk capacity must Err");
    Ok(())
}

#[test]
fn prover_key_too_small() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let srs_small = MercuryPCS::<E>::gen_srs_for_testing(&mut rng, 3)?;
    let (ck3, _vk3) = MercuryPCS::<E>::trim(&srs_small, None, Some(3))?;
    let p = rand_poly(5, &mut rng);
    let pt = rand_point(5, &mut rng);
    // commit and open must Err (not panic) when the key is too small.
    assert!(MercuryPCS::<E>::commit(&ck3, &p).is_err());
    assert!(MercuryPCS::<E>::open(&ck3, &p, &pt).is_err());
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Transcript statement binding
// ════════════════════════════════════════════════════════════════════

#[test]
fn transcript_statement_binding() -> Result<(), PCSError> {
    // A proof produced for (com, pt, val) must not verify against a different
    // commitment even if the value happens to match, because C_f is bound before
    // the first challenge.
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let com = MercuryPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
    // scale commitment: value unchanged in the call but binding differs.
    let other_com = Commitment((com.0.into_group() * Fr::from(3u64)).into_affine());
    assert!(!MercuryPCS::<E>::verify(
        &vk, &other_com, &pt, &val, &proof
    )?);
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Serialization roundtrip + proof-size assertion
// ════════════════════════════════════════════════════════════════════

#[test]
fn proof_serialization_roundtrip() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 5;
    let (ck, _vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let (proof, _val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
    let mut bytes = Vec::new();
    proof.serialize_compressed(&mut bytes).unwrap();
    let back = MercuryProof::<E>::deserialize_compressed(&bytes[..]).unwrap();
    assert_eq!(proof, back);
    Ok(())
}

#[test]
fn proof_size_matches_field_count() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 6;
    let (ck, _vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let (proof, _val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
    let mut bytes = Vec::new();
    proof.serialize_compressed(&mut bytes).unwrap();

    let g1_sz = <<E as Pairing>::G1Affine as CanonicalSerialize>::compressed_size(&proof.comm_h);
    let fr_sz = <Fr as CanonicalSerialize>::compressed_size(&proof.g_zeta);
    let mu_sz = proof.mu.compressed_size();
    // 8 G1 + 6 field elements + mu, exactly (no redundant challenges/remainders).
    let expected = 8 * g1_sz + 6 * fr_sz + mu_sz;
    assert_eq!(
        bytes.len(),
        expected,
        "proof payload must be exactly 8 G1 + 6 F (+mu)"
    );
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Sumcheck batch adapter (multi_open / batch_verify)
// ════════════════════════════════════════════════════════════════════

fn batch_open_verify(
    ck: &MercuryProverParam<E>,
    vk: &MercuryVerifierParam<E>,
    polys: &[Arc<DenseMultilinearExtension<Fr>>],
    points: &[Vec<Fr>],
) -> Result<bool, PCSError> {
    let evals: Vec<Fr> = polys
        .iter()
        .zip(points.iter())
        .map(|(f, p)| f.evaluate(p).unwrap())
        .collect();
    let commitments: Vec<_> = polys
        .iter()
        .map(|poly| MercuryPCS::<E>::commit(ck, poly).unwrap())
        .collect();
    let mut tr = IOPTranscript::<Fr>::new(b"mercury-batch-test");
    tr.append_field_element(b"init", &Fr::zero())?;
    let batch_proof = MercuryPCS::<E>::multi_open(ck, polys, points, &evals, &mut tr)?;
    let mut tr2 = IOPTranscript::<Fr>::new(b"mercury-batch-test");
    tr2.append_field_element(b"init", &Fr::zero())?;
    MercuryPCS::<E>::batch_verify(vk, &commitments, points, &batch_proof, &mut tr2)
}

#[test]
fn batch_k1() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv + 2);
    let polys = vec![rand_poly(nv, &mut rng)];
    let points = vec![rand_point(nv, &mut rng)];
    assert!(batch_open_verify(&ck, &vk, &polys, &points)?);
    Ok(())
}

#[test]
fn batch_multiple_distinct_points() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv + 3);
    let polys: Vec<_> = (0..5).map(|_| rand_poly(nv, &mut rng)).collect();
    let points: Vec<_> = (0..5).map(|_| rand_point(nv, &mut rng)).collect();
    assert!(batch_open_verify(&ck, &vk, &polys, &points)?);
    Ok(())
}

#[test]
fn batch_multiple_same_point() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv + 3);
    let polys: Vec<_> = (0..4).map(|_| rand_poly(nv, &mut rng)).collect();
    let pt = rand_point(nv, &mut rng);
    let points: Vec<_> = (0..4).map(|_| pt.clone()).collect();
    assert!(batch_open_verify(&ck, &vk, &polys, &points)?);
    Ok(())
}

#[test]
fn batch_non_power_of_two_k() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv + 3);
    for k in [3usize, 5, 6, 7] {
        let polys: Vec<_> = (0..k).map(|_| rand_poly(nv, &mut rng)).collect();
        let points: Vec<_> = (0..k).map(|_| rand_point(nv, &mut rng)).collect();
        assert!(batch_open_verify(&ck, &vk, &polys, &points)?, "k={k}");
    }
    Ok(())
}

#[test]
fn batch_reject_wrong_eval() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv + 2);
    let polys: Vec<_> = (0..3).map(|_| rand_poly(nv, &mut rng)).collect();
    let points: Vec<_> = (0..3).map(|_| rand_point(nv, &mut rng)).collect();
    let mut evals: Vec<Fr> = polys
        .iter()
        .zip(points.iter())
        .map(|(f, p)| f.evaluate(p).unwrap())
        .collect();
    evals[1] += Fr::one();
    let commitments: Vec<_> = polys
        .iter()
        .map(|poly| MercuryPCS::<E>::commit(&ck, poly).unwrap())
        .collect();
    let mut tr = IOPTranscript::<Fr>::new(b"mercury-batch-test");
    tr.append_field_element(b"init", &Fr::zero())?;
    let batch_proof = MercuryPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tr)?;
    let mut tr2 = IOPTranscript::<Fr>::new(b"mercury-batch-test");
    tr2.append_field_element(b"init", &Fr::zero())?;
    let res = MercuryPCS::<E>::batch_verify(&vk, &commitments, &points, &batch_proof, &mut tr2);
    assert!(
        res.is_err() || !res.unwrap(),
        "batch with wrong eval must reject"
    );
    Ok(())
}

#[test]
fn batch_reject_wrong_commitment() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv + 2);
    let polys: Vec<_> = (0..3).map(|_| rand_poly(nv, &mut rng)).collect();
    let points: Vec<_> = (0..3).map(|_| rand_point(nv, &mut rng)).collect();
    let evals: Vec<Fr> = polys
        .iter()
        .zip(points.iter())
        .map(|(f, p)| f.evaluate(p).unwrap())
        .collect();
    let mut commitments: Vec<_> = polys
        .iter()
        .map(|poly| MercuryPCS::<E>::commit(&ck, poly).unwrap())
        .collect();
    commitments[0] = MercuryPCS::<E>::commit(&ck, &rand_poly(nv, &mut rng))?;
    let mut tr = IOPTranscript::<Fr>::new(b"mercury-batch-test");
    tr.append_field_element(b"init", &Fr::zero())?;
    let batch_proof = MercuryPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tr)?;
    let mut tr2 = IOPTranscript::<Fr>::new(b"mercury-batch-test");
    tr2.append_field_element(b"init", &Fr::zero())?;
    let ok = MercuryPCS::<E>::batch_verify(&vk, &commitments, &points, &batch_proof, &mut tr2)?;
    assert!(!ok, "batch with wrong commitment must reject");
    Ok(())
}

#[test]
fn batch_malformed_inputs_error() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, _vk) = setup(nv + 2);
    let polys: Vec<_> = (0..3).map(|_| rand_poly(nv, &mut rng)).collect();
    let points: Vec<_> = (0..3).map(|_| rand_point(nv, &mut rng)).collect();
    let evals: Vec<Fr> = polys
        .iter()
        .zip(points.iter())
        .map(|(f, p)| f.evaluate(p).unwrap())
        .collect();
    let mut tr = IOPTranscript::<Fr>::new(b"m");
    tr.append_field_element(b"init", &Fr::zero())?;
    // empty
    assert!(MercuryPCS::<E>::multi_open(&ck, &[], &[], &[], &mut tr).is_err());
    // length mismatch (evals shorter)
    assert!(MercuryPCS::<E>::multi_open(&ck, &polys, &points, &evals[..2], &mut tr).is_err());
    // inconsistent num_vars
    let mut polys_bad = polys.clone();
    polys_bad[0] = rand_poly(nv + 1, &mut rng);
    assert!(MercuryPCS::<E>::multi_open(&ck, &polys_bad, &points, &evals, &mut tr).is_err());
    // wrong point length
    let mut points_bad = points.clone();
    points_bad[0] = rand_point(nv + 1, &mut rng);
    assert!(MercuryPCS::<E>::multi_open(&ck, &polys, &points_bad, &evals, &mut tr).is_err());
    Ok(())
}

#[test]
fn batch_malicious_sumcheck_point_length_no_panic() -> Result<(), PCSError> {
    // Feed batch_verify a proof whose sumcheck point length is inconsistent.
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv + 2);
    let polys: Vec<_> = (0..3).map(|_| rand_poly(nv, &mut rng)).collect();
    let points: Vec<_> = (0..3).map(|_| rand_point(nv, &mut rng)).collect();
    let evals: Vec<Fr> = polys
        .iter()
        .zip(points.iter())
        .map(|(f, p)| f.evaluate(p).unwrap())
        .collect();
    let commitments: Vec<_> = polys
        .iter()
        .map(|poly| MercuryPCS::<E>::commit(&ck, poly).unwrap())
        .collect();
    let mut tr = IOPTranscript::<Fr>::new(b"m");
    tr.append_field_element(b"init", &Fr::zero())?;
    let mut batch_proof = MercuryPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tr)?;
    // Corrupt the sumcheck point to an inconsistent length.
    batch_proof.sum_check_proof.point.push(Fr::rand(&mut rng));
    let res = panic::catch_unwind(|| {
        let mut tr2 = IOPTranscript::<Fr>::new(b"m");
        tr2.append_field_element(b"init", &Fr::zero()).unwrap();
        MercuryPCS::<E>::batch_verify(&vk, &commitments, &points, &batch_proof, &mut tr2)
    });
    assert!(
        res.is_ok(),
        "batch_verify panicked on malicious point length"
    );
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// catch_unwind: malicious raw inputs never panic
// ════════════════════════════════════════════════════════════════════

#[test]
fn catch_unwind_random_proofs() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let com = MercuryPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = MercuryPCS::<E>::open(&ck, &p, &pt)?;
    // Randomly tamper many fields and confirm no panic.
    for _ in 0..32 {
        let mut bad = proof.clone();
        bad.comm_h = tamper_g1(&bad.comm_h);
        bad.g_zeta = Fr::rand(&mut rng);
        bad.h_zeta_inv = Fr::rand(&mut rng);
        let res = panic::catch_unwind(|| MercuryPCS::<E>::verify(&vk, &com, &pt, &val, &bad));
        assert!(res.is_ok());
    }
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// BDFG20 coefficient-level algebraic tests (independent of end-to-end verify)
// ════════════════════════════════════════════════════════════════════

/// Naive polynomial multiplication reference.
fn poly_mul(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Fr::zero(); a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

fn poly_trim(v: &[Fr]) -> Vec<Fr> {
    let mut e = v.len();
    while e > 0 && v[e - 1].is_zero() {
        e -= 1;
    }
    v[..e].to_vec()
}

fn assert_poly_eq(a: &[Fr], b: &[Fr], msg: &str) {
    assert_eq!(poly_trim(a), poly_trim(b), "{msg}");
}

/// `(X - alpha)(X - zeta)(X - zeta_inv)`.
fn z_t_poly(alpha: Fr, zeta: Fr, zeta_inv: Fr) -> Vec<Fr> {
    let a = poly_mul(&[-alpha, Fr::one()], &[-zeta, Fr::one()]);
    poly_mul(&a, &[-zeta_inv, Fr::one()])
}

/// Non-degenerate `(alpha, zeta, zeta_inv, beta, z)`: `zeta != 0`, `zeta^2 !=
/// 1`, and `z`/`alpha` distinct from the three interpolation nodes.
fn nondegenerate_challenges(rng: &mut impl Rng) -> (Fr, Fr, Fr, Fr, Fr) {
    loop {
        let zeta = Fr::rand(rng);
        if zeta.is_zero() {
            continue;
        }
        let zeta_inv = zeta.inverse().unwrap();
        if zeta == zeta_inv {
            continue;
        }
        let alpha = Fr::rand(rng);
        if alpha == zeta || alpha == zeta_inv {
            continue;
        }
        // The BDFG20 batching test needs every polynomial family to contribute
        // to m(X), so exclude beta = 0 rather than relying on a negligible
        // random event not occurring.
        let beta = Fr::rand(rng);
        if beta.is_zero() {
            continue;
        }
        let z = Fr::rand(rng);
        if z == zeta || z == zeta_inv || z == alpha {
            continue;
        }
        return (alpha, zeta, zeta_inv, beta, z);
    }
}

#[test]
fn bdfg_coefficient_identities_and_commitments() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for mu in [2usize, 3, 4, 5] {
        let (_t, b, _b_row, n) = mercury_dims(mu).unwrap();
        let (ck, vk) = setup(mu);

        // Random committed polynomials with the real Mercury degree bounds.
        let g: Vec<Fr> = (0..b).map(|_| Fr::rand(&mut rng)).collect();
        let h: Vec<Fr> = (0..b).map(|_| Fr::rand(&mut rng)).collect();
        let s: Vec<Fr> = (0..b - 1).map(|_| Fr::rand(&mut rng)).collect();
        let d: Vec<Fr> = (0..b).map(|_| Fr::rand(&mut rng)).collect();

        let (alpha, zeta, zeta_inv, beta, z) = nondegenerate_challenges(&mut rng);

        // Evals consistent with the polynomials (so m is divisible by Z_T).
        let g_zeta = poly_eval(&g, zeta);
        let g_zeta_inv = poly_eval(&g, zeta_inv);
        let h_zeta = poly_eval(&h, zeta);
        let h_zeta_inv = poly_eval(&h, zeta_inv);
        let h_alpha = poly_eval(&h, alpha);
        let s_zeta = poly_eval(&s, zeta);
        let s_zeta_inv = poly_eval(&s, zeta_inv);
        let d_zeta = poly_eval(&d, zeta);

        let inp = BdfgProverInput {
            g: &g,
            h: &h,
            s: &s,
            d: &d,
            g_zeta,
            g_zeta_inv,
            h_zeta,
            h_zeta_inv,
            h_alpha,
            s_zeta,
            s_zeta_inv,
            d_zeta,
            zeta,
            zeta_inv,
            alpha,
        };

        // Round 1: m = Z_T * W  (coefficient level).
        let mpolys = bdfg_build_m(&inp, beta)?;
        let zt = z_t_poly(alpha, zeta, zeta_inv);
        assert_poly_eq(
            &mpolys.m,
            &poly_mul(&zt, &mpolys.quot_m),
            &format!("m != Z_T * W at mu={mu}"),
        );

        // Round 2: L = (X - z) * W'  (coefficient level).
        let lpolys = bdfg_build_l(&inp, &mpolys, beta, z)?;
        assert_poly_eq(
            &lpolys.l,
            &poly_mul(&[-z, Fr::one()], &lpolys.quot_l),
            &format!("L != (X-z) W' at mu={mu}"),
        );

        // Commitments of the reference witnesses.
        let comm_w = ck.commit(&mpolys.quot_m)?;
        let comm_w_prime = ck.commit(&lpolys.quot_l)?;

        // Exercise the transcript wrapper itself, not just the pure helpers.
        // Preview its Fiat-Shamir challenges, construct the independent
        // reference witnesses, then replay the same transcript through
        // bdfg_prove and compare the emitted commitments directly.
        let mut wrapper_checked = false;
        for salt in 0u64..64 {
            let mut tr = IOPTranscript::<Fr>::new(b"mercury-bdfg-wrapper-test");
            tr.append_field_element(b"init", &Fr::from(salt))?;
            let mut preview = tr.clone();
            let fs_beta = preview.get_and_append_challenge(L_BETA)?;
            let fs_mpolys = bdfg_build_m(&inp, fs_beta)?;
            let fs_comm_w = ck.commit(&fs_mpolys.quot_m)?;
            preview.append_serializable_element(L_W, &fs_comm_w)?;
            let fs_z = preview.get_and_append_challenge(L_ZBDFG)?;
            if validate_zbdfg(fs_z, zeta, zeta_inv, alpha).is_err() {
                continue;
            }
            let fs_lpolys = bdfg_build_l(&inp, &fs_mpolys, fs_beta, fs_z)?;
            let fs_comm_w_prime = ck.commit(&fs_lpolys.quot_l)?;
            let (prover_comm_w, prover_comm_w_prime) = bdfg_prove(&ck, &inp, mu, n, &mut tr)?;
            assert_eq!(
                prover_comm_w, fs_comm_w,
                "bdfg_prove W commitment mismatch at mu={mu}"
            );
            assert_eq!(
                prover_comm_w_prime, fs_comm_w_prime,
                "bdfg_prove W' commitment mismatch at mu={mu}"
            );
            wrapper_checked = true;
            break;
        }
        assert!(
            wrapper_checked,
            "unable to derive a non-colliding BDFG20 test challenge"
        );

        // Verifier homomorphic reconstruction, with the same (beta, z).
        let proof = MercuryProof::<E> {
            comm_h: ck.commit(&h)?,
            comm_g: ck.commit(&g)?,
            comm_q: <E as Pairing>::G1Affine::default(),
            comm_s: ck.commit(&s)?,
            comm_d: ck.commit(&d)?,
            comm_quot_f: <E as Pairing>::G1Affine::default(),
            comm_w,
            comm_w_prime,
            g_zeta,
            g_zeta_inv,
            h_zeta,
            h_zeta_inv,
            s_zeta,
            s_zeta_inv,
            mu,
        };
        let ev = BdfgVerifyEvals {
            zeta,
            zeta_inv,
            alpha,
            g_zeta,
            g_zeta_inv,
            h_zeta,
            h_zeta_inv,
            h_alpha,
            s_zeta,
            s_zeta_inv,
            d_zeta,
        };
        let (lhs_1, lhs_2) = bdfg_verify_lhs_pure(&vk, &proof, &ev, beta, z, mu, n)?;
        // lhs_2 = [W'(tau)]_1 and lhs_1 = [tau*W'(tau)]_1 = commit(X * W').
        assert_eq!(lhs_2.into_affine(), comm_w_prime, "lhs_2 != [W'(tau)]");
        let mut x_wprime = vec![Fr::zero()];
        x_wprime.extend_from_slice(&lpolys.quot_l);
        assert_eq!(
            lhs_1.into_affine(),
            ck.commit(&x_wprime)?,
            "lhs_1 != [tau W'(tau)] = commit(X * W')"
        );
    }
    Ok(())
}

#[test]
fn bdfg_reject_inconsistent_evals() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let mu = 4;
    let (_t, b, _b_row, _n) = mercury_dims(mu).unwrap();
    let g: Vec<Fr> = (0..b).map(|_| Fr::rand(&mut rng)).collect();
    let h: Vec<Fr> = (0..b).map(|_| Fr::rand(&mut rng)).collect();
    let s: Vec<Fr> = (0..b - 1).map(|_| Fr::rand(&mut rng)).collect();
    let d: Vec<Fr> = (0..b).map(|_| Fr::rand(&mut rng)).collect();
    let (alpha, zeta, zeta_inv, beta, _z) = nondegenerate_challenges(&mut rng);

    let mk = |g_zeta: Fr, h_alpha: Fr| BdfgProverInput {
        g: &g,
        h: &h,
        s: &s,
        d: &d,
        g_zeta,
        g_zeta_inv: poly_eval(&g, zeta_inv),
        h_zeta: poly_eval(&h, zeta),
        h_zeta_inv: poly_eval(&h, zeta_inv),
        h_alpha,
        s_zeta: poly_eval(&s, zeta),
        s_zeta_inv: poly_eval(&s, zeta_inv),
        d_zeta: poly_eval(&d, zeta),
        zeta,
        zeta_inv,
        alpha,
    };

    // Correct evals: divisible.
    assert!(bdfg_build_m(&mk(poly_eval(&g, zeta), poly_eval(&h, alpha)), beta).is_ok());
    // Wrong g_zeta: m no longer vanishes at zeta -> not divisible by (X - zeta).
    let bad_g = poly_eval(&g, zeta) + Fr::one();
    assert!(
        bdfg_build_m(&mk(bad_g, poly_eval(&h, alpha)), beta).is_err(),
        "wrong g_zeta must break divisibility"
    );
    // Wrong h_alpha: m no longer vanishes at alpha -> not divisible by (X - alpha).
    let bad_h_alpha = poly_eval(&h, alpha) + Fr::one();
    assert!(
        bdfg_build_m(&mk(poly_eval(&g, zeta), bad_h_alpha), beta).is_err(),
        "wrong h_alpha must break divisibility"
    );
    Ok(())
}

#[test]
fn bdfg_verifier_reconstruction_rejects_wrong_challenge() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let mu = 4;
    let (_t, b, _b_row, n) = mercury_dims(mu).unwrap();
    let (ck, vk) = setup(mu);
    let g: Vec<Fr> = (0..b).map(|_| Fr::rand(&mut rng)).collect();
    let h: Vec<Fr> = (0..b).map(|_| Fr::rand(&mut rng)).collect();
    let s: Vec<Fr> = (0..b - 1).map(|_| Fr::rand(&mut rng)).collect();
    let d: Vec<Fr> = (0..b).map(|_| Fr::rand(&mut rng)).collect();
    let (alpha, zeta, zeta_inv, beta, z) = nondegenerate_challenges(&mut rng);
    let g_zeta = poly_eval(&g, zeta);
    let g_zeta_inv = poly_eval(&g, zeta_inv);
    let h_zeta = poly_eval(&h, zeta);
    let h_zeta_inv = poly_eval(&h, zeta_inv);
    let h_alpha = poly_eval(&h, alpha);
    let s_zeta = poly_eval(&s, zeta);
    let s_zeta_inv = poly_eval(&s, zeta_inv);
    let d_zeta = poly_eval(&d, zeta);
    let inp = BdfgProverInput {
        g: &g,
        h: &h,
        s: &s,
        d: &d,
        g_zeta,
        g_zeta_inv,
        h_zeta,
        h_zeta_inv,
        h_alpha,
        s_zeta,
        s_zeta_inv,
        d_zeta,
        zeta,
        zeta_inv,
        alpha,
    };
    let mpolys = bdfg_build_m(&inp, beta)?;
    let lpolys = bdfg_build_l(&inp, &mpolys, beta, z)?;
    let comm_w = ck.commit(&mpolys.quot_m)?;
    let comm_w_prime = ck.commit(&lpolys.quot_l)?;
    let proof = MercuryProof::<E> {
        comm_h: ck.commit(&h)?,
        comm_g: ck.commit(&g)?,
        comm_q: <E as Pairing>::G1Affine::default(),
        comm_s: ck.commit(&s)?,
        comm_d: ck.commit(&d)?,
        comm_quot_f: <E as Pairing>::G1Affine::default(),
        comm_w,
        comm_w_prime,
        g_zeta,
        g_zeta_inv,
        h_zeta,
        h_zeta_inv,
        s_zeta,
        s_zeta_inv,
        mu,
    };
    let ev = BdfgVerifyEvals {
        zeta,
        zeta_inv,
        alpha,
        g_zeta,
        g_zeta_inv,
        h_zeta,
        h_zeta_inv,
        h_alpha,
        s_zeta,
        s_zeta_inv,
        d_zeta,
    };
    let mut x_wprime = vec![Fr::zero()];
    x_wprime.extend_from_slice(&lpolys.quot_l);
    let target = ck.commit(&x_wprime)?;

    // Correct challenge reproduces [tau W'(tau)].
    let (lhs_1_ok, _) = bdfg_verify_lhs_pure(&vk, &proof, &ev, beta, z, mu, n)?;
    assert_eq!(lhs_1_ok.into_affine(), target);

    // A wrong beta reconstructs a different group element.
    let (lhs_1_bad, _) = bdfg_verify_lhs_pure(&vk, &proof, &ev, beta + Fr::one(), z, mu, n)?;
    assert_ne!(
        lhs_1_bad.into_affine(),
        target,
        "wrong beta must break the reconstruction"
    );
    // A wrong z as well.
    let z2 = {
        let mut z2 = z + Fr::one();
        while z2 == zeta || z2 == zeta_inv || z2 == alpha {
            z2 += Fr::one();
        }
        z2
    };
    let (lhs_1_bad_z, _) = bdfg_verify_lhs_pure(&vk, &proof, &ev, beta, z2, mu, n)?;
    assert_ne!(
        lhs_1_bad_z.into_affine(),
        target,
        "wrong z must break the reconstruction"
    );
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// odd-nv rectangular-split invariant / differential tests
// ════════════════════════════════════════════════════════════════════

#[test]
fn odd_nv_matrix_restriction_matches_evaluation() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for mu in [3usize, 5, 7] {
        let (t, b, b_row, _n) = mercury_dims(mu).unwrap();
        assert_eq!(b_row, b / 2, "odd mu must give a b x (b/2) rectangle");
        let p = rand_poly(mu, &mut rng);
        let pt = rand_point(mu, &mut rng);
        let coeffs = p.to_evaluations();
        let u1 = &pt[..t];
        let u2 = &pt[t..];
        let eq_col = build_eq_vec::<Fr>(u1, b);
        let eq_row = build_eq_vec::<Fr>(u2, b_row);
        let h = compute_h(&coeffs, &eq_col, b_row, b);
        // hhat(u2) = <eq_row, h> must equal the multilinear evaluation.
        let mut v_check = Fr::zero();
        for (j, &e) in eq_row.iter().enumerate().take(b_row) {
            v_check += e * h[j];
        }
        assert_eq!(
            v_check,
            p.evaluate(&pt).unwrap(),
            "restriction IPA at mu={mu}"
        );
    }
    Ok(())
}

#[test]
fn odd_nv_fold_identity() {
    let mut rng = test_rng();
    for mu in [3usize, 5, 7] {
        let (_t, b, b_row, n) = mercury_dims(mu).unwrap();
        let coeffs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let alpha = Fr::rand(&mut rng);
        let (g, q) = divide_by_binomial(&coeffs, b, b_row, alpha);
        assert_eq!(g.len(), b);
        assert!(g.len() >= b_row, "g degree < b holds");
        for _ in 0..4 {
            let r = Fr::rand(&mut rng);
            let lhs = poly_eval(&coeffs, r);
            let rhs = (r.pow([b as u64]) - alpha) * poly_eval(&q, r) + poly_eval(&g, r);
            assert_eq!(lhs, rhs, "fold identity at mu={mu}");
        }
    }
}

#[test]
fn odd_nv_structured_s_identity() {
    let mut rng = test_rng();
    for mu in [3usize, 5, 7] {
        let (t, b, b_row, _n) = mercury_dims(mu).unwrap();
        let g: Vec<Fr> = (0..b).map(|_| Fr::rand(&mut rng)).collect();
        let mut h: Vec<Fr> = (0..b_row).map(|_| Fr::rand(&mut rng)).collect();
        h.resize(b, Fr::zero());
        let u1: Vec<Fr> = (0..t).map(|_| Fr::rand(&mut rng)).collect();
        let u2: Vec<Fr> = (0..(mu - t)).map(|_| Fr::rand(&mut rng)).collect();
        let mut u2_full = u2.clone();
        u2_full.resize(t, Fr::zero());
        let gamma = Fr::rand(&mut rng);
        let s = make_s_polynomial_structured(&g, &h, &u1, &u2_full, t, b, gamma);

        let eq_col = build_eq_vec::<Fr>(&u1, b);
        let eq_row = build_eq_vec::<Fr>(&u2, b_row);
        let ip_g: Fr = g.iter().zip(eq_col.iter()).map(|(a, c)| *a * *c).sum();
        let ip_h: Fr = h
            .iter()
            .take(b_row)
            .zip(eq_row.iter())
            .map(|(a, c)| *a * *c)
            .sum();

        let z = loop {
            let z = Fr::rand(&mut rng);
            if !z.is_zero() {
                break z;
            }
        };
        let z_inv = z.inverse().expect("nonzero z has an inverse");
        let pu1_z = pu_eval(&u1, z);
        let pu1_zi = pu_eval(&u1, z_inv);
        let pu2_z = pu_eval(&u2, z);
        let pu2_zi = pu_eval(&u2, z_inv);
        let lhs = poly_eval(&g, z) * pu1_zi
            + poly_eval(&g, z_inv) * pu1_z
            + gamma * (poly_eval(&h, z) * pu2_zi + poly_eval(&h, z_inv) * pu2_z);
        let rhs =
            (ip_g + gamma * ip_h).double() + z * poly_eval(&s, z) + z_inv * poly_eval(&s, z_inv);
        assert_eq!(lhs, rhs, "S symmetric-Laurent identity at mu={mu}");
    }
}

#[test]
fn odd_nv_rectangular_matches_zero_row_extension() {
    // This is a local-layout differential, not a replay of Nova's odd-nv path.
    // Appending N zero coefficients introduces one new *high* variable fixed at
    // zero, so the original b x (b/2) rectangle becomes a b x b matrix with
    // zero upper rows. The low-variable column point is unchanged, hence g and
    // h must agree coefficient-wise. Nova's code inserts a point coordinate at
    // a different position and uses a different variable split, so its raw g/h
    // vectors are not expected to equal these ones without a permutation map.
    let mut rng = test_rng();
    for mu in [3usize, 5, 7] {
        let (t, b, b_row, n) = mercury_dims(mu).unwrap();
        assert_eq!(b_row, b / 2);
        assert_eq!(b * b, 2 * n, "square layout has b^2 = 2N for odd mu");
        let coeffs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let alpha = Fr::rand(&mut rng);

        // rectangular b x (b/2)
        let (g_rect, _q_rect) = divide_by_binomial(&coeffs, b, b_row, alpha);
        // padded square b x b: append N zeros (new highest variable = 0-half).
        let mut padded = coeffs.clone();
        padded.resize(2 * n, Fr::zero());
        let (g_sq, _q_sq) = divide_by_binomial(&padded, b, b, alpha);
        assert_eq!(g_rect, g_sq, "g rectangular != g padded-square at mu={mu}");

        // h from a random column point.
        let u1: Vec<Fr> = (0..t).map(|_| Fr::rand(&mut rng)).collect();
        let eq_col = build_eq_vec::<Fr>(&u1, b);
        let h_rect = compute_h(&coeffs, &eq_col, b_row, b);
        let h_sq = compute_h(&padded, &eq_col, b, b);
        assert_eq!(h_rect, h_sq, "h rectangular != h padded-square at mu={mu}");

        // The zero-row extension represents the original MLE with one new
        // highest variable fixed to zero.
        let original = DenseMultilinearExtension::from_evaluations_vec(mu, coeffs);
        let extended = DenseMultilinearExtension::from_evaluations_vec(mu + 1, padded);
        let point: Vec<Fr> = (0..mu).map(|_| Fr::rand(&mut rng)).collect();
        let mut extended_point = point.clone();
        extended_point.push(Fr::zero());
        assert_eq!(
            original.evaluate(&point),
            extended.evaluate(&extended_point),
            "zero-row extension changes the MLE value at mu={mu}"
        );
    }
}
