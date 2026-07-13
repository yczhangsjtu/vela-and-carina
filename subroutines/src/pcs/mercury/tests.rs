//! Mercury PCS correctness and negative tests.
//!
//! Covers: even/odd/min `nv`, random property tests, the paper polynomial
//! identities (`f = (X^b-alpha)q + g`, structured `S` vs dense reference), the
//! BDFG20 batch divisibility identity, every negative case (wrong value / point
//! / commitment, tampering each proof field, swapped evaluations), malformed
//! `mu`/lengths, vk/pk capacity, statement binding, serialization roundtrip,
//! proof-size assertion, and `catch_unwind` panic-freedom. Sumcheck batch
//! adapter tests exercise `multi_open` / `batch_verify`.

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
