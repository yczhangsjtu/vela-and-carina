//! VelaPCS test suite: positive, negative, edge-case, and property tests.

use super::*;
use ark_bls12_381::{Bls12_381, Fr};
use ark_serialize::CanonicalSerialize;
use ark_std::{test_rng, One, UniformRand, Zero};

type E = Bls12_381;

fn setup(nv: usize) -> (VelaProverParam<E>, VelaVerifierParam<E>) {
    let mut rng = test_rng();
    let srs = VelaPCS::<E>::gen_srs_for_testing(&mut rng, nv).unwrap();
    VelaPCS::<E>::trim(&srs, None, Some(nv)).unwrap()
}

fn rand_point(nv: usize, rng: &mut impl Rng) -> Vec<Fr> {
    (0..nv).map(|_| Fr::rand(rng)).collect()
}

fn rand_poly(nv: usize, rng: &mut impl Rng) -> Arc<DenseMultilinearExtension<Fr>> {
    Arc::new(DenseMultilinearExtension::rand(nv, rng))
}

// ── Positive: commit/open/verify over several sizes ──
#[test]
fn test_commit_open_verify() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [1usize, 2, 4, 6, 8] {
        let (ck, vk) = setup(nv);
        let poly = rand_poly(nv, &mut rng);
        let point = rand_point(nv, &mut rng);
        let com = VelaPCS::<E>::commit(&ck, &poly)?;
        let (proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;
        assert_eq!(value, poly.evaluate(&point).unwrap());
        assert!(VelaPCS::<E>::verify(&vk, &com, &point, &value, &proof)?);
    }
    Ok(())
}

// ── Proof shape: exactly 2 G1 + 4 F ──
#[test]
fn test_proof_shape_bytes() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, _vk) = setup(6);
    let poly = rand_poly(6, &mut rng);
    let point = rand_point(6, &mut rng);
    let (proof, _) = VelaPCS::<E>::open(&ck, &poly, &point)?;
    let g1 = {
        let mut b = vec![];
        proof.cm_hbar.serialize_compressed(&mut b).unwrap();
        b.len()
    };
    let fr = {
        let mut b = vec![];
        proof.f_z.serialize_compressed(&mut b).unwrap();
        b.len()
    };
    let mut all = vec![];
    proof.serialize_compressed(&mut all).unwrap();
    // 2 G1 + 4 F + a small mu (usize). Assert the group/field payload matches.
    assert!(all.len() >= 2 * g1 + 4 * fr);
    println!(
        "VelaPCS proof: 2 G1 ({} B) + 4 F ({} B) = {} B payload; serialized {} B",
        g1,
        fr,
        2 * g1 + 4 * fr,
        all.len()
    );
    Ok(())
}

// ── Wrong value / point / commitment ──
#[test]
fn test_reject_wrong_value() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(4);
    let poly = rand_poly(4, &mut rng);
    let point = rand_point(4, &mut rng);
    let com = VelaPCS::<E>::commit(&ck, &poly)?;
    let (proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;
    let mut wrong = value + Fr::one();
    if wrong == value {
        wrong += Fr::one();
    }
    assert!(!VelaPCS::<E>::verify(&vk, &com, &point, &wrong, &proof)?);
    Ok(())
}

#[test]
fn test_reject_wrong_point() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(4);
    let poly = rand_poly(4, &mut rng);
    let point = rand_point(4, &mut rng);
    let com = VelaPCS::<E>::commit(&ck, &poly)?;
    let (proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;
    let wp = rand_point(4, &mut rng);
    if wp != point {
        assert!(!VelaPCS::<E>::verify(&vk, &com, &wp, &value, &proof)?);
    }
    Ok(())
}

#[test]
fn test_reject_wrong_commitment() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(4);
    let poly = rand_poly(4, &mut rng);
    let poly2 = rand_poly(4, &mut rng);
    let point = rand_point(4, &mut rng);
    let com2 = VelaPCS::<E>::commit(&ck, &poly2)?;
    let (proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;
    assert!(!VelaPCS::<E>::verify(&vk, &com2, &point, &value, &proof)?);
    Ok(())
}

// ── Tampered proof fields ──
macro_rules! tamper_test {
    ($name:ident, $field:ident) => {
        #[test]
        fn $name() -> Result<(), PCSError> {
            let mut rng = test_rng();
            let (ck, vk) = setup(4);
            let poly = rand_poly(4, &mut rng);
            let point = rand_point(4, &mut rng);
            let com = VelaPCS::<E>::commit(&ck, &poly)?;
            let (mut proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;
            proof.$field += Fr::one();
            assert!(!VelaPCS::<E>::verify(&vk, &com, &point, &value, &proof)?);
            Ok(())
        }
    };
}
tamper_test!(test_reject_tampered_f_z, f_z);
tamper_test!(test_reject_tampered_f_z_inv, f_z_inv);
tamper_test!(test_reject_tampered_hbar_z, hbar_z);
tamper_test!(test_reject_tampered_hbar_z_inv, hbar_z_inv);

#[test]
fn test_reject_tampered_cm_hbar() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(4);
    let poly = rand_poly(4, &mut rng);
    let point = rand_point(4, &mut rng);
    let com = VelaPCS::<E>::commit(&ck, &poly)?;
    let (mut proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;
    proof.cm_hbar = (proof.cm_hbar.into_group() * Fr::from(2u64)).into_affine();
    assert!(!VelaPCS::<E>::verify(&vk, &com, &point, &value, &proof)?);
    Ok(())
}

#[test]
fn test_reject_tampered_pi() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(4);
    let poly = rand_poly(4, &mut rng);
    let point = rand_point(4, &mut rng);
    let com = VelaPCS::<E>::commit(&ck, &poly)?;
    let (mut proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;
    proof.pi = (proof.pi.into_group() * Fr::from(3u64)).into_affine();
    assert!(!VelaPCS::<E>::verify(&vk, &com, &point, &value, &proof)?);
    Ok(())
}

// ── Tampered z: since z is Fiat-Shamir-derived, the verifier recomputes it and
//    a stale/forged z cannot be injected through the proof (z is not a field of
//    VelaProof). This test forges a proof by rerunning the prover with a
// poisoned    transcript and checks the honest verifier rejects it. ──
#[test]
fn test_reject_tampered_alpha_via_transcript_desync() -> Result<(), PCSError> {
    // If a malicious prover derives alpha from a different transcript than the
    // verifier, the pairing check must fail. We simulate by building a proof
    // whose pi is computed with a wrong alpha.
    let mut rng = test_rng();
    let (ck, vk) = setup(4);
    let poly = rand_poly(4, &mut rng);
    let point = rand_point(4, &mut rng);
    let com = VelaPCS::<E>::commit(&ck, &poly)?;
    let (proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;

    // Recompute pi with a wrong alpha and splice it in.
    let coeffs = poly.to_evaluations();
    let n = 1usize << 4;
    let laurent_h = compute_laurent_h(&coeffs, 4, &point);
    let hbar_coeffs = compute_hbar(&laurent_h, n - 1);
    // Re-derive z honestly to get the right points, then use a bogus alpha.
    let mut t = new_transcript::<E>(4, &com, &point, &value)?;
    t.append_serializable_element(LABEL_CM_HBAR, &proof.cm_hbar)?;
    let z = t.get_and_append_challenge_vectors(LABEL_Z, 1)?[0];
    let z_inv = validate_z(z)?;
    let rf = two_point_remainder(z, proof.f_z, z_inv, proof.f_z_inv)?;
    let rh = two_point_remainder(z, proof.hbar_z, z_inv, proof.hbar_z_inv)?;
    let bogus_alpha = Fr::rand(&mut rng);
    let q = build_quotient(&coeffs, &hbar_coeffs, &rf, &rh, bogus_alpha, z + z_inv);
    let bogus_pi = ck.commit(&q)?;
    let mut forged = proof.clone();
    forged.pi = bogus_pi;
    assert!(!VelaPCS::<E>::verify(&vk, &com, &point, &value, &forged)?);
    Ok(())
}

// ── Malformed proof: wrong mu ──
#[test]
fn test_reject_malformed_mu() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(4);
    let poly = rand_poly(4, &mut rng);
    let point = rand_point(4, &mut rng);
    let com = VelaPCS::<E>::commit(&ck, &poly)?;
    let (mut proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;
    proof.mu = 5; // inconsistent with the 4-length point
    let r = VelaPCS::<E>::verify(&vk, &com, &point, &value, &proof);
    assert!(r.is_err() || !r.unwrap());
    Ok(())
}

// ── Wrong point length: error, no panic ──
#[test]
fn test_reject_wrong_point_len_no_panic() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(4);
    let poly = rand_poly(4, &mut rng);
    let point = rand_point(4, &mut rng);
    let com = VelaPCS::<E>::commit(&ck, &poly)?;
    let (proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;
    let short = rand_point(2, &mut rng);
    assert!(VelaPCS::<E>::verify(&vk, &com, &short, &value, &proof).is_err());
    let long = rand_point(8, &mut rng);
    assert!(VelaPCS::<E>::verify(&vk, &com, &long, &value, &proof).is_err());
    Ok(())
}

// ── Huge mu: error, no panic (unchecked-shift safety) ──
#[test]
fn test_reject_huge_mu_no_panic() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(2);
    let poly = rand_poly(2, &mut rng);
    let point = rand_point(2, &mut rng);
    let com = VelaPCS::<E>::commit(&ck, &poly)?;
    let (mut proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;
    proof.mu = usize::BITS as usize;
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        VelaPCS::<E>::verify(&vk, &com, &point, &value, &proof)
    }));
    match r {
        Ok(verdict) => assert!(verdict.is_err() || !verdict.unwrap()),
        Err(_) => panic!("verify panicked on huge mu"),
    }
    Ok(())
}

// ── Edge points z in {0, 1, -1}: validate_z rejects ──
#[test]
fn test_validate_z_rejects_edges() {
    assert!(validate_z(Fr::zero()).is_err());
    assert!(validate_z(Fr::one()).is_err());
    assert!(validate_z(-Fr::one()).is_err());
    let mut rng = test_rng();
    let z = Fr::rand(&mut rng);
    if !z.is_zero() && z.square() != Fr::one() {
        assert_eq!(z * validate_z(z).unwrap(), Fr::one());
    }
}

// ── Duplicate two-point rejection: if z = z^{-1} the remainder is undefined ──
#[test]
fn test_two_point_remainder_duplicate_rejected() {
    let x = Fr::one();
    // x_inv = x => denom zero
    let r = two_point_remainder(x, Fr::from(3u64), x, Fr::from(5u64));
    assert!(r.is_err());
}

// ── Batching decoupling attack: two individually-false evaluations that the
//    attacker hopes cancel under alpha must be rejected. We craft a proof with
//    both f_z and hbar_z shifted so that (delta_f + alpha delta_h) could vanish
//    only for a specific alpha; the FS alpha is drawn after the evals, so it
// will    not match, and additionally the identity check fails. ──
#[test]
fn test_batching_decoupling_attack_rejected() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(4);
    let poly = rand_poly(4, &mut rng);
    let point = rand_point(4, &mut rng);
    let com = VelaPCS::<E>::commit(&ck, &poly)?;
    let (mut proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;
    // Shift both f_z and hbar_z; a naive batching-only check might be fooled for
    // some alpha, but VelaPCS binds alpha after the evals and also runs the
    // identity check.
    let d = Fr::rand(&mut rng);
    proof.f_z += d;
    proof.hbar_z += d;
    assert!(!VelaPCS::<E>::verify(&vk, &com, &point, &value, &proof)?);
    Ok(())
}

// ── Property test: structured construction == direct coefficient construction
// ──
#[test]
fn test_property_structured_matches_direct() {
    let mut rng = test_rng();
    for nv in [1usize, 2, 3, 5, 7] {
        let n = 1usize << nv;
        let coeffs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let r: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

        // Structured Laurent h.
        let lh = compute_laurent_h(&coeffs, nv, &r);
        let offset = n - 1;

        // Direct: h_d = sum_{i-j=d} a_i eq(j;r).
        let eqr: Vec<Fr> = (0..n)
            .map(|j| {
                (0..nv).fold(Fr::one(), |acc, k| {
                    let bit = (j >> k) & 1;
                    acc * if bit == 1 { r[k] } else { Fr::one() - r[k] }
                })
            })
            .collect();
        let mut direct = vec![Fr::zero(); 2 * n - 1];
        for (i, &ai) in coeffs.iter().enumerate() {
            for (j, &ej) in eqr.iter().enumerate() {
                direct[offset + i - j] += ai * ej;
            }
        }
        assert_eq!(lh, direct, "structured h != direct h at nv={}", nv);

        // Constant coeff = f(r).
        let poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            nv,
            coeffs.clone(),
        ));
        let fr_val = poly.evaluate(&r).unwrap();
        assert_eq!(lh[offset], fr_val, "h_0 != f(r) at nv={}", nv);

        // hbar decomposition & degree.
        let hbar = compute_hbar(&lh, offset);
        assert_eq!(hbar.len(), n - 1);
    }
}

// ── Randomized end-to-end soundness: wrong value always rejected ──
#[test]
fn test_property_wrong_value_rejected() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for _ in 0..20 {
        let nv = 4;
        let (ck, vk) = setup(nv);
        let poly = rand_poly(nv, &mut rng);
        let point = rand_point(nv, &mut rng);
        let com = VelaPCS::<E>::commit(&ck, &poly)?;
        let (proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;
        assert!(VelaPCS::<E>::verify(&vk, &com, &point, &value, &proof)?);
        let mut wrong = Fr::rand(&mut rng);
        if wrong == value {
            wrong += Fr::one();
        }
        assert!(!VelaPCS::<E>::verify(&vk, &com, &point, &wrong, &proof)?);
    }
    Ok(())
}

// ── Serialization round-trip ──
#[test]
fn test_proof_serialization_roundtrip() -> Result<(), PCSError> {
    use ark_serialize::CanonicalDeserialize;
    let mut rng = test_rng();
    let (ck, vk) = setup(4);
    let poly = rand_poly(4, &mut rng);
    let point = rand_point(4, &mut rng);
    let com = VelaPCS::<E>::commit(&ck, &poly)?;
    let (proof, value) = VelaPCS::<E>::open(&ck, &poly, &point)?;
    let mut bytes = vec![];
    proof.serialize_compressed(&mut bytes).unwrap();
    let proof2 = VelaProof::<E>::deserialize_compressed(&bytes[..]).unwrap();
    assert_eq!(proof, proof2);
    assert!(VelaPCS::<E>::verify(&vk, &com, &point, &value, &proof2)?);
    Ok(())
}

// ── Sum-check batch opening: distinct points, repeated points, k not a power
// of 2 ──
fn batch_case(k: usize, same_point: bool) -> Result<(), PCSError> {
    use transcript::IOPTranscript;
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv);
    let polys: Vec<_> = (0..k).map(|_| rand_poly(nv, &mut rng)).collect();
    let shared = rand_point(nv, &mut rng);
    let points: Vec<_> = if same_point {
        vec![shared; k]
    } else {
        polys.iter().map(|_| rand_point(nv, &mut rng)).collect()
    };
    let evals: Vec<Fr> = polys
        .iter()
        .zip(points.iter())
        .map(|(p, pt)| p.evaluate(pt).unwrap())
        .collect();
    let coms: Vec<_> = polys
        .iter()
        .map(|p| VelaPCS::<E>::commit(&ck, p).unwrap())
        .collect();
    let mut tp = IOPTranscript::new(b"vela-batch-test");
    tp.append_field_element(b"init", &Fr::zero())?;
    let bp = VelaPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
    let mut tv = IOPTranscript::new(b"vela-batch-test");
    tv.append_field_element(b"init", &Fr::zero())?;
    assert!(VelaPCS::<E>::batch_verify(
        &vk, &coms, &points, &bp, &mut tv
    )?);
    Ok(())
}

#[test]
fn test_batch_distinct_points() -> Result<(), PCSError> {
    batch_case(3, false)
}
#[test]
fn test_batch_repeated_points() -> Result<(), PCSError> {
    batch_case(3, true)
}
#[test]
fn test_batch_single() -> Result<(), PCSError> {
    batch_case(1, false)
}
#[test]
fn test_batch_non_power_of_two() -> Result<(), PCSError> {
    batch_case(5, false)
}

#[test]
fn test_batch_rejects_wrong_eval() -> Result<(), PCSError> {
    use transcript::IOPTranscript;
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv);
    let polys: Vec<_> = (0..3).map(|_| rand_poly(nv, &mut rng)).collect();
    let points: Vec<_> = polys.iter().map(|_| rand_point(nv, &mut rng)).collect();
    let mut evals: Vec<Fr> = polys
        .iter()
        .zip(points.iter())
        .map(|(p, pt)| p.evaluate(pt).unwrap())
        .collect();
    let coms: Vec<_> = polys
        .iter()
        .map(|p| VelaPCS::<E>::commit(&ck, p).unwrap())
        .collect();
    evals[0] += Fr::one();
    let mut tp = IOPTranscript::new(b"t");
    tp.append_field_element(b"init", &Fr::zero())?;
    let bp = VelaPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
    let mut tv = IOPTranscript::new(b"t");
    tv.append_field_element(b"init", &Fr::zero())?;
    let r = VelaPCS::<E>::batch_verify(&vk, &coms, &points, &bp, &mut tv);
    assert!(r.is_err() || !r.unwrap());
    Ok(())
}

// ── open_with_commitment: matches trait open ──
#[test]
fn test_open_with_commitment_matches_trait_open() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [2usize, 4, 6, 8] {
        let (ck, vk) = setup(nv);
        let poly = rand_poly(nv, &mut rng);
        let point = rand_point(nv, &mut rng);
        let value = poly.evaluate(&point).unwrap();
        let com = VelaPCS::<E>::commit(&ck, &poly)?;
        let (proof_a, val_a) = VelaPCS::<E>::open_with_commitment(&ck, &poly, &point, value, &com)?;
        let (proof_b, val_b) = VelaPCS::<E>::open(&ck, &poly, &point)?;
        assert_eq!(val_a, val_b);
        assert!(VelaPCS::<E>::verify(&vk, &com, &point, &val_a, &proof_a)?);
        assert!(VelaPCS::<E>::verify(&vk, &com, &point, &val_b, &proof_b)?);
    }
    Ok(())
}

// ── open_with_commitment: wrong commitment is rejected ──
#[test]
fn test_open_with_commitment_rejects_wrong_commitment() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [2usize, 4, 8] {
        let (ck, vk) = setup(nv);
        let poly = rand_poly(nv, &mut rng);
        let poly2 = rand_poly(nv, &mut rng);
        let point = rand_point(nv, &mut rng);
        let value = poly.evaluate(&point).unwrap();
        let wrong_com = VelaPCS::<E>::commit(&ck, &poly2)?;
        let r = VelaPCS::<E>::open_with_commitment(&ck, &poly, &point, value, &wrong_com);
        if let Ok((proof, val)) = r {
            let com = VelaPCS::<E>::commit(&ck, &poly)?;
            // Verify with the SAME vk from the initial setup
            assert!(
                !VelaPCS::<E>::verify(&vk, &com, &point, &val, &proof)?,
                "wrong commitment should not produce verifiable proof"
            );
            // Also assert it doesn't verify under the wrong commitment
            assert!(
                !VelaPCS::<E>::verify(&vk, &wrong_com, &point, &val, &proof)?,
                "proof under wrong commitment should not verify"
            );
        }
    }
    Ok(())
}

// ── open_with_commitment: generated proof is accepted by normal verifier ──
#[test]
fn test_open_with_commitment_valid_proof_accepted() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [2usize, 4, 6, 8, 10] {
        let (ck, vk) = setup(nv);
        let poly = rand_poly(nv, &mut rng);
        let point = rand_point(nv, &mut rng);
        let value = poly.evaluate(&point).unwrap();
        let com = VelaPCS::<E>::commit(&ck, &poly)?;
        let (proof, val) = VelaPCS::<E>::open_with_commitment(&ck, &poly, &point, value, &com)?;
        assert_eq!(val, value);
        assert!(VelaPCS::<E>::verify(&vk, &com, &point, &val, &proof)?);
        // Wrong value must be rejected.
        assert!(!VelaPCS::<E>::verify(
            &vk,
            &com,
            &point,
            &(val + Fr::one()),
            &proof
        )?);
    }
    Ok(())
}

// ── open_with_commitment: wrong point length does not panic ──
#[test]
fn test_open_with_commitment_wrong_point_len_no_panic() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, _vk) = setup(4);
    let poly = rand_poly(4, &mut rng);
    let com = VelaPCS::<E>::commit(&ck, &poly)?;
    let value = poly.evaluate(&[Fr::zero(); 4]).unwrap();
    let short = vec![Fr::zero(); 2];
    assert!(VelaPCS::<E>::open_with_commitment(&ck, &poly, &short, value, &com).is_err());
    let long = vec![Fr::zero(); 8];
    assert!(VelaPCS::<E>::open_with_commitment(&ck, &poly, &long, value, &com).is_err());
    Ok(())
}
