//! Correctness, tamper, panic-safety, and batch tests for NRG-KZG.

use super::*;
use crate::pcs::PolynomialCommitmentScheme;
use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::UniformRand;
use ark_serialize::Compress;
use ark_std::{test_rng, vec::Vec};

type E = Bls12_381;

fn setup(nv: usize) -> (NestedGridKzgProverParam<E>, NestedGridKzgVerifierParam<E>) {
    let mut rng = test_rng();
    let srs = NestedGridKzgPCS::<E>::gen_srs_for_testing(&mut rng, nv).unwrap();
    NestedGridKzgPCS::<E>::trim(&srs, None, Some(nv)).unwrap()
}

fn rand_poly(nv: usize, rng: &mut impl Rng) -> Arc<DenseMultilinearExtension<Fr>> {
    let n = 1usize << nv;
    let evals: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
    Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, evals))
}

fn rand_point(nv: usize, rng: &mut impl Rng) -> Vec<Fr> {
    (0..nv).map(|_| Fr::rand(rng)).collect()
}

// ── bivariate helpers for algebraic tests ──

/// Evaluate a bivariate polynomial given as `mat[i + big_ml*j]` at `(x,y)`.
fn eval_bivariate(mat: &[Fr], big_ml: usize, big_mr: usize, x: Fr, y: Fr) -> Fr {
    let mut acc = Fr::zero();
    for j in 0..big_mr {
        let mut yp = Fr::one();
        for _ in 0..j {
            yp *= y;
        }
        for i in 0..big_ml {
            let mut xp = Fr::one();
            for _ in 0..i {
                xp *= x;
            }
            acc += mat[i + big_ml * j] * xp * yp;
        }
    }
    acc
}

/// Dense O(M^2) reference for the reciprocal witness.
fn reciprocal_witness_dense(a: &[Fr], psi: &[Fr]) -> Vec<Fr> {
    // W(X) = a(X) psi(X^{-1}); w_k = sum_j a[k+j] psi[j].
    // S[i] = w_{i+1} + w_{-(i+1)}.
    let big_m = a.len();
    let w = |k: isize| -> Fr {
        let mut acc = Fr::zero();
        for (j, &pj) in psi.iter().enumerate() {
            let idx = k + j as isize;
            if idx >= 0 && (idx as usize) < big_m {
                acc += a[idx as usize] * pj;
            }
        }
        acc
    };
    (0..big_m - 1)
        .map(|i| {
            let k = (i + 1) as isize;
            w(k) + w(-k)
        })
        .collect()
}

// ════════════════════════════════════════════════════════════════════
// Positive: end-to-end single open/verify
// ════════════════════════════════════════════════════════════════════

#[test]
fn test_single_open_verify_even_and_odd() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [4usize, 5, 6, 7, 8] {
        let (ck, vk) = setup(nv);
        let poly = rand_poly(nv, &mut rng);
        let point = rand_point(nv, &mut rng);
        let com = NestedGridKzgPCS::<E>::commit(&ck, &poly)?;
        let (proof, value) = NestedGridKzgPCS::<E>::open(&ck, &poly, &point)?;
        // value equals the direct multilinear evaluation
        assert_eq!(value, poly.evaluate(&point).unwrap());
        assert!(
            NestedGridKzgPCS::<E>::verify(&vk, &com, &point, &value, &proof)?,
            "verify failed at nv={nv}"
        );
        // payload is exactly 4 G1 + 8 F
        assert_eq!(proof.cryptographic_payload_bytes(), 4 * 48 + 8 * 32);
        // canonical serialized size adds only the 4-byte mu metadata
        assert_eq!(
            proof.serialized_size(Compress::Yes),
            4 * 48 + 8 * 32 + 4,
            "canonical size = payload + u32 mu"
        );
    }
    Ok(())
}

#[test]
fn test_core_open_matches_trait_open() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 6;
    let (ck, vk) = setup(nv);
    let poly = rand_poly(nv, &mut rng);
    let point = rand_point(nv, &mut rng);
    let com = NestedGridKzgPCS::<E>::commit(&ck, &poly)?;
    let (proof, value) = NestedGridKzgPCS::<E>::open_with_commitment(&ck, &poly, &point, &com)?;
    assert!(NestedGridKzgPCS::<E>::verify(
        &vk, &com, &point, &value, &proof
    )?);
    Ok(())
}

#[test]
fn test_commitment_matches_reference_bivariate_msm() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 6;
    let (ck, _vk) = setup(nv);
    let poly = rand_poly(nv, &mut rng);
    let com = NestedGridKzgPCS::<E>::commit(&ck, &poly)?;
    let big_ml = ck.big_ml();
    let big_mr = ck.big_mr();
    let mut reference = <E as Pairing>::G1::zero();
    for j in 0..big_mr {
        for i in 0..big_ml {
            let base = ck.g1_powers[ck.base_index(i, j)];
            reference += base.into_group() * poly.evaluations[i + big_ml * j];
        }
    }
    assert_eq!(com.0, reference.into_affine());
    Ok(())
}

#[test]
fn test_reciprocal_witness_structured_vs_dense() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for m in [2usize, 3, 4] {
        let big_m = 1usize << m;
        let a: Vec<Fr> = (0..big_m).map(|_| Fr::rand(&mut rng)).collect();
        let u: Vec<Fr> = (0..m).map(|_| Fr::rand(&mut rng)).collect();
        let psi = build_eq_x_r_vec(&u)?;
        let structured = reciprocal_witness(&a, &u, m);
        let dense = reciprocal_witness_dense(&a, &psi);
        assert_eq!(structured, dense, "reciprocal witness mismatch at m={m}");
        // Constant Laurent coefficient of a(X)psi(X^{-1})+a(X^{-1})psi(X) is 2<a,psi>.
        let ip: Fr = a.iter().zip(psi.iter()).map(|(x, y)| *x * *y).sum();
        // Reconstruct via the reciprocal identity at a random z.
        let z = Fr::rand(&mut rng);
        let z_inv = z.inverse().unwrap();
        let lhs = horner(&a, z) * eval_tensor(&u, z_inv) + horner(&a, z_inv) * eval_tensor(&u, z);
        let rhs = ip.double() + z * horner(&structured, z) + z_inv * horner(&structured, z_inv);
        assert_eq!(lhs, rhs, "reciprocal identity failed at m={m}");
    }
    Ok(())
}

#[test]
fn test_grid_interpolation_accurate() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let r = Fr::rand(&mut rng);
    let r_inv = r.inverse().unwrap();
    let s = Fr::rand(&mut rng);
    let s_inv = s.inverse().unwrap();
    let (v_pp, v_pn, v_np, v_nn) = (
        Fr::rand(&mut rng),
        Fr::rand(&mut rng),
        Fr::rand(&mut rng),
        Fr::rand(&mut rng),
    );
    let px_s = two_point_remainder(r, v_pp, r_inv, v_np)?;
    let px_sinv = two_point_remainder(r, v_pn, r_inv, v_nn)?;
    let i0 = two_point_remainder(s, px_s[0], s_inv, px_sinv[0])?;
    let i1 = two_point_remainder(s, px_s[1], s_inv, px_sinv[1])?;
    // I as 2x2 matrix [i + 2*j].
    let i_mat = [i0[0], i1[0], i0[1], i1[1]];
    assert_eq!(eval_bivariate(&i_mat, 2, 2, r, s), v_pp);
    assert_eq!(eval_bivariate(&i_mat, 2, 2, r, s_inv), v_pn);
    assert_eq!(eval_bivariate(&i_mat, 2, 2, r_inv, s), v_np);
    assert_eq!(eval_bivariate(&i_mat, 2, 2, r_inv, s_inv), v_nn);
    Ok(())
}

#[test]
fn test_bivariate_quotient_coefficient_identity() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [4usize, 5, 6] {
        let (m_left, m_right) = split_exponents(nv);
        let big_ml = 1usize << m_left;
        let big_mr = 1usize << m_right;
        let n = big_ml * big_mr;
        let f: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let r = Fr::rand(&mut rng);
        let r_inv = r.inverse().unwrap();
        let s = Fr::rand(&mut rng);
        let s_inv = s.inverse().unwrap();

        // true grid values
        let v_pp = eval_bivariate(&f, big_ml, big_mr, r, s);
        let v_pn = eval_bivariate(&f, big_ml, big_mr, r, s_inv);
        let v_np = eval_bivariate(&f, big_ml, big_mr, r_inv, s);
        let v_nn = eval_bivariate(&f, big_ml, big_mr, r_inv, s_inv);
        let px_s = two_point_remainder(r, v_pp, r_inv, v_np)?;
        let px_sinv = two_point_remainder(r, v_pn, r_inv, v_nn)?;
        let i0 = two_point_remainder(s, px_s[0], s_inv, px_sinv[0])?;
        let i1 = two_point_remainder(s, px_s[1], s_inv, px_sinv[1])?;

        let p_r = r + r_inv;
        let q_s = s + s_inv;

        // Q_X per column + remainder
        let mut qx = vec![vec![Fr::zero(); big_ml - 2]; big_mr];
        let mut r0_col = vec![Fr::zero(); big_mr];
        let mut r1_col = vec![Fr::zero(); big_mr];
        for j in 0..big_mr {
            let mut col = f[big_ml * j..big_ml * (j + 1)].to_vec();
            if j == 0 {
                col[0] -= i0[0];
                col[1] -= i1[0];
            } else if j == 1 {
                col[0] -= i0[1];
                col[1] -= i1[1];
            }
            let (q, rem) = div_by_monic_quadratic(&col, p_r);
            qx[j][..q.len()].copy_from_slice(&q);
            r0_col[j] = rem[0];
            r1_col[j] = rem[1];
        }
        let (qy0, remy0) = div_by_monic_quadratic(&r0_col, q_s);
        let (qy1, remy1) = div_by_monic_quadratic(&r1_col, q_s);
        assert_eq!(remy0, [Fr::zero(); 2]);
        assert_eq!(remy1, [Fr::zero(); 2]);

        // Reconstruct Q_X(X,Y), Q_Y(X,Y), I(X,Y) as bivariate matrices and check
        // f - I == Z_A Q_X + Z_B Q_Y coefficient by coefficient.
        // Evaluate the identity at many random points (equivalent to coeff check
        // for these low-degree factors given enough points).
        for _ in 0..4 {
            let x = Fr::rand(&mut rng);
            let y = Fr::rand(&mut rng);
            let z_a = x * x - p_r * x + Fr::one();
            let z_b = y * y - q_s * y + Fr::one();
            let f_val = eval_bivariate(&f, big_ml, big_mr, x, y);
            let i_mat = [i0[0], i1[0], i0[1], i1[1]];
            let i_val = eval_bivariate(&i_mat, 2, 2, x, y);
            // Q_X value
            let mut qx_val = Fr::zero();
            for j in 0..big_mr {
                let mut yp = Fr::one();
                for _ in 0..j {
                    yp *= y;
                }
                let mut xp = Fr::one();
                for &c in &qx[j] {
                    qx_val += c * xp * yp;
                    xp *= x;
                }
            }
            // Q_Y value (deg_X<2)
            let mut qy_val = Fr::zero();
            let mut yp = Fr::one();
            for j in 0..(big_mr - 2) {
                qy_val += (qy0[j] + qy1[j] * x) * yp;
                yp *= y;
            }
            assert_eq!(
                f_val - i_val,
                z_a * qx_val + z_b * qy_val,
                "grid identity nv={nv}"
            );
        }
    }
    Ok(())
}

#[test]
fn test_witness_quotient_identities() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for m in [2usize, 3, 4] {
        let big_m = 1usize << m;
        let a: Vec<Fr> = (0..big_m).map(|_| Fr::rand(&mut rng)).collect();
        let u: Vec<Fr> = (0..m).map(|_| Fr::rand(&mut rng)).collect();
        let s0 = reciprocal_witness(&a, &u, m);
        let r = Fr::rand(&mut rng);
        let r_inv = r.inverse().unwrap();
        let t_plus = horner(&s0, r);
        let t_minus = horner(&s0, r_inv);
        let l0 = two_point_remainder(r, t_plus, r_inv, t_minus)?;
        let mut s0m = s0.clone();
        s0m[0] -= l0[0];
        s0m[1] -= l0[1];
        let p_r = r + r_inv;
        let (w0, rem) = div_by_monic_quadratic(&s0m, p_r);
        assert_eq!(rem, [Fr::zero(); 2], "S0-L0 not divisible by Z_A at m={m}");
        // Check S0-L0 == Z_A W0 at random points.
        for _ in 0..3 {
            let x = Fr::rand(&mut rng);
            let z_a = x * x - p_r * x + Fr::one();
            let lhs = horner(&s0, x) - (l0[0] + l0[1] * x);
            let rhs = z_a * horner(&w0, x);
            assert_eq!(lhs, rhs);
        }
    }
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Tamper: statement mismatch
// ════════════════════════════════════════════════════════════════════

fn rejected(r: Result<bool, PCSError>) {
    assert!(matches!(r, Ok(false) | Err(_)), "expected rejection");
}

#[test]
fn test_reject_wrong_value() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(5);
    let poly = rand_poly(5, &mut rng);
    let point = rand_point(5, &mut rng);
    let com = NestedGridKzgPCS::<E>::commit(&ck, &poly)?;
    let (proof, value) = NestedGridKzgPCS::<E>::open(&ck, &poly, &point)?;
    rejected(NestedGridKzgPCS::<E>::verify(
        &vk,
        &com,
        &point,
        &(value + Fr::one()),
        &proof,
    ));
    Ok(())
}

#[test]
fn test_reject_wrong_point() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(6);
    let poly = rand_poly(6, &mut rng);
    let point = rand_point(6, &mut rng);
    let com = NestedGridKzgPCS::<E>::commit(&ck, &poly)?;
    let (proof, value) = NestedGridKzgPCS::<E>::open(&ck, &poly, &point)?;
    let mut wrong = point.clone();
    wrong[0] += Fr::one();
    rejected(NestedGridKzgPCS::<E>::verify(
        &vk, &com, &wrong, &value, &proof,
    ));
    Ok(())
}

#[test]
fn test_reject_wrong_commitment() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(6);
    let poly = rand_poly(6, &mut rng);
    let poly2 = rand_poly(6, &mut rng);
    let point = rand_point(6, &mut rng);
    let com2 = NestedGridKzgPCS::<E>::commit(&ck, &poly2)?;
    let (proof, value) = NestedGridKzgPCS::<E>::open(&ck, &poly, &point)?;
    rejected(NestedGridKzgPCS::<E>::verify(
        &vk, &com2, &point, &value, &proof,
    ));
    Ok(())
}

macro_rules! tamper_g1_test {
    ($name:ident, $field:ident) => {
        #[test]
        fn $name() -> Result<(), PCSError> {
            let mut rng = test_rng();
            let (ck, vk) = setup(6);
            let poly = rand_poly(6, &mut rng);
            let point = rand_point(6, &mut rng);
            let com = NestedGridKzgPCS::<E>::commit(&ck, &poly)?;
            let (mut proof, value) = NestedGridKzgPCS::<E>::open(&ck, &poly, &point)?;
            proof.$field = (proof.$field.into_group() * Fr::from(2u64)).into_affine();
            rejected(NestedGridKzgPCS::<E>::verify(
                &vk, &com, &point, &value, &proof,
            ));
            Ok(())
        }
    };
}
tamper_g1_test!(test_tamper_cm_s0, cm_s0);
tamper_g1_test!(test_tamper_cm_s1, cm_s1);
tamper_g1_test!(test_tamper_pi_x, pi_x);
tamper_g1_test!(test_tamper_pi_y, pi_y);

macro_rules! tamper_scalar_test {
    ($name:ident, $field:ident) => {
        #[test]
        fn $name() -> Result<(), PCSError> {
            let mut rng = test_rng();
            let (ck, vk) = setup(6);
            let poly = rand_poly(6, &mut rng);
            let point = rand_point(6, &mut rng);
            let com = NestedGridKzgPCS::<E>::commit(&ck, &poly)?;
            let (mut proof, value) = NestedGridKzgPCS::<E>::open(&ck, &poly, &point)?;
            proof.$field += Fr::one();
            rejected(NestedGridKzgPCS::<E>::verify(
                &vk, &com, &point, &value, &proof,
            ));
            Ok(())
        }
    };
}
tamper_scalar_test!(test_tamper_a_plus, a_plus);
tamper_scalar_test!(test_tamper_a_minus, a_minus);
tamper_scalar_test!(test_tamper_t0_plus, t0_plus);
tamper_scalar_test!(test_tamper_v_pp, v_pp);
tamper_scalar_test!(test_tamper_v_pn, v_pn);
tamper_scalar_test!(test_tamper_v_np, v_np);
tamper_scalar_test!(test_tamper_v_nn, v_nn);
tamper_scalar_test!(test_tamper_t1_plus, t1_plus);

#[test]
fn test_swap_two_grid_values() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(6);
    let poly = rand_poly(6, &mut rng);
    let point = rand_point(6, &mut rng);
    let com = NestedGridKzgPCS::<E>::commit(&ck, &poly)?;
    let (mut proof, value) = NestedGridKzgPCS::<E>::open(&ck, &poly, &point)?;
    if proof.v_pp != proof.v_nn {
        core::mem::swap(&mut proof.v_pp, &mut proof.v_nn);
        rejected(NestedGridKzgPCS::<E>::verify(
            &vk, &com, &point, &value, &proof,
        ));
    }
    Ok(())
}

#[test]
fn test_reject_wrong_mu() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(6);
    let poly = rand_poly(6, &mut rng);
    let point = rand_point(6, &mut rng);
    let com = NestedGridKzgPCS::<E>::commit(&ck, &poly)?;
    let (mut proof, value) = NestedGridKzgPCS::<E>::open(&ck, &poly, &point)?;
    proof.mu = 5;
    rejected(NestedGridKzgPCS::<E>::verify(
        &vk, &com, &point, &value, &proof,
    ));
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Panic-safety: malicious / malformed inputs must not panic
// ════════════════════════════════════════════════════════════════════

fn assert_no_panic_verify(
    vk: &NestedGridKzgVerifierParam<E>,
    com: &Commitment<E>,
    point: &[Fr],
    value: &Fr,
    proof: &NestedGridKzgProof<E>,
) {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        NestedGridKzgPCS::<E>::verify(vk, com, &point.to_vec(), value, proof)
    }));
    match res {
        Ok(verdict) => assert!(
            verdict.is_err() || !verdict.unwrap(),
            "malicious input must fail without panic"
        ),
        Err(_) => panic!("verify panicked on malicious input"),
    }
}

#[test]
fn test_srs_gen_rejects_small_nv_no_panic() {
    let mut rng = test_rng();
    for nv in [0usize, 1, 2, 3] {
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            NestedGridKzgPCS::<E>::gen_srs_for_testing(&mut rng, nv)
        }));
        assert!(matches!(res, Ok(Err(_))), "nv={nv} must Err, not panic");
    }
}

#[test]
fn test_verify_malicious_mu_values() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(6);
    let poly = rand_poly(6, &mut rng);
    let point = rand_point(6, &mut rng);
    let com = NestedGridKzgPCS::<E>::commit(&ck, &poly)?;
    let (base_proof, value) = NestedGridKzgPCS::<E>::open(&ck, &poly, &point)?;
    for bad_mu in [0u32, 3, u32::MAX, usize::BITS, 60] {
        let mut proof = base_proof.clone();
        proof.mu = bad_mu;
        assert_no_panic_verify(&vk, &com, &point, &value, &proof);
    }
    Ok(())
}

#[test]
fn test_verify_wrong_point_length_no_panic() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck, vk) = setup(6);
    let poly = rand_poly(6, &mut rng);
    let point = rand_point(6, &mut rng);
    let com = NestedGridKzgPCS::<E>::commit(&ck, &poly)?;
    let (proof, value) = NestedGridKzgPCS::<E>::open(&ck, &poly, &point)?;
    for len in [0usize, 2, 5, 7, 12] {
        let p = rand_point(len, &mut rng);
        let res = NestedGridKzgPCS::<E>::verify(&vk, &com, &p, &value, &proof);
        assert!(res.is_err() || !res.unwrap(), "len {len} must be rejected");
    }
    Ok(())
}

#[test]
fn test_verifier_key_capacity_insufficient() -> Result<(), PCSError> {
    let mut rng = test_rng();
    // Prove at nv=8 but verify with a key trimmed to nv=6.
    let srs = NestedGridKzgPCS::<E>::gen_srs_for_testing(&mut rng, 8)?;
    let (ck8, _vk8) = NestedGridKzgPCS::<E>::trim(&srs, None, Some(8))?;
    let (_ck6, vk6) = NestedGridKzgPCS::<E>::trim(&srs, None, Some(6))?;
    let poly = rand_poly(8, &mut rng);
    let point = rand_point(8, &mut rng);
    let com = NestedGridKzgPCS::<E>::commit(&ck8, &poly)?;
    let (proof, value) = NestedGridKzgPCS::<E>::open(&ck8, &poly, &point)?;
    let res = NestedGridKzgPCS::<E>::verify(&vk6, &com, &point, &value, &proof);
    assert!(res.is_err(), "vk capacity too small must Err");
    Ok(())
}

#[test]
fn test_prover_key_too_small() -> Result<(), PCSError> {
    let mut rng = test_rng();
    // SRS supports only nv=4; requesting nv=6 must fail.
    let srs = NestedGridKzgPCS::<E>::gen_srs_for_testing(&mut rng, 4)?;
    assert!(NestedGridKzgPCS::<E>::trim(&srs, None, Some(6)).is_err());
    // Dimension mismatch between prover param and polynomial must fail.
    let big = NestedGridKzgPCS::<E>::gen_srs_for_testing(&mut rng, 8)?;
    let (ck4, _) = NestedGridKzgPCS::<E>::trim(&big, None, Some(4))?;
    let poly6 = rand_poly(6, &mut rng);
    assert!(NestedGridKzgPCS::<E>::commit(&ck4, &poly6).is_err());
    let point6 = rand_point(6, &mut rng);
    assert!(NestedGridKzgPCS::<E>::open(&ck4, &poly6, &point6).is_err());
    Ok(())
}

#[test]
fn test_srs_trim_illegal_sizes() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let srs = NestedGridKzgPCS::<E>::gen_srs_for_testing(&mut rng, 8)?;
    assert!(NestedGridKzgPCS::<E>::trim(&srs, None, Some(3)).is_err());
    assert!(NestedGridKzgPCS::<E>::trim(&srs, None, Some(0)).is_err());
    assert!(NestedGridKzgPCS::<E>::trim(&srs, None, Some(9)).is_err());
    Ok(())
}

#[test]
fn test_two_point_remainder_duplicate_points_errors() {
    let mut rng = test_rng();
    let x = Fr::rand(&mut rng);
    let y0 = Fr::rand(&mut rng);
    let y1 = Fr::rand(&mut rng);
    assert!(two_point_remainder(x, y0, x, y1).is_err());
}

#[test]
fn test_transcript_statement_binding() -> Result<(), PCSError> {
    // A proof made for one (poly, point) must not verify against a different
    // committed statement even if the claimed value happens to match.
    let mut rng = test_rng();
    let (ck, vk) = setup(6);
    let poly_a = rand_poly(6, &mut rng);
    let point_a = rand_point(6, &mut rng);
    let _com_a = NestedGridKzgPCS::<E>::commit(&ck, &poly_a)?;
    let (proof_a, value_a) = NestedGridKzgPCS::<E>::open(&ck, &poly_a, &point_a)?;

    // different commitment
    let poly_b = rand_poly(6, &mut rng);
    let com_b = NestedGridKzgPCS::<E>::commit(&ck, &poly_b)?;
    rejected(NestedGridKzgPCS::<E>::verify(
        &vk, &com_b, &point_a, &value_a, &proof_a,
    ));
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Batch (sum-check) opening / verification
// ════════════════════════════════════════════════════════════════════

fn batch_roundtrip(k: usize, points: Vec<Vec<Fr>>) -> Result<bool, PCSError> {
    let mut rng = test_rng();
    let nv = 6;
    let (ck, vk) = setup(nv);
    let polys: Vec<_> = (0..k).map(|_| rand_poly(nv, &mut rng)).collect();
    let evals: Vec<Fr> = polys
        .iter()
        .zip(points.iter())
        .map(|(p, pt)| p.evaluate(pt).unwrap())
        .collect();
    let comms: Vec<_> = polys
        .iter()
        .map(|p| NestedGridKzgPCS::<E>::commit(&ck, p).unwrap())
        .collect();
    let mut tp = IOPTranscript::new(b"batch-test");
    tp.append_field_element(b"init", &Fr::zero())?;
    let proof = NestedGridKzgPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
    let mut tv = IOPTranscript::new(b"batch-test");
    tv.append_field_element(b"init", &Fr::zero())?;
    NestedGridKzgPCS::<E>::batch_verify(&vk, &comms, &points, &proof, &mut tv)
}

#[test]
fn test_batch_k1() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let points = vec![rand_point(6, &mut rng)];
    assert!(batch_roundtrip(1, points)?);
    Ok(())
}

#[test]
fn test_batch_distinct_points() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let points: Vec<_> = (0..3).map(|_| rand_point(6, &mut rng)).collect();
    assert!(batch_roundtrip(3, points)?);
    Ok(())
}

#[test]
fn test_batch_repeated_points() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let pt = rand_point(6, &mut rng);
    let points = vec![pt.clone(), pt.clone(), pt];
    assert!(batch_roundtrip(3, points)?);
    Ok(())
}

#[test]
fn test_batch_rejects_wrong_eval() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 6;
    let (ck, vk) = setup(nv);
    let polys: Vec<_> = (0..3).map(|_| rand_poly(nv, &mut rng)).collect();
    let points: Vec<_> = (0..3).map(|_| rand_point(nv, &mut rng)).collect();
    let mut evals: Vec<Fr> = polys
        .iter()
        .zip(points.iter())
        .map(|(p, pt)| p.evaluate(pt).unwrap())
        .collect();
    let comms: Vec<_> = polys
        .iter()
        .map(|p| NestedGridKzgPCS::<E>::commit(&ck, p).unwrap())
        .collect();
    evals[0] += Fr::one();
    let mut tp = IOPTranscript::new(b"batch-test");
    tp.append_field_element(b"init", &Fr::zero())?;
    let proof = NestedGridKzgPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
    let mut tv = IOPTranscript::new(b"batch-test");
    tv.append_field_element(b"init", &Fr::zero())?;
    rejected(NestedGridKzgPCS::<E>::batch_verify(
        &vk, &comms, &points, &proof, &mut tv,
    ));
    Ok(())
}

#[test]
fn test_batch_rejects_wrong_commitment() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 6;
    let (ck, vk) = setup(nv);
    let polys: Vec<_> = (0..3).map(|_| rand_poly(nv, &mut rng)).collect();
    let points: Vec<_> = (0..3).map(|_| rand_point(nv, &mut rng)).collect();
    let evals: Vec<Fr> = polys
        .iter()
        .zip(points.iter())
        .map(|(p, pt)| p.evaluate(pt).unwrap())
        .collect();
    let mut comms: Vec<_> = polys
        .iter()
        .map(|p| NestedGridKzgPCS::<E>::commit(&ck, p).unwrap())
        .collect();
    let mut tp = IOPTranscript::new(b"batch-test");
    tp.append_field_element(b"init", &Fr::zero())?;
    let proof = NestedGridKzgPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp)?;
    // corrupt one commitment
    comms[1] = NestedGridKzgPCS::<E>::commit(&ck, &rand_poly(nv, &mut rng))?;
    let mut tv = IOPTranscript::new(b"batch-test");
    tv.append_field_element(b"init", &Fr::zero())?;
    rejected(NestedGridKzgPCS::<E>::batch_verify(
        &vk, &comms, &points, &proof, &mut tv,
    ));
    Ok(())
}

#[test]
fn test_batch_malformed_lengths() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 6;
    let (ck, _vk) = setup(nv);
    let polys: Vec<_> = (0..3).map(|_| rand_poly(nv, &mut rng)).collect();
    let points: Vec<_> = (0..2).map(|_| rand_point(nv, &mut rng)).collect(); // mismatched
    let evals: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();
    let mut tp = IOPTranscript::new(b"batch-test");
    tp.append_field_element(b"init", &Fr::zero())?;
    assert!(NestedGridKzgPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tp).is_err());
    Ok(())
}
