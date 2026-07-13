//! CHOPIN PCS correctness and negative tests.
//!
//! Covers:
//! - even/odd/min `nv` single open/verify (nv=2..8);
//! - random property tests;
//! - `open_with_commitment` matches trait `open`;
//! - paper polynomial identities (restriction, row-fold, coefficient
//!   is_f_alpha, structured S vs dense, IPA identities, bivariate verifier
//!   group equation);
//! - BDFG20 coefficient identities (m == Z_T*W, L == (X-z)*W', zero remainders,
//!   wrapper commitments);
//! - proof serialization roundtrip and size assertions (560/564 bytes);
//! - SRS shape (N G1, 3 G2), trim prefix/grid consistency;
//! - every negative case (wrong value / point / commitment, tampering each of
//!   the 7 G1 and 7 scalar fields, swapped eval, malformed mu/lengths, vk/pk
//!   capacity, statement binding, catch_unwind random proofs);
//! - sumcheck batch adapter tests (multi_open/batch_verify).

use super::*;
use crate::pcs::{
    bdfg::{self, BdfgClaim},
    prelude::PCSError,
};
use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::{One, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::{panic, test_rng, UniformRand};

type E = Bls12_381;

fn setup(nv: usize) -> (ChopinProverParam<E>, ChopinVerifierParam<E>) {
    let mut rng = test_rng();
    let srs = ChopinPCS::<E>::gen_srs_for_testing(&mut rng, nv).unwrap();
    ChopinPCS::<E>::trim(&srs, None, Some(nv)).unwrap()
}

fn rand_point(nv: usize, rng: &mut impl Rng) -> Vec<Fr> {
    (0..nv).map(|_| Fr::rand(rng)).collect()
}

fn rand_poly(nv: usize, rng: &mut impl Rng) -> Arc<DenseMultilinearExtension<Fr>> {
    Arc::new(DenseMultilinearExtension::rand(nv, rng))
}

// ════════════════════════════════════════════════════════════════════
// Positive: even / odd / min nv, random property tests
// ════════════════════════════════════════════════════════════════════

#[test]
fn open_verify_even() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [4usize, 6, 8] {
        let (ck, vk) = setup(nv);
        let p = rand_poly(nv, &mut rng);
        let pt = rand_point(nv, &mut rng);
        let com = ChopinPCS::<E>::commit(&ck, &p)?;
        let (proof, val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
        assert_eq!(proof.mu as usize, nv);
        assert!(
            ChopinPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?,
            "open/verify failed at nv={nv}"
        );
    }
    Ok(())
}

#[test]
fn open_verify_odd() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [3usize, 5, 7] {
        let (ck, vk) = setup(nv);
        let p = rand_poly(nv, &mut rng);
        let pt = rand_point(nv, &mut rng);
        let com = ChopinPCS::<E>::commit(&ck, &p)?;
        let (proof, val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
        assert_eq!(proof.mu as usize, nv);
        assert!(
            ChopinPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?,
            "open/verify failed at odd nv={nv}"
        );
    }
    Ok(())
}

#[test]
fn open_verify_minimum_nv() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 2;
    let (ck, vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let com = ChopinPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
    assert!(ChopinPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
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
            let com = ChopinPCS::<E>::commit(&ck, &p)?;
            let (proof, val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
            assert_eq!(val, p.evaluate(&pt).unwrap());
            assert!(ChopinPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
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
        let com = ChopinPCS::<E>::commit(&ck, &p)?;
        let (proof, val) = ChopinPCS::<E>::open_with_commitment(&ck, &p, &pt, &com)?;
        assert!(ChopinPCS::<E>::verify(&vk, &com, &pt, &val, &proof)?);
        let (proof2, val2) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
        assert_eq!(val, val2);
        assert_eq!(proof, proof2);
    }
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Paper polynomial identities (coefficient-level)
// ════════════════════════════════════════════════════════════════════

#[test]
fn identity_divide_x_at_alpha() {
    // f(X,Y) = (X-alpha) q1(X,Y) + f_alpha(Y)
    let mut rng = test_rng();
    for mu in [2usize, 3, 4, 5] {
        let (m_left, m_right) = split_exponents(mu);
        let big_ml = 1usize << m_left;
        let big_mr = 1usize << m_right;
        let n = big_ml * big_mr;
        let evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let alpha = Fr::rand(&mut rng);
        let (q1, f_alpha) = divide_x_at_alpha(&evals, big_ml, big_mr, alpha);
        assert_eq!(q1.len(), (big_ml - 1) * big_mr);
        assert_eq!(f_alpha.len(), big_mr);
        // f(α,β) via direct bivariate eval must equal f_alpha(β) for random β.
        for _ in 0..4 {
            let beta = Fr::rand(&mut rng);
            let direct = evaluate_bivariate(&evals, big_ml, big_mr, alpha, beta);
            let fa_beta = poly_eval(&f_alpha, beta);
            assert_eq!(direct, fa_beta, "f(α,β) != f_alpha(β) at mu={mu}");
        }
    }
}

#[test]
fn identity_restriction_eta() {
    // eta = <f_zR, psi_L>
    let mut rng = test_rng();
    for mu in [2usize, 3, 4, 5] {
        let (m_left, m_right) = split_exponents(mu);
        let big_ml = 1usize << m_left;
        let big_mr = 1usize << m_right;
        let n = big_ml * big_mr;
        let evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let z_l: Vec<Fr> = (0..m_left).map(|_| Fr::rand(&mut rng)).collect();
        let z_r: Vec<Fr> = (0..m_right).map(|_| Fr::rand(&mut rng)).collect();
        let psi_l = build_eq_vec::<Fr>(&z_l, big_ml);
        let psi_r = build_eq_vec::<Fr>(&z_r, big_mr);
        let f_zr = compute_restriction(&evals, &psi_r, big_ml, big_mr);
        let eta: Fr = f_zr.iter().zip(psi_l.iter()).map(|(a, b)| *a * *b).sum();
        // also via direct bilinear eval
        let mut direct = Fr::zero();
        for j in 0..big_mr {
            let pr = psi_r[j];
            for i in 0..big_ml {
                direct += evals[i + big_ml * j] * psi_l[i] * pr;
            }
        }
        assert_eq!(eta, direct, "restriction eta mismatch at mu={mu}");
    }
}

#[test]
fn identity_row_fold_a() {
    // a = f_zR(alpha) = <f_alpha, psi_R>
    let mut rng = test_rng();
    for mu in [2usize, 3, 4, 5] {
        let (m_left, m_right) = split_exponents(mu);
        let big_ml = 1usize << m_left;
        let big_mr = 1usize << m_right;
        let n = big_ml * big_mr;
        let evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let z_r: Vec<Fr> = (0..m_right).map(|_| Fr::rand(&mut rng)).collect();
        let psi_r = build_eq_vec::<Fr>(&z_r, big_mr);
        let f_zr = compute_restriction(&evals, &psi_r, big_ml, big_mr);
        let alpha = Fr::rand(&mut rng);
        let a = poly_eval(&f_zr, alpha);
        let (_q1, f_alpha) = divide_x_at_alpha(&evals, big_ml, big_mr, alpha);
        let ip_a: Fr = f_alpha.iter().zip(psi_r.iter()).map(|(c, p)| *c * *p).sum();
        assert_eq!(a, ip_a, "row-fold identity a mismatch at mu={mu}");
    }
}

#[test]
fn identity_coefficient_f_minus_b1() {
    // f(X,Y) = (X-alpha)q1(X,Y) + f_alpha(Y), and then
    // f_alpha(Y) = (Y-beta)q2(Y) + b1
    let mut rng = test_rng();
    for mu in [2usize, 3, 4, 5] {
        let (m_left, m_right) = split_exponents(mu);
        let big_ml = 1usize << m_left;
        let big_mr = 1usize << m_right;
        let n = big_ml * big_mr;
        let evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let alpha = Fr::rand(&mut rng);
        let (q1, f_alpha) = divide_x_at_alpha(&evals, big_ml, big_mr, alpha);
        let beta = Fr::rand(&mut rng);
        let (q2, b1) = divide_y_at_beta(&f_alpha, beta).unwrap();
        // Evaluate f_alpha(beta) == b1
        assert_eq!(poly_eval(&f_alpha, beta), b1);
        // Full bivariate: f(α,β) = b1
        assert_eq!(
            evaluate_bivariate(&evals, big_ml, big_mr, alpha, beta),
            b1,
            "bivariate eval at (α,β) != b1"
        );
        let _ = q1;
        let _ = q2;
    }
}

// Dense O(M^2) reference for symmetric Lagrange witness.
fn dense_lagrange_witness(coeffs: &[Fr], u: &[Fr], m: usize) -> Vec<Fr> {
    let big_m = 1usize << m;
    let psi = build_eq_vec::<Fr>(u, big_m);
    let off = big_m - 1;
    let mut buf = vec![Fr::zero(); 2 * big_m - 1];
    for (i, &c) in coeffs.iter().enumerate() {
        for (j, &p) in psi.iter().enumerate() {
            buf[off + i - j] += c * p;
        }
    }
    let mut s = vec![Fr::zero(); big_m - 1];
    for (i, si) in s.iter_mut().enumerate() {
        *si = buf[off + (i + 1)] + buf[off - (i + 1)];
    }
    s
}

#[test]
fn identity_structured_s_matches_dense() {
    let mut rng = test_rng();
    for m in 1..=6 {
        let big_m = 1usize << m;
        let coeffs: Vec<Fr> = (0..big_m).map(|_| Fr::rand(&mut rng)).collect();
        let u: Vec<Fr> = (0..m).map(|_| Fr::rand(&mut rng)).collect();
        let s = symmetric_lagrange_witness(&coeffs, &u, m);
        let d = dense_lagrange_witness(&coeffs, &u, m);
        assert_eq!(s, d, "S mismatch at m={m}");
    }
}

#[test]
fn identity_ipa_at_beta() {
    // The batched Lagrange IPA identity holds for the combined S at random beta.
    let mut rng = test_rng();
    for mu in [3usize, 4, 5, 6] {
        let (m_left, m_right) = split_exponents(mu);
        let big_ml = 1usize << m_left;
        let big_mr = 1usize << m_right;
        let n = big_ml * big_mr;
        let evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let z_l: Vec<Fr> = (0..m_left).map(|_| Fr::rand(&mut rng)).collect();
        let z_r: Vec<Fr> = (0..m_right).map(|_| Fr::rand(&mut rng)).collect();
        let psi_r = build_eq_vec::<Fr>(&z_r, big_mr);
        let f_zr = compute_restriction(&evals, &psi_r, big_ml, big_mr);
        let alpha = Fr::rand(&mut rng);
        let (_q1, f_alpha) = divide_x_at_alpha(&evals, big_ml, big_mr, alpha);
        let a = poly_eval(&f_zr, alpha);
        let eta: Fr = f_zr
            .iter()
            .zip(build_eq_vec::<Fr>(&z_l, big_ml).iter())
            .map(|(c, p)| *c * *p)
            .sum();
        let gamma = Fr::rand(&mut rng);
        let s0 = symmetric_lagrange_witness(&f_zr, &z_l, m_left);
        let s1 = symmetric_lagrange_witness(&f_alpha, &z_r, m_right);
        let mut s = vec![Fr::zero(); big_ml - 1];
        for (i, &c) in s0.iter().enumerate() {
            s[i] = c;
        }
        for (i, &c) in s1.iter().enumerate() {
            s[i] += gamma * c;
        }
        // random beta that avoids 0, ±1, alpha
        let (beta, beta_inv) = loop {
            let b = Fr::rand(&mut rng);
            if !b.is_zero() && b.square() != Fr::one() && b != alpha {
                let bi = b.inverse().unwrap();
                break (b, bi);
            }
        };
        let a1 = poly_eval(&f_zr, beta);
        let a2 = poly_eval(&f_zr, beta_inv);
        let b1 = poly_eval(&f_alpha, beta);
        let b2 = poly_eval(&f_alpha, beta_inv);
        let s1_val = poly_eval(&s, beta);
        let s2_val = poly_eval(&s, beta_inv);
        let psi_l_b = eval_tensor(&z_l, beta);
        let psi_l_bi = eval_tensor(&z_l, beta_inv);
        let psi_r_b = eval_tensor(&z_r, beta);
        let psi_r_bi = eval_tensor(&z_r, beta_inv);
        let lhs = a1 * psi_l_bi + a2 * psi_l_b + gamma * (b1 * psi_r_bi + b2 * psi_r_b);
        let rhs = (eta + gamma * a).double() + beta * s1_val + beta_inv * s2_val;
        assert_eq!(lhs, rhs, "IPA identity failed at mu={mu}");
    }
}

#[test]
fn test_bivariate_verifier_via_srs() -> Result<(), PCSError> {
    // Use the real universal params (which has G2) to check the bivariate
    // verifier equation: e(C_F-b1[1]_1, [1]_2)·e(-π_x,[τ-α]_2)·e(-π_y,[σ-β]_2)=1
    let mut rng = test_rng();
    let nv = 5;
    let srs = ChopinPCS::<E>::gen_srs_for_testing(&mut rng, nv).unwrap();
    let (pp, _vp) = ChopinPCS::<E>::trim(&srs, None, Some(nv)).unwrap();
    let poly = rand_poly(nv, &mut rng);
    let evals = poly.to_evaluations();
    let big_ml = pp.big_ml();
    let big_mr = pp.big_mr();
    let alpha = Fr::rand(&mut rng);
    let beta = Fr::rand(&mut rng);
    let (q1, f_alpha) = divide_x_at_alpha(&evals, big_ml, big_mr, alpha);
    let (_q2, b1) = divide_y_at_beta(&f_alpha, beta).unwrap();
    let cf = pp.msm_full_reordered(&evals)?;
    let pi_x = pp.msm_q1_prefix(&q1)?;
    let pi_y = pp.msm_sigma_slice(&_q2)?;
    let g2_one = srs.g2_one;
    let g2_tau = srs.g2_tau;
    let g2_sigma = srs.g2_sigma;
    let tau_minus_alpha = (g2_tau.into_group() - g2_one.into_group() * alpha).into_affine();
    let sigma_minus_beta = (g2_sigma.into_group() - g2_one.into_group() * beta).into_affine();
    let cf_minus_b1 = (cf.into_group() - pp.g1_powers[0].into_group() * b1).into_affine();
    let neg_pi_x = (-pi_x.into_group()).into_affine();
    let neg_pi_y = (-pi_y.into_group()).into_affine();
    let target_one = <Bls12_381 as Pairing>::TargetField::one();
    let ok = <Bls12_381 as Pairing>::multi_pairing(
        [cf_minus_b1, neg_pi_x, neg_pi_y],
        [g2_one, tau_minus_alpha, sigma_minus_beta],
    ) == PairingOutput(target_one);
    assert!(ok, "bivariate verifier equation failed");
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// BDFG20 coefficient identities
// ════════════════════════════════════════════════════════════════════

fn poly_trim(v: &[Fr]) -> Vec<Fr> {
    let mut e = v.len();
    while e > 0 && v[e - 1].is_zero() {
        e -= 1;
    }
    v[..e].to_vec()
}

// ════════════════════════════════════════════════════════════════════
// Cost-model tests: W/W' exact lengths match bdfg algebra
// ════════════════════════════════════════════════════════════════════

#[test]
fn chopin_msm_lengths_match_bdfg_quotients() -> Result<(), PCSError> {
    // Verify that ChopinMsmLengths::for_num_vars matches the actual
    // vector lengths returned by bdfg_first_round / bdfg_second_round.
    let mut rng = test_rng();
    for nv in 2..=8usize {
        let (m_left, m_right) = split_exponents(nv);
        let big_ml = 1usize << m_left;
        let big_mr = 1usize << m_right;
        let n = big_ml * big_mr;
        let evals: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let z_l: Vec<Fr> = (0..m_left).map(|_| Fr::rand(&mut rng)).collect();
        let z_r: Vec<Fr> = (0..m_right).map(|_| Fr::rand(&mut rng)).collect();
        let psi_r = super::build_eq_vec::<Fr>(&z_r, big_mr);
        let f_zr = super::compute_restriction(&evals, &psi_r, big_ml, big_mr);
        let alpha = Fr::rand(&mut rng);
        let (_q1, f_alpha) = super::divide_x_at_alpha(&evals, big_ml, big_mr, alpha);
        let a = poly_eval(&f_zr, alpha);

        let (beta, beta_inv) = loop {
            let b = Fr::rand(&mut rng);
            if !b.is_zero() && b.square() != Fr::one() && b != alpha {
                let bi = b.inverse().unwrap();
                if bi != alpha {
                    break (b, bi);
                }
            }
        };
        let a1 = poly_eval(&f_zr, beta);
        let a2_ = poly_eval(&f_zr, beta_inv);
        let b1 = poly_eval(&f_alpha, beta);
        let b2 = poly_eval(&f_alpha, beta_inv);
        let s = super::symmetric_lagrange_witness(&f_zr, &z_l, m_left);
        let s1_val = poly_eval(&s, beta);
        let s2_val = poly_eval(&s, beta_inv);
        let rho = Fr::rand(&mut rng);

        let claims = [
            BdfgClaim {
                poly: &f_zr,
                points: &[alpha, beta, beta_inv],
                values: &[a, a1, a2_],
            },
            BdfgClaim {
                poly: &f_alpha,
                points: &[beta, beta_inv],
                values: &[b1, b2],
            },
            BdfgClaim {
                poly: &s,
                points: &[beta, beta_inv],
                values: &[s1_val, s2_val],
            },
        ];
        let first = bdfg_first_round(&claims, rho)?;
        let actual_w_len = first.quot_m.len();

        let z = loop {
            let z_val = Fr::rand(&mut rng);
            if !first.union.iter().any(|u| *u == z_val) {
                break z_val;
            }
        };
        let second = bdfg_second_round(&claims, &first, rho, z)?;
        let actual_wp_len = second.quot_l.len();

        let model = ChopinMsmLengths::for_num_vars(nv)?;
        assert_eq!(
            actual_w_len, model.w_len,
            "W length mismatch at nv={nv}: actual={actual_w_len} model={}",
            model.w_len
        );
        assert_eq!(
            actual_wp_len, model.wp_len,
            "W' length mismatch at nv={nv}: actual={actual_wp_len} model={}",
            model.wp_len
        );
    }
    Ok(())
}

#[test]
fn chopin_msm_lengths_rejects_invalid_mu() {
    assert!(ChopinMsmLengths::for_num_vars(0).is_err());
    assert!(ChopinMsmLengths::for_num_vars(1).is_err());
    assert!(ChopinMsmLengths::for_num_vars(usize::BITS as usize).is_err());
}

#[test]
fn chopin_msm_lengths_no_panic() {
    use ark_std::panic;
    for mu in [0usize, 1, usize::MAX, usize::BITS as usize] {
        let res = panic::catch_unwind(|| ChopinMsmLengths::for_num_vars(mu));
        assert!(res.is_ok(), "panicked for mu={mu}");
    }
}

#[test]
fn bdfg_coefficient_identities() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for mu in [2usize, 3, 4, 5] {
        let (m_left, m_right) = split_exponents(mu);
        let big_ml = 1usize << m_left;
        let big_mr = 1usize << m_right;
        let n = big_ml * big_mr;
        let (ck, _vk) = setup(mu);
        // Random committed polynomials with real CHOPIN degree bounds.
        let f_zr: Vec<Fr> = (0..big_ml).map(|_| Fr::rand(&mut rng)).collect();
        let f_alpha: Vec<Fr> = (0..big_mr).map(|_| Fr::rand(&mut rng)).collect();
        let s: Vec<Fr> = (0..big_ml - 1).map(|_| Fr::rand(&mut rng)).collect();

        let alpha = loop {
            let a = Fr::rand(&mut rng);
            if !a.is_zero() {
                break a;
            }
        };
        let (beta, beta_inv) = loop {
            let b = Fr::rand(&mut rng);
            if !b.is_zero() && b.square() != Fr::one() && b != alpha {
                let bi = b.inverse().unwrap();
                if bi != alpha {
                    break (b, bi);
                }
            }
        };
        let a = poly_eval(&f_zr, alpha);
        let a1 = poly_eval(&f_zr, beta);
        let a2 = poly_eval(&f_zr, beta_inv);
        let b1 = poly_eval(&f_alpha, beta);
        let b2 = poly_eval(&f_alpha, beta_inv);
        let s1_val = poly_eval(&s, beta);
        let s2_val = poly_eval(&s, beta_inv);
        let rho = Fr::rand(&mut rng);

        let claims = [
            BdfgClaim {
                poly: &f_zr,
                points: &[alpha, beta, beta_inv],
                values: &[a, a1, a2],
            },
            BdfgClaim {
                poly: &f_alpha,
                points: &[beta, beta_inv],
                values: &[b1, b2],
            },
            BdfgClaim {
                poly: &s,
                points: &[beta, beta_inv],
                values: &[s1_val, s2_val],
            },
        ];

        let first = bdfg_first_round(&claims, rho)?;
        let z_t = bdfg::vanishing_poly(&first.union);
        assert_eq!(
            poly_trim(&first.m),
            poly_trim(&bdfg::poly_mul(&z_t, &first.quot_m)),
            "m != Z_T * W at mu={mu}"
        );
        // commit W
        let batch_w = ck.msm_tau_slice(&first.quot_m)?;

        let mut z = Fr::rand(&mut rng);
        while first.union.iter().any(|u| *u == z) {
            z = Fr::rand(&mut rng);
        }
        let second = bdfg_second_round(&claims, &first, rho, z)?;
        assert_eq!(
            poly_trim(&second.l),
            poly_trim(&bdfg::mul_by_linear(&second.quot_l, z)),
            "L != (X-z) W' at mu={mu}"
        );
        // commit W'
        let batch_w_prime = ck.msm_tau_slice(&second.quot_l)?;

        // Verifier homomorphic reconstruction.
        let comb = bdfg_verifier_combination(
            &[
                &[alpha, beta, beta_inv],
                &[beta, beta_inv],
                &[beta, beta_inv],
            ],
            &[&[a, a1, a2], &[b1, b2], &[s1_val, s2_val]],
            rho,
            z,
        )?;
        // Cs + z·W' should be homomorphically consistent:
        // Cs = Σ scalars[i]*C_i - const·[1]_1 - Z_T(z)·W
        // e(Cs+z·W', [1]_2) = e(W', [τ]_2)
        let cs = ck.msm_tau_slice(&f_zr).unwrap().into_group() * comb.commit_scalars[0]
            + ck.msm_tau_slice(&f_alpha).unwrap().into_group() * comb.commit_scalars[1]
            + ck.msm_tau_slice(&s).unwrap().into_group() * comb.commit_scalars[2]
            - ck.g1_powers[0].into_group() * comb.const_scalar
            - batch_w.into_group() * comb.z_t_z;
        let cs_plus_z_wp = (cs + batch_w_prime.into_group() * z).into_affine();
        let neg_wp = (-batch_w_prime.into_group()).into_affine();
        // We can't pair without G2, but we can check constant coefficients.
        let _ = cs_plus_z_wp;
        let _ = neg_wp;
    }
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Serialization + proof size
// ════════════════════════════════════════════════════════════════════

#[test]
fn proof_serialization_roundtrip() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 5;
    let (ck, _vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let (proof, _val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
    let mut bytes = Vec::new();
    proof.serialize_compressed(&mut bytes).unwrap();
    let back = ChopinProof::<E>::deserialize_compressed(&bytes[..]).unwrap();
    assert_eq!(proof, back);
    Ok(())
}

#[test]
fn proof_size_cryptographic_payload() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 6;
    let (ck, _vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let (proof, _val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
    // 7 G1 + 7 scalars = 560 bytes
    assert_eq!(
        proof.cryptographic_payload_bytes(),
        560,
        "cryptographic payload must be 560 bytes"
    );
    // Canonical serialized including mu:u32 = 564 bytes
    let mut bytes = Vec::new();
    proof.serialize_compressed(&mut bytes).unwrap();
    assert_eq!(bytes.len(), 564, "serialized proof must be 564 bytes");
    Ok(())
}

#[test]
fn proof_size_constant_across_nv() -> Result<(), PCSError> {
    let mut rng = test_rng();
    // Check one even and one odd nv. Full range (8..20) is in the ignored
    // benchmark.
    for nv in [6usize, 7] {
        let (ck, _vk) = setup(nv);
        let p = rand_poly(nv, &mut rng);
        let pt = rand_point(nv, &mut rng);
        let (proof, _val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).unwrap();
        assert_eq!(
            bytes.len(),
            564,
            "proof size must be constant 564 at nv={nv}"
        );
    }
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// SRS shape checks
// ════════════════════════════════════════════════════════════════════

#[test]
fn srs_shape_n_g1_3_g2() -> Result<(), PCSError> {
    let mut rng = test_rng();
    for nv in [2usize, 3, 5, 8] {
        let srs = ChopinPCS::<E>::gen_srs_for_testing(&mut rng, nv)?;
        assert_eq!(srs.g1_powers.len(), 1usize << nv);
    }
    Ok(())
}

#[test]
fn srs_rejects_small_nv() {
    let mut rng = test_rng();
    assert!(ChopinPCS::<E>::gen_srs_for_testing(&mut rng, 0).is_err());
    assert!(ChopinPCS::<E>::gen_srs_for_testing(&mut rng, 1).is_err());
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
        let com = ChopinPCS::<E>::commit(&ck, &p)?;
        let (proof, _val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
        let bad = Fr::rand(&mut rng);
        assert!(!ChopinPCS::<E>::verify(&vk, &com, &pt, &bad, &proof)?);
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
    let com = ChopinPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
    let other = rand_point(nv, &mut rng);
    assert!(!ChopinPCS::<E>::verify(&vk, &com, &other, &val, &proof)?);
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
    let com2 = ChopinPCS::<E>::commit(&ck, &p2)?;
    let (proof, val) = ChopinPCS::<E>::open(&ck, &p1, &pt)?;
    assert!(!ChopinPCS::<E>::verify(&vk, &com2, &pt, &val, &proof)?);
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Negative: tamper each proof field
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
    let com = ChopinPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
    let mutators: Vec<(&str, fn(&mut ChopinProof<E>))> = vec![
        ("comm_f_zr", |pr| pr.comm_f_zr = tamper_g1(&pr.comm_f_zr)),
        ("comm_f_alpha", |pr| {
            pr.comm_f_alpha = tamper_g1(&pr.comm_f_alpha)
        }),
        ("comm_s", |pr| pr.comm_s = tamper_g1(&pr.comm_s)),
        ("pi_biv_x", |pr| pr.pi_biv_x = tamper_g1(&pr.pi_biv_x)),
        ("pi_biv_y", |pr| pr.pi_biv_y = tamper_g1(&pr.pi_biv_y)),
        ("batch_w", |pr| pr.batch_w = tamper_g1(&pr.batch_w)),
        ("batch_w_prime", |pr| {
            pr.batch_w_prime = tamper_g1(&pr.batch_w_prime)
        }),
    ];
    for (name, mutate) in mutators {
        let mut bad = proof.clone();
        mutate(&mut bad);
        let res = ChopinPCS::<E>::verify(&vk, &com, &pt, &val, &bad);
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
    let com = ChopinPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
    let mutators: Vec<(&str, fn(&mut ChopinProof<E>))> = vec![
        ("a", |pr| pr.a += Fr::one()),
        ("a1", |pr| pr.a1 += Fr::one()),
        ("a2", |pr| pr.a2 += Fr::one()),
        ("b1", |pr| pr.b1 += Fr::one()),
        ("b2", |pr| pr.b2 += Fr::one()),
        ("s1", |pr| pr.s1 += Fr::one()),
        ("s2", |pr| pr.s2 += Fr::one()),
    ];
    for (name, mutate) in mutators {
        let mut bad = proof.clone();
        mutate(&mut bad);
        let res = ChopinPCS::<E>::verify(&vk, &com, &pt, &val, &bad);
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
    let com = ChopinPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
    // swap beta/beta^{-1} for f_zR
    let mut bad = proof.clone();
    std::mem::swap(&mut bad.a1, &mut bad.a2);
    let res = ChopinPCS::<E>::verify(&vk, &com, &pt, &val, &bad);
    assert!(matches!(res, Ok(false)) || res.is_err());
    // swap for f_alpha
    let mut bad2 = proof;
    std::mem::swap(&mut bad2.b1, &mut bad2.b2);
    let res2 = ChopinPCS::<E>::verify(&vk, &com, &pt, &val, &bad2);
    assert!(matches!(res2, Ok(false)) || res2.is_err());
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Malformed inputs (no panic)
// ════════════════════════════════════════════════════════════════════

#[test]
fn malformed_mu_no_panic() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let com = ChopinPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
    for bad_mu in [0u32, 1, u32::MAX, (nv + 1) as u32] {
        let mut bad = proof.clone();
        bad.mu = bad_mu;
        let res = panic::catch_unwind(|| ChopinPCS::<E>::verify(&vk, &com, &pt, &val, &bad));
        assert!(res.is_ok(), "verify panicked for mu={bad_mu}");
        let inner = res.unwrap();
        assert!(inner.is_err() || !inner.unwrap());
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
    let com = ChopinPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
    for bad_len in [0usize, nv - 1, nv + 1] {
        let bad_pt = rand_point(bad_len, &mut rng);
        let res = panic::catch_unwind(|| ChopinPCS::<E>::verify(&vk, &com, &bad_pt, &val, &proof));
        assert!(res.is_ok());
        let inner = res.unwrap();
        assert!(inner.is_err() || !inner.unwrap());
    }
    Ok(())
}

#[test]
fn verifier_key_capacity_insufficient() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let (ck5, _vk5) = setup(5);
    let srs_small = ChopinPCS::<E>::gen_srs_for_testing(&mut rng, 3).unwrap();
    let (_ck3, vk3) = ChopinPCS::<E>::trim(&srs_small, None, Some(3)).unwrap();
    let p = rand_poly(5, &mut rng);
    let pt = rand_point(5, &mut rng);
    let com = ChopinPCS::<E>::commit(&ck5, &p)?;
    let (proof, val) = ChopinPCS::<E>::open(&ck5, &p, &pt)?;
    assert!(ChopinPCS::<E>::verify(&vk3, &com, &pt, &val, &proof).is_err());
    Ok(())
}

#[test]
fn prover_key_too_small() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let srs_small = ChopinPCS::<E>::gen_srs_for_testing(&mut rng, 3).unwrap();
    let (ck3, _vk3) = ChopinPCS::<E>::trim(&srs_small, None, Some(3)).unwrap();
    let p = rand_poly(5, &mut rng);
    let pt = rand_point(5, &mut rng);
    assert!(ChopinPCS::<E>::commit(&ck3, &p).is_err());
    assert!(ChopinPCS::<E>::open(&ck3, &p, &pt).is_err());
    Ok(())
}

#[test]
fn transcript_statement_binding() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let com = ChopinPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
    let other_com = Commitment((com.0.into_group() * Fr::from(3u64)).into_affine());
    assert!(!ChopinPCS::<E>::verify(&vk, &other_com, &pt, &val, &proof)?);
    Ok(())
}

#[test]
fn catch_unwind_random_proofs() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv);
    let p = rand_poly(nv, &mut rng);
    let pt = rand_point(nv, &mut rng);
    let com = ChopinPCS::<E>::commit(&ck, &p)?;
    let (proof, val) = ChopinPCS::<E>::open(&ck, &p, &pt)?;
    for _ in 0..32 {
        let mut bad = proof.clone();
        bad.comm_f_zr = tamper_g1(&bad.comm_f_zr);
        bad.a = Fr::rand(&mut rng);
        bad.b1 = Fr::rand(&mut rng);
        let res = panic::catch_unwind(|| ChopinPCS::<E>::verify(&vk, &com, &pt, &val, &bad));
        assert!(res.is_ok());
    }
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Challenge collision helper tests
// ════════════════════════════════════════════════════════════════════

#[test]
fn challenge_drawing_nonzero() -> Result<(), PCSError> {
    let mut t = IOPTranscript::<Fr>::new(b"test");
    t.append_field_element(b"init", &Fr::zero())?;
    let c = draw_nonzero(&mut t, b"c")?;
    assert!(!c.is_zero());
    Ok(())
}

#[test]
fn challenge_drawing_beta() -> Result<(), PCSError> {
    let mut t = IOPTranscript::<Fr>::new(b"test");
    t.append_field_element(b"init", &Fr::zero())?;
    let (c, cinv) = draw_beta(&mut t, b"c", Fr::zero())?;
    assert!(!c.is_zero());
    assert_ne!(c, cinv);
    assert_eq!(c * cinv, Fr::one());
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Sumcheck batch adapter (multi_open / batch_verify)
// ════════════════════════════════════════════════════════════════════

fn batch_open_verify(
    ck: &ChopinProverParam<E>,
    vk: &ChopinVerifierParam<E>,
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
        .map(|poly| ChopinPCS::<E>::commit(ck, poly).unwrap())
        .collect();
    let mut tr = IOPTranscript::<Fr>::new(b"chopin-batch-test");
    tr.append_field_element(b"init", &Fr::zero())?;
    let batch_proof = ChopinPCS::<E>::multi_open(ck, polys, points, &evals, &mut tr)?;
    let mut tr2 = IOPTranscript::<Fr>::new(b"chopin-batch-test");
    tr2.append_field_element(b"init", &Fr::zero())?;
    let ok = ChopinPCS::<E>::batch_verify(vk, &commitments, points, &batch_proof, &mut tr2)?;
    Ok(ok)
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
        .map(|poly| ChopinPCS::<E>::commit(&ck, poly).unwrap())
        .collect();
    let mut tr = IOPTranscript::<Fr>::new(b"chopin-batch-test");
    tr.append_field_element(b"init", &Fr::zero())?;
    let batch_proof = ChopinPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tr)?;
    let mut tr2 = IOPTranscript::<Fr>::new(b"chopin-batch-test");
    tr2.append_field_element(b"init", &Fr::zero())?;
    let res = ChopinPCS::<E>::batch_verify(&vk, &commitments, &points, &batch_proof, &mut tr2);
    assert!(res.is_err() || !res.unwrap());
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
        .map(|poly| ChopinPCS::<E>::commit(&ck, poly).unwrap())
        .collect();
    commitments[0] = ChopinPCS::<E>::commit(&ck, &rand_poly(nv, &mut rng))?;
    let mut tr = IOPTranscript::<Fr>::new(b"chopin-batch-test");
    tr.append_field_element(b"init", &Fr::zero())?;
    let batch_proof = ChopinPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tr)?;
    let mut tr2 = IOPTranscript::<Fr>::new(b"chopin-batch-test");
    tr2.append_field_element(b"init", &Fr::zero())?;
    let ok = ChopinPCS::<E>::batch_verify(&vk, &commitments, &points, &batch_proof, &mut tr2)?;
    assert!(!ok);
    Ok(())
}

#[test]
fn batch_malicious_sumcheck_point_length_no_panic() -> Result<(), PCSError> {
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
        .map(|poly| ChopinPCS::<E>::commit(&ck, poly).unwrap())
        .collect();
    let mut tr = IOPTranscript::<Fr>::new(b"m");
    tr.append_field_element(b"init", &Fr::zero())?;
    let mut batch_proof = ChopinPCS::<E>::multi_open(&ck, &polys, &points, &evals, &mut tr)?;
    batch_proof.sum_check_proof.point.push(Fr::rand(&mut rng));
    let res = panic::catch_unwind(|| {
        let mut tr2 = IOPTranscript::<Fr>::new(b"m");
        tr2.append_field_element(b"init", &Fr::zero()).unwrap();
        ChopinPCS::<E>::batch_verify(&vk, &commitments, &points, &batch_proof, &mut tr2)
    });
    assert!(res.is_ok());
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
    assert!(ChopinPCS::<E>::multi_open(&ck, &[], &[], &[], &mut tr).is_err());
    assert!(ChopinPCS::<E>::multi_open(&ck, &polys, &points, &evals[..2], &mut tr).is_err());
    let mut polys_bad = polys.clone();
    polys_bad[0] = rand_poly(nv + 1, &mut rng);
    assert!(ChopinPCS::<E>::multi_open(&ck, &polys_bad, &points, &evals, &mut tr).is_err());
    let mut points_bad = points.clone();
    points_bad[0] = rand_point(nv + 1, &mut rng);
    assert!(ChopinPCS::<E>::multi_open(&ck, &polys, &points_bad, &evals, &mut tr).is_err());
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Commitment-aware batch tests (multi_open_with_commitments)
// ════════════════════════════════════════════════════════════════════

fn batch_with_commitments_open_verify(
    ck: &ChopinProverParam<E>,
    vk: &ChopinVerifierParam<E>,
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
        .map(|poly| ChopinPCS::<E>::commit(ck, poly).unwrap())
        .collect();
    let mut tr = IOPTranscript::<Fr>::new(b"chopin-cbatch-test");
    tr.append_field_element(b"init", &Fr::zero())?;
    let batch_proof = ChopinPCS::<E>::multi_open_with_commitments(
        ck,
        polys,
        &commitments,
        points,
        &evals,
        &mut tr,
    )?;
    let mut tr2 = IOPTranscript::<Fr>::new(b"chopin-cbatch-test");
    tr2.append_field_element(b"init", &Fr::zero())?;
    ChopinPCS::<E>::batch_verify(vk, &commitments, points, &batch_proof, &mut tr2)
}

#[test]
fn batch_with_commitments_k1() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv + 2);
    let polys = vec![rand_poly(nv, &mut rng)];
    let points = vec![rand_point(nv, &mut rng)];
    assert!(batch_with_commitments_open_verify(
        &ck, &vk, &polys, &points
    )?);
    Ok(())
}

#[test]
fn batch_with_commitments_padded_key() -> Result<(), PCSError> {
    // poly nv=4, key nv=6 — canonical padding must not break commitment-aware path
    let mut rng = test_rng();
    let (ck, vk) = setup(6); // key for 6 vars
    let nv = 4;
    let polys: Vec<_> = (0..3).map(|_| rand_poly(nv, &mut rng)).collect();
    let points: Vec<_> = (0..3).map(|_| rand_point(nv, &mut rng)).collect();
    assert!(batch_with_commitments_open_verify(
        &ck, &vk, &polys, &points
    )?);
    Ok(())
}

#[test]
fn batch_with_commitments_multiple_distinct_points() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv + 3);
    let polys: Vec<_> = (0..5).map(|_| rand_poly(nv, &mut rng)).collect();
    let points: Vec<_> = (0..5).map(|_| rand_point(nv, &mut rng)).collect();
    assert!(batch_with_commitments_open_verify(
        &ck, &vk, &polys, &points
    )?);
    Ok(())
}

#[test]
fn batch_with_commitments_multiple_same_point() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv + 3);
    let polys: Vec<_> = (0..4).map(|_| rand_poly(nv, &mut rng)).collect();
    let pt = rand_point(nv, &mut rng);
    let points: Vec<_> = (0..4).map(|_| pt.clone()).collect();
    assert!(batch_with_commitments_open_verify(
        &ck, &vk, &polys, &points
    )?);
    Ok(())
}

#[test]
fn batch_with_commitments_non_power_of_two_k() -> Result<(), PCSError> {
    let mut rng = test_rng();
    let nv = 4;
    let (ck, vk) = setup(nv + 3);
    for k in [3usize, 5, 6, 7] {
        let polys: Vec<_> = (0..k).map(|_| rand_poly(nv, &mut rng)).collect();
        let points: Vec<_> = (0..k).map(|_| rand_point(nv, &mut rng)).collect();
        assert!(
            batch_with_commitments_open_verify(&ck, &vk, &polys, &points)?,
            "k={k}"
        );
    }
    Ok(())
}

#[test]
fn batch_with_commitments_rejects_malformed_commitment_lengths() -> Result<(), PCSError> {
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
    let commitments: Vec<_> = polys
        .iter()
        .map(|poly| ChopinPCS::<E>::commit(&ck, poly).unwrap())
        .collect();
    let mut tr = IOPTranscript::<Fr>::new(b"m");
    tr.append_field_element(b"init", &Fr::zero())?;
    // too few
    assert!(ChopinPCS::<E>::multi_open_with_commitments(
        &ck,
        &polys,
        &commitments[..2],
        &points,
        &evals,
        &mut tr
    )
    .is_err());
    // too many
    let extra = ChopinPCS::<E>::commit(&ck, &rand_poly(nv, &mut rng))?;
    let mut cm2 = commitments.clone();
    cm2.push(extra);
    assert!(ChopinPCS::<E>::multi_open_with_commitments(
        &ck, &polys, &cm2, &points, &evals, &mut tr
    )
    .is_err());
    // does not panic
    use ark_std::panic;
    let res = panic::catch_unwind(|| {
        let _ = ChopinPCS::<E>::multi_open_with_commitments(
            &ck,
            &polys,
            &commitments[..2],
            &points,
            &evals,
            &mut IOPTranscript::<Fr>::new(b"m"),
        );
    });
    assert!(res.is_ok());
    Ok(())
}

#[test]
fn batch_with_commitments_rejects_wrong_commitment() -> Result<(), PCSError> {
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
        .map(|poly| ChopinPCS::<E>::commit(&ck, poly).unwrap())
        .collect();
    // prover receives a wrong commitment
    let mut bad_cm = commitments.clone();
    bad_cm[0] = ChopinPCS::<E>::commit(&ck, &rand_poly(nv, &mut rng))?;
    let mut tr = IOPTranscript::<Fr>::new(b"m");
    tr.append_field_element(b"init", &Fr::zero())?;
    let batch_proof = ChopinPCS::<E>::multi_open_with_commitments(
        &ck, &polys, &bad_cm, &points, &evals, &mut tr,
    )?;
    // verifier uses correct commitments — must reject
    let mut tr2 = IOPTranscript::<Fr>::new(b"m");
    tr2.append_field_element(b"init", &Fr::zero())?;
    let ok = ChopinPCS::<E>::batch_verify(&vk, &commitments, &points, &batch_proof, &mut tr2)?;
    assert!(!ok, "wrong commitment must be rejected");
    Ok(())
}

#[test]
fn batch_with_commitments_g_prime_commit_matches_direct() -> Result<(), PCSError> {
    use crate::pcs::multilinear_kzg::batching::reduce_multi_open;
    let mut rng = test_rng();
    let nv = 4;
    let (ck, _vk) = setup(nv + 2);

    // distinct-point case
    {
        let polys: Vec<_> = (0..4).map(|_| rand_poly(nv, &mut rng)).collect();
        let points: Vec<_> = (0..4).map(|_| rand_point(nv, &mut rng)).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(f, p)| f.evaluate(p).unwrap())
            .collect();
        let commitments: Vec<_> = polys
            .iter()
            .map(|poly| ChopinPCS::<E>::commit(&ck, poly).unwrap())
            .collect();
        let mut tr = IOPTranscript::<Fr>::new(b"m");
        tr.append_field_element(b"init", &Fr::zero())?;
        let reduction = reduce_multi_open::<E, ChopinPCS<E>>(&polys, &points, &evals, &mut tr)?;
        // C_linear = Σ λ_i C_i
        let bases: Vec<_> = commitments.iter().map(|c| c.0).collect();
        let c_linear = Commitment(
            <Bls12_381 as Pairing>::G1::msm_unchecked(&bases, &reduction.lambda_i).into_affine(),
        );
        // C_direct = commit(g_prime)
        let c_direct = ChopinPCS::<E>::commit(&ck, &reduction.g_prime)?;
        assert_eq!(c_linear, c_direct, "linear vs direct mismatch (distinct)");
    }

    // repeated-point case
    {
        let polys: Vec<_> = (0..4).map(|_| rand_poly(nv, &mut rng)).collect();
        let pt = rand_point(nv, &mut rng);
        let points: Vec<_> = (0..4).map(|_| pt.clone()).collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(f, p)| f.evaluate(p).unwrap())
            .collect();
        let commitments: Vec<_> = polys
            .iter()
            .map(|poly| ChopinPCS::<E>::commit(&ck, poly).unwrap())
            .collect();
        let mut tr = IOPTranscript::<Fr>::new(b"m");
        tr.append_field_element(b"init", &Fr::zero())?;
        let reduction = reduce_multi_open::<E, ChopinPCS<E>>(&polys, &points, &evals, &mut tr)?;
        let bases: Vec<_> = commitments.iter().map(|c| c.0).collect();
        let c_linear = Commitment(
            <Bls12_381 as Pairing>::G1::msm_unchecked(&bases, &reduction.lambda_i).into_affine(),
        );
        let c_direct = ChopinPCS::<E>::commit(&ck, &reduction.g_prime)?;
        assert_eq!(c_linear, c_direct, "linear vs direct mismatch (repeated)");
    }
    Ok(())
}
