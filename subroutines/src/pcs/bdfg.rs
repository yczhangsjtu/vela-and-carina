//! Shared BDFG20 multi-polynomial / multi-point batch-opening algebra
//! (Boneh, Drake, Fisch, Gabizon, ePrint 2020/081, §4; the protocol drawn in
//! Chopin Figure 6).
//!
//! Both [`crate::pcs::mercury`] and [`crate::pcs::chopin`] batch several
//! univariate polynomials, each opened at its own (multi-point) evaluation set,
//! into a single KZG batch proof with two witness commitments `W`, `W'`. The
//! two schemes differ only in *which* claims they batch and in the transcript
//! labels / SRS slice used to commit; the underlying polynomial algebra is
//! identical. Rather than keep two near-identical private copies, that algebra
//! lives here once.
//!
//! This module is deliberately transcript-free and commitment-free: it operates
//! purely on coefficient vectors. Commitments, the SRS slice, and the
//! Fiat-Shamir labels are supplied by the caller (Mercury / Chopin), so this
//! module never depends on a concrete `ProverParam`.
//!
//! # Protocol (Figure 6)
//! Given claims `{(p_t, S_t, values_t)}_{t in [m]}` with union `T = ∪_t S_t`:
//!
//! Round 1 (challenge `rho`):
//! ```text
//!   ℓ_t   = interpolant with ℓ_t(z) = values_t at z ∈ S_t
//!   m(X)  = Σ_t rho^{t-1} Z_{T\S_t}(X) (p_t(X) - ℓ_t(X))
//!   W(X)  = m(X) / Z_T(X)                    (exact; zero remainder)
//! ```
//! Round 2 (challenge `z`):
//! ```text
//!   m_z(X) = Σ_t rho^{t-1} Z_{T\S_t}(z) (p_t(X) - ℓ_t(z))
//!   L(X)   = m_z(X) - Z_T(z) W(X)
//!   W'(X)  = L(X) / (X - z)                   (exact; zero remainder)
//! ```
//! Verifier reconstructs
//! ```text
//!   C_s = Σ_t rho^{t-1} Z_{T\S_t}(z) C_t
//!         - (Σ_t rho^{t-1} Z_{T\S_t}(z) ℓ_t(z)) [1]_1
//!         - Z_T(z) W
//! ```
//! and checks `e(C_s + z W', [1]_2) · e(-W', [τ]_2) = 1_GT`.
//!
//! # Figure 6 vs Figure 7
//! This module implements **Figure 6** (two witnesses `W`, `W'`; the verifier
//! checks one 2-term pairing product). It does **not** implement Figure 7 (the
//! modified standard-model batch proof, which splits the witness identity into
//! two same-point KZG openings and sends an extra group element). See
//! `docs/hyperplonk_chopin_design.md`.

use crate::pcs::prelude::PCSError;
use ark_ff::Field;
use ark_std::{string::ToString, vec, vec::Vec, One, Zero};

// ════════════════════════════════════════════════════════════════════
// Polynomial primitives (shared, FFT-free, generic over any field)
// ════════════════════════════════════════════════════════════════════

/// Horner evaluation of a coefficient vector `sum_i coeffs[i] X^i`.
pub(crate) fn poly_eval<F: Field>(coeffs: &[F], x: F) -> F {
    let mut acc = F::zero();
    for c in coeffs.iter().rev() {
        acc = acc * x + *c;
    }
    acc
}

/// `p(X) * (X - root)` (returns a new vector one longer).
pub(crate) fn mul_by_linear<F: Field>(coeffs: &[F], root: F) -> Vec<F> {
    if coeffs.is_empty() {
        return Vec::new();
    }
    let mut out = vec![F::zero(); coeffs.len() + 1];
    for (i, &c) in coeffs.iter().enumerate() {
        out[i + 1] += c;
        out[i] -= root * c;
    }
    out
}

/// Divide `p(X)` by `(X - root)`, returning `(quotient, remainder = p(root))`.
pub(crate) fn divide_by_linear<F: Field>(coeffs: &[F], root: F) -> (Vec<F>, F) {
    if coeffs.is_empty() {
        return (Vec::new(), F::zero());
    }
    let n = coeffs.len();
    if n == 1 {
        return (Vec::new(), coeffs[0]);
    }
    let mut q = vec![F::zero(); n - 1];
    let mut carry = F::zero();
    for i in (0..n - 1).rev() {
        let c = coeffs[i + 1] + root * carry;
        q[i] = c;
        carry = c;
    }
    let rem = coeffs[0] + root * carry;
    (q, rem)
}

/// `dst[i] += scale * src[i]` (grows `dst` as needed).
pub(crate) fn add_scaled<F: Field>(dst: &mut Vec<F>, src: &[F], scale: F) {
    if scale.is_zero() {
        return;
    }
    if dst.len() < src.len() {
        dst.resize(src.len(), F::zero());
    }
    for (i, &c) in src.iter().enumerate() {
        dst[i] += scale * c;
    }
}

/// `p - q` (coefficient-wise, result length `max(|p|, |q|)`).
pub(crate) fn poly_sub<F: Field>(p: &[F], q: &[F]) -> Vec<F> {
    let mut out = p.to_vec();
    if out.len() < q.len() {
        out.resize(q.len(), F::zero());
    }
    for (i, &c) in q.iter().enumerate() {
        out[i] -= c;
    }
    out
}

/// `dst[0] -= c` (grows `dst` if empty).
pub(crate) fn subtract_const<F: Field>(dst: &mut Vec<F>, c: F) {
    if dst.is_empty() {
        dst.push(-c);
    } else {
        dst[0] -= c;
    }
}

/// Lagrange interpolation through `(xs[i], ys[i])`; `xs` must be pairwise
/// distinct. Supports the small point sets used by BDFG20.
pub(crate) fn lagrange_interpolate<F: Field>(xs: &[F], ys: &[F]) -> Result<Vec<F>, PCSError> {
    let n = xs.len();
    if n != ys.len() || n == 0 {
        return Err(PCSError::InvalidProof(
            "interpolation length mismatch".to_string(),
        ));
    }
    let mut coeffs = vec![F::zero(); n];
    for i in 0..n {
        let mut num = vec![F::one()];
        let mut denom = F::one();
        for j in 0..n {
            if j == i {
                continue;
            }
            num = mul_by_linear(&num, xs[j]);
            denom *= xs[i] - xs[j];
        }
        let inv = denom
            .inverse()
            .ok_or_else(|| PCSError::InvalidProof("duplicate interpolation nodes".to_string()))?;
        let scale = ys[i] * inv;
        for (k, &c) in num.iter().enumerate() {
            coeffs[k] += scale * c;
        }
    }
    Ok(coeffs)
}

/// Vanishing polynomial `prod_{r in roots} (X - r)`. Empty roots -> the
/// constant polynomial `1`.
pub(crate) fn vanishing_poly<F: Field>(roots: &[F]) -> Vec<F> {
    let mut acc = vec![F::one()];
    for &r in roots {
        acc = mul_by_linear(&acc, r);
    }
    acc
}

// ════════════════════════════════════════════════════════════════════
// Claim set and union
// ════════════════════════════════════════════════════════════════════

/// A single batch-opening claim: polynomial `poly` opens to `values[k]` at
/// `points[k]` for all `k`. `points` must be pairwise distinct.
pub(crate) struct BdfgClaim<'a, F: Field> {
    pub poly: &'a [F],
    pub points: &'a [F],
    pub values: &'a [F],
}

/// Union `T = ∪_t point_sets[t]` in deterministic first-seen order. Each
/// individual set must be pairwise distinct (rejected otherwise). Prover and
/// verifier feed the same claim order, so both obtain the same `T`.
pub(crate) fn union_points<F: Field>(point_sets: &[&[F]]) -> Result<Vec<F>, PCSError> {
    let mut union: Vec<F> = Vec::new();
    for set in point_sets {
        // pairwise-distinct within the set
        for (a, &pa) in set.iter().enumerate() {
            for &pb in set.iter().skip(a + 1) {
                if pa == pb {
                    return Err(PCSError::InvalidProof(
                        "BDFG20 evaluation set has a repeated point".to_string(),
                    ));
                }
            }
            if !union.iter().any(|&u| u == pa) {
                union.push(pa);
            }
        }
    }
    Ok(union)
}

/// Elements of `union` that are NOT in `points` (`T \ S_t`).
fn complement<F: Field>(union: &[F], points: &[F]) -> Vec<F> {
    union
        .iter()
        .copied()
        .filter(|u| !points.iter().any(|p| p == u))
        .collect()
}

// ════════════════════════════════════════════════════════════════════
// Prover round 1: m(X), W(X) = m / Z_T
// ════════════════════════════════════════════════════════════════════

/// Round-1 output. `interpolants[t]` is `ℓ_t`; `m` is the combined numerator;
/// `quot_m` is `W = m / Z_T`.
pub(crate) struct BdfgFirstRound<F: Field> {
    pub union: Vec<F>,
    pub interpolants: Vec<Vec<F>>,
    /// The combined numerator `m(X)`. Kept for coefficient-level tests
    /// (`m == Z_T * W`); the prover only needs `quot_m` for the commitment.
    #[allow(dead_code)]
    pub m: Vec<F>,
    pub quot_m: Vec<F>,
}

/// Compute `ℓ_t`, `m(X) = Σ_t rho^{t-1} Z_{T\S_t}(X)(p_t - ℓ_t)` and
/// `W = m / Z_T`. Errors (`InvalidProver`) if `m` is not divisible by `Z_T`
/// (i.e. some claimed value is inconsistent with its polynomial).
pub(crate) fn bdfg_first_round<F: Field>(
    claims: &[BdfgClaim<F>],
    rho: F,
) -> Result<BdfgFirstRound<F>, PCSError> {
    let point_sets: Vec<&[F]> = claims.iter().map(|c| c.points).collect();
    let union = union_points(&point_sets)?;

    let mut interpolants = Vec::with_capacity(claims.len());
    let mut m: Vec<F> = Vec::new();
    let mut rho_pow = F::one();
    for claim in claims {
        if claim.points.len() != claim.values.len() {
            return Err(PCSError::InvalidProof(
                "BDFG20 claim points/values length mismatch".to_string(),
            ));
        }
        let ell = lagrange_interpolate(claim.points, claim.values)?;
        let diff = poly_sub(claim.poly, &ell);
        let comp = complement(&union, claim.points);
        let z_comp = vanishing_poly(&comp);
        let term = poly_mul(&z_comp, &diff);
        add_scaled(&mut m, &term, rho_pow);
        interpolants.push(ell);
        rho_pow *= rho;
    }

    // W = m / Z_T (exact).
    let mut quot = m.clone();
    for &r in &union {
        let (q, rem) = divide_by_linear(&quot, r);
        if !rem.is_zero() {
            return Err(PCSError::InvalidProver(
                "BDFG20 m(X) not divisible by Z_T".to_string(),
            ));
        }
        quot = q;
    }

    Ok(BdfgFirstRound {
        union,
        interpolants,
        m,
        quot_m: quot,
    })
}

// ════════════════════════════════════════════════════════════════════
// Prover round 2: L(X), W'(X) = L / (X - z)
// ════════════════════════════════════════════════════════════════════

/// Round-2 output. `l` is `L(X)`; `quot_l` is `W' = L / (X - z)`.
pub(crate) struct BdfgSecondRound<F: Field> {
    /// `L(X)`. Kept for coefficient-level tests (`L == (X-z) W'`); the prover
    /// only needs `quot_l`.
    #[allow(dead_code)]
    pub l: Vec<F>,
    pub quot_l: Vec<F>,
}

/// Compute `m_z`, `L(X) = m_z - Z_T(z) W`, and `W' = L / (X - z)`. Errors
/// (`InvalidProver`) if `L` is not divisible by `(X - z)`.
pub(crate) fn bdfg_second_round<F: Field>(
    claims: &[BdfgClaim<F>],
    first: &BdfgFirstRound<F>,
    rho: F,
    z: F,
) -> Result<BdfgSecondRound<F>, PCSError> {
    let mut l: Vec<F> = Vec::new();
    let mut rho_pow = F::one();
    for (t, claim) in claims.iter().enumerate() {
        let comp = complement(&first.union, claim.points);
        let z_comp_z = poly_eval(&vanishing_poly(&comp), z);
        let ell_z = poly_eval(&first.interpolants[t], z);
        // rho^{t} Z_{T\S_t}(z) (p_t(X) - ℓ_t(z))
        add_scaled(&mut l, claim.poly, rho_pow * z_comp_z);
        subtract_const(&mut l, rho_pow * z_comp_z * ell_z);
        rho_pow *= rho;
    }
    // L = m_z - Z_T(z) W
    let z_t_z = poly_eval(&vanishing_poly(&first.union), z);
    add_scaled(&mut l, &first.quot_m, -z_t_z);

    let (quot_l, rem) = divide_by_linear(&l, z);
    if !rem.is_zero() {
        return Err(PCSError::InvalidProver(
            "BDFG20 L(X) not divisible by (X - z)".to_string(),
        ));
    }
    Ok(BdfgSecondRound { l, quot_l })
}

// ════════════════════════════════════════════════════════════════════
// Verifier reconstruction scalars
// ════════════════════════════════════════════════════════════════════

/// The scalars the verifier uses to reconstruct `C_s`:
/// `commit_scalars[t] = rho^{t-1} Z_{T\S_t}(z)` (multiplies `C_t`),
/// `const_scalar = Σ_t commit_scalars[t] ℓ_t(z)` (multiplies `[1]_1`),
/// `z_t_z = Z_T(z)` (multiplies `W`).
pub(crate) struct BdfgVerifierCombination<F: Field> {
    pub commit_scalars: Vec<F>,
    pub const_scalar: F,
    pub z_t_z: F,
}

/// Reconstruct the verifier combination from the claims' points/values only
/// (no polynomials). `point_sets`/`value_sets` MUST be in the same order the
/// prover used.
pub(crate) fn bdfg_verifier_combination<F: Field>(
    point_sets: &[&[F]],
    value_sets: &[&[F]],
    rho: F,
    z: F,
) -> Result<BdfgVerifierCombination<F>, PCSError> {
    if point_sets.len() != value_sets.len() {
        return Err(PCSError::InvalidProof(
            "BDFG20 verifier: points/values count mismatch".to_string(),
        ));
    }
    let union = union_points(point_sets)?;
    let z_t_z = poly_eval(&vanishing_poly(&union), z);

    let mut commit_scalars = Vec::with_capacity(point_sets.len());
    let mut const_scalar = F::zero();
    let mut rho_pow = F::one();
    for (points, values) in point_sets.iter().zip(value_sets.iter()) {
        if points.len() != values.len() {
            return Err(PCSError::InvalidProof(
                "BDFG20 verifier: claim points/values length mismatch".to_string(),
            ));
        }
        let comp = complement(&union, points);
        let z_comp_z = poly_eval(&vanishing_poly(&comp), z);
        let ell = lagrange_interpolate(points, values)?;
        let ell_z = poly_eval(&ell, z);
        let cs = rho_pow * z_comp_z;
        const_scalar += cs * ell_z;
        commit_scalars.push(cs);
        rho_pow *= rho;
    }
    Ok(BdfgVerifierCombination {
        commit_scalars,
        const_scalar,
        z_t_z,
    })
}

// ════════════════════════════════════════════════════════════════════
// Small helpers
// ════════════════════════════════════════════════════════════════════

/// Schoolbook polynomial multiplication (used only on tiny vanishing factors).
pub(crate) fn poly_mul<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![F::zero(); a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_std::{test_rng, UniformRand};

    fn poly_trim(v: &[Fr]) -> Vec<Fr> {
        let mut e = v.len();
        while e > 0 && v[e - 1].is_zero() {
            e -= 1;
        }
        v[..e].to_vec()
    }

    #[test]
    fn divide_and_mul_by_linear_are_inverse() {
        let mut rng = test_rng();
        for deg in 1..8 {
            let p: Vec<Fr> = (0..=deg).map(|_| Fr::rand(&mut rng)).collect();
            let root = Fr::rand(&mut rng);
            let prod = mul_by_linear(&p, root);
            let (q, rem) = divide_by_linear(&prod, root);
            assert!(rem.is_zero());
            assert_eq!(poly_trim(&q), poly_trim(&p));
            // remainder of divide equals p(root)
            let (_q2, rem2) = divide_by_linear(&p, root);
            assert_eq!(rem2, poly_eval(&p, root));
        }
    }

    #[test]
    fn interpolate_matches_values() {
        let mut rng = test_rng();
        let xs: Vec<Fr> = (0..4).map(|_| Fr::rand(&mut rng)).collect();
        let ys: Vec<Fr> = (0..4).map(|_| Fr::rand(&mut rng)).collect();
        let poly = lagrange_interpolate(&xs, &ys).unwrap();
        for (x, y) in xs.iter().zip(ys.iter()) {
            assert_eq!(poly_eval(&poly, *x), *y);
        }
    }

    #[test]
    fn vanishing_poly_vanishes() {
        let mut rng = test_rng();
        let roots: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();
        let z = vanishing_poly(&roots);
        for r in &roots {
            assert!(poly_eval(&z, *r).is_zero());
        }
        assert_eq!(z.len(), roots.len() + 1);
    }

    #[test]
    fn union_rejects_repeats_and_dedups() {
        let a = Fr::from(3u64);
        let b = Fr::from(5u64);
        let c = Fr::from(7u64);
        // repeated inside one set -> error
        assert!(union_points(&[&[a, a][..]]).is_err());
        // dedup across sets in first-seen order
        let u = union_points(&[&[a, b][..], &[b, c][..]]).unwrap();
        assert_eq!(u, vec![a, b, c]);
    }

    #[test]
    fn first_and_second_round_identities() {
        let mut rng = test_rng();
        // three polynomials with distinct multi-point sets, union of 3 points.
        let alpha = Fr::rand(&mut rng);
        let mut beta = Fr::rand(&mut rng);
        while beta == alpha || beta.is_zero() {
            beta = Fr::rand(&mut rng);
        }
        let beta_inv = beta.inverse().unwrap();
        let p0: Vec<Fr> = (0..6).map(|_| Fr::rand(&mut rng)).collect();
        let p1: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();
        let p2: Vec<Fr> = (0..4).map(|_| Fr::rand(&mut rng)).collect();
        let s0 = [alpha, beta, beta_inv];
        let s1 = [beta, beta_inv];
        let s2 = [beta, beta_inv];
        let v0 = [poly_eval(&p0, alpha), poly_eval(&p0, beta), poly_eval(&p0, beta_inv)];
        let v1 = [poly_eval(&p1, beta), poly_eval(&p1, beta_inv)];
        let v2 = [poly_eval(&p2, beta), poly_eval(&p2, beta_inv)];
        let claims = [
            BdfgClaim { poly: &p0, points: &s0, values: &v0 },
            BdfgClaim { poly: &p1, points: &s1, values: &v1 },
            BdfgClaim { poly: &p2, points: &s2, values: &v2 },
        ];
        let rho = Fr::rand(&mut rng);
        let first = bdfg_first_round(&claims, rho).unwrap();
        // m == Z_T * W
        let z_t = vanishing_poly(&first.union);
        assert_eq!(
            poly_trim(&first.m),
            poly_trim(&poly_mul(&z_t, &first.quot_m)),
            "m != Z_T * W"
        );
        let mut z = Fr::rand(&mut rng);
        while first.union.iter().any(|u| *u == z) {
            z = Fr::rand(&mut rng);
        }
        let second = bdfg_second_round(&claims, &first, rho, z).unwrap();
        // L == (X - z) W'
        assert_eq!(
            poly_trim(&second.l),
            poly_trim(&mul_by_linear(&second.quot_l, z)),
            "L != (X-z) W'"
        );
    }

    #[test]
    fn first_round_rejects_inconsistent_value() {
        let mut rng = test_rng();
        let p0: Vec<Fr> = (0..4).map(|_| Fr::rand(&mut rng)).collect();
        let a = Fr::from(2u64);
        let b = Fr::from(3u64);
        let good = [poly_eval(&p0, a), poly_eval(&p0, b)];
        let bad = [poly_eval(&p0, a) + Fr::one(), poly_eval(&p0, b)];
        let pts = [a, b];
        assert!(bdfg_first_round(&[BdfgClaim { poly: &p0, points: &pts, values: &good }], Fr::from(7u64)).is_ok());
        assert!(bdfg_first_round(&[BdfgClaim { poly: &p0, points: &pts, values: &bad }], Fr::from(7u64)).is_err());
    }
}
