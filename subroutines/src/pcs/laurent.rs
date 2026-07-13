//! Shared reciprocal / symmetric-Laurent kernel.
//!
//! Both [`crate::pcs::recipcs`] and [`crate::pcs::mercury`] rely on the same
//! FFT-free structured multiplication of a polynomial by the reciprocal of a
//! tensor-product ("eq"/Lagrange) polynomial. Rather than keep two nearly
//! identical private copies, the formula lives here once.
//!
//! For `coeffs(X) = sum_{i<N} a_i X^i` with `N = 2^m`, and a point `r in F^m`,
//! define the tensor polynomial
//!   T_r(X) = prod_{k=0}^{m-1} ((1-r_k) + r_k X^{2^k}),   [X^i] T_r = eq(i; r).
//! This routine returns the Laurent coefficient buffer of
//!   C(X) = coeffs(X) * T_r(1/X)
//! whose degrees range over `-(N-1) ..= (N-1)`. The buffer has length `2N-1`;
//! entry `offset + d` (with `offset = N-1`) holds the coefficient of `X^d`.
//!
//! The result is built by `m` structured shift-add passes, each touching the
//! whole `2N-1` buffer once, for `O(N*m)` field operations and no FFT. This is
//! what lets the reciprocal-style PCS backends stay generic over any prime
//! field without an FFT-friendly (2-adic) structure.

use ark_ff::Field;
use ark_std::{vec, vec::Vec};

/// Offset of the constant (degree-0) coefficient inside the buffer returned by
/// [`mul_by_reciprocal_tensor`]: `offset = 2^m - 1`.
#[inline]
pub(crate) fn laurent_offset(m: usize) -> usize {
    (1usize << m) - 1
}

/// Compute the Laurent buffer of `coeffs(X) * T_r(1/X)` (see module docs).
///
/// - `coeffs` must have at least `N = 2^m` entries; only `coeffs[..N]` is read.
/// - `r` must have at least `m` entries; only `r[..m]` is read.
///
/// Returns a buffer of length `2N-1`; index `offset + d` (offset = `N-1`) is
/// the coefficient of `X^d`, `d in -(N-1)..=(N-1)`.
pub(crate) fn mul_by_reciprocal_tensor<F: Field>(coeffs: &[F], m: usize, r: &[F]) -> Vec<F> {
    let n = 1usize << m;
    // Length invariants required by the algorithm. Callers (ReciPCS, Mercury)
    // always pass exactly-sized buffers; these assertions document and guard the
    // contract so a mis-sized internal call fails loudly in debug/test builds
    // rather than silently reading out of range or producing a wrong result.
    debug_assert!(
        coeffs.len() >= n,
        "mul_by_reciprocal_tensor: coeffs.len() {} < N = 2^{} = {}",
        coeffs.len(),
        m,
        n
    );
    debug_assert!(
        r.len() >= m,
        "mul_by_reciprocal_tensor: r.len() {} < m = {}",
        r.len(),
        m
    );
    let offset = n - 1;
    let len = 2 * n - 1;
    let mut buf = vec![F::zero(); len];
    // Seed with coeffs at non-negative degrees 0..N-1.
    buf[offset..(offset + n)].copy_from_slice(&coeffs[..n]);
    for (k, &rk) in r.iter().enumerate().take(m) {
        let s = 1usize << k;
        let omrk = F::one() - rk;
        let mut next = vec![F::zero(); len];
        for i in 0..len {
            let val = buf[i];
            if val.is_zero() {
                continue;
            }
            // multiply by ((1-r_k) + r_k X^{-2^k}): keep at same degree, and
            // shift down by 2^k (X^{-2^k}).
            next[i] += omrk * val;
            if i >= s {
                next[i - s] += rk * val;
            }
        }
        buf = next;
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::{One, Zero};
    use ark_std::{test_rng, UniformRand};

    // Direct O(N^2) Laurent multiplication reference.
    fn dense_reference(coeffs: &[Fr], m: usize, r: &[Fr]) -> Vec<Fr> {
        let n = 1usize << m;
        // T_r(1/X) coefficients as Laurent: [X^{-e}] with e = sum of chosen bits.
        // Build eq vector: eq[i] = prod_k (r_k if bit k else 1-r_k).
        let mut eq = vec![Fr::one(); n];
        for (i, e) in eq.iter_mut().enumerate() {
            let mut acc = Fr::one();
            for (k, &rk) in r.iter().enumerate().take(m) {
                acc *= if (i >> k) & 1 == 1 {
                    rk
                } else {
                    Fr::one() - rk
                };
            }
            *e = acc;
        }
        let offset = n - 1;
        let mut out = vec![Fr::zero(); 2 * n - 1];
        for (i, &a) in coeffs.iter().enumerate().take(n) {
            for (j, &e) in eq.iter().enumerate().take(n) {
                // a X^i * e X^{-j} = a e X^{i-j}
                let d = i as isize - j as isize;
                out[(offset as isize + d) as usize] += a * e;
            }
        }
        out
    }

    #[test]
    fn structured_matches_dense() {
        let mut rng = test_rng();
        for m in 1..=8 {
            let n = 1usize << m;
            let coeffs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let r: Vec<Fr> = (0..m).map(|_| Fr::rand(&mut rng)).collect();
            let got = mul_by_reciprocal_tensor(&coeffs, m, &r);
            let want = dense_reference(&coeffs, m, &r);
            assert_eq!(got, want, "mismatch at m={m}");
        }
    }
}
