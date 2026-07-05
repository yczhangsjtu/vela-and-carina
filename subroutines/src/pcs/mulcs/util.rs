//! Utility functions for Mulcs Claymore identity — polynomial ops in
//! coefficient form.

use ark_ff::Field;
use rayon::prelude::*;

/// A univariate polynomial c_0 + c_1·X + ... + c_d·X^d
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct UnivarPoly<F: Field> {
    pub coeffs: Vec<F>,
}

#[allow(dead_code)]
impl<F: Field> UnivarPoly<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        Self { coeffs }
    }

    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    pub fn evaluate(&self, x: F) -> F {
        let mut result = F::ZERO;
        for c in self.coeffs.iter().rev() {
            result = result * x + *c;
        }
        result
    }

    /// f(X) · X^{shift} + f(X) · r — structured product with single factor
    pub fn mul_shift_add(&self, shift: usize, r: F) -> UnivarPoly<F> {
        let n = self.coeffs.len();
        let new_len = n + shift;
        let mut new_coeffs = vec![F::ZERO; new_len];

        new_coeffs[shift..]
            .par_iter_mut()
            .zip(self.coeffs.par_iter())
            .for_each(|(dst, &c)| *dst += c);

        new_coeffs[..n]
            .par_iter_mut()
            .zip(self.coeffs.par_iter())
            .for_each(|(dst, &c)| *dst += c * r);

        UnivarPoly::new(new_coeffs)
    }

    /// P_eq(X) = Π_{k=1}^μ ((1-r_k)·X^{2^{k-1}} + r_k)
    pub fn structured_eq_product(mu: usize, r: &[F]) -> UnivarPoly<F> {
        assert_eq!(r.len(), mu);
        let mut poly = UnivarPoly::new(vec![F::one()]);
        for k in 0..mu {
            let shift = 1 << k;
            let a = F::one() - r[k]; // X^{shift} coefficient
            let b = r[k]; // constant coefficient
            let n = poly.coeffs.len();
            let new_len = n + shift;
            let mut new_coeffs = vec![F::ZERO; new_len];
            new_coeffs[..n]
                .par_iter_mut()
                .zip(poly.coeffs.par_iter())
                .for_each(|(dst, &c)| *dst += c * b);
            new_coeffs[shift..shift + n]
                .par_iter_mut()
                .zip(poly.coeffs.par_iter())
                .for_each(|(dst, &c)| *dst += c * a);
            poly = UnivarPoly::new(new_coeffs);
        }
        poly
    }

    /// Multiply f_v by structured_product_eq(r)
    pub fn mul_structured_eq(&self, mu: usize, r: &[F]) -> UnivarPoly<F> {
        assert_eq!(r.len(), mu);
        let mut result = self.clone();
        for k in 0..mu {
            let shift = 1 << k;
            let a = F::one() - r[k];
            let b = r[k];
            let n = result.coeffs.len();
            let new_len = n + shift;
            let mut new_coeffs = vec![F::ZERO; new_len];
            new_coeffs[..n]
                .par_iter_mut()
                .zip(result.coeffs.par_iter())
                .for_each(|(dst, &c)| *dst += c * b);
            new_coeffs[shift..shift + n]
                .par_iter_mut()
                .zip(result.coeffs.par_iter())
                .for_each(|(dst, &c)| *dst += c * a);
            result = UnivarPoly::new(new_coeffs);
        }
        result
    }

    /// Compute h(X) = f_v(X) · P_eq(X) − y · X^{N-1}
    /// This is the core Claymore helper polynomial.
    pub fn compute_h(f_v: &[F], mu: usize, r: &[F], y: F) -> UnivarPoly<F> {
        let n = 1 << mu;
        let f_poly = UnivarPoly::new(f_v.to_vec());
        let mut h = f_poly.mul_structured_eq(mu, r);
        let h_len = h.coeffs.len();
        if h_len > n - 1 {
            h.coeffs[n - 1] -= y;
        } else {
            h.coeffs.resize(n, F::ZERO);
            h.coeffs[n - 1] -= y;
        }
        h
    }

    /// Compute h̄(X): h̄_i = h_i / (γ^{N-1} - γ^i) for i≠N-1, δ at i=N-1
    pub fn compute_h_bar(h: &UnivarPoly<F>, gamma: F, n: usize, delta: F) -> UnivarPoly<F> {
        let len = h.coeffs.len();
        let max_pow = std::cmp::max(len, n);
        let mut gamma_pows = Vec::with_capacity(max_pow);
        let mut acc = F::one();
        for _ in 0..max_pow {
            gamma_pows.push(acc);
            acc *= gamma;
        }
        let gamma_n1 = gamma_pows[n - 1];

        let mut inv_denoms: Vec<F> = (0..len)
            .into_par_iter()
            .map(|i| {
                if i == n - 1 {
                    F::one()
                } else {
                    gamma_n1 - gamma_pows[i]
                }
            })
            .collect();

        ark_ff::batch_inversion(&mut inv_denoms);

        let coeffs: Vec<F> = h
            .coeffs
            .par_iter()
            .enumerate()
            .map(|(i, &hi)| {
                if i == n - 1 {
                    delta
                } else {
                    hi * inv_denoms[i]
                }
            })
            .collect();

        UnivarPoly::new(coeffs)
    }
}

/// Evaluate y = Σ coeffs[i] · eq(i, point) — multilinear evaluation from
/// coefficient form
#[allow(dead_code)]
pub(crate) fn eval_multilinear_from_coeffs<F: Field>(coeffs: &[F], point: &[F]) -> F {
    let mu = point.len();
    let n = 1 << mu;
    let mut result = F::ZERO;
    for i in 0..n {
        let mut t = coeffs[i];
        for k in 0..mu {
            t *= if ((i >> k) & 1) == 1 {
                point[k]
            } else {
                F::one() - point[k]
            };
        }
        result += t;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_std::{test_rng, UniformRand};

    #[test]
    fn test_compute_h_identity() {
        let mut rng = test_rng();
        let mu = 4;
        let n = 1 << mu;
        let f_v: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let r: Vec<Fr> = (0..mu).map(|_| Fr::rand(&mut rng)).collect();
        let y = eval_multilinear_from_coeffs(&f_v, &r);

        let h = UnivarPoly::compute_h(&f_v, mu, &r, y);

        // Claymore identity: γ^{N-1}·h̄(z) - h̄(γz) =
        // z^{N-1}·(f(z)·Π((1-r_k)+r_k·z^{-2^k}) - y) For h at N: h_{N-1} should
        // be zero when y matches the evaluation
        assert_eq!(
            h.coeffs[n - 1],
            Fr::ZERO,
            "h_{{N-1}} should vanish when y = f_tilde(r)"
        );
    }

    #[test]
    fn test_h_bar_identity() {
        let mut rng = test_rng();
        let mu = 4;
        let n = 1 << mu;
        let gamma = Fr::rand(&mut rng);
        let delta = Fr::rand(&mut rng);

        let h = UnivarPoly::new((0..2 * n - 1).map(|_| Fr::rand(&mut rng)).collect());
        let h_bar = UnivarPoly::compute_h_bar(&h, gamma, n, delta);

        let z = Fr::rand(&mut rng);
        let gamma_n1 = gamma.pow([(n - 1) as u64]);
        let z_n1 = z.pow([(n - 1) as u64]);

        let lhs = gamma_n1 * h_bar.evaluate(z) - h_bar.evaluate(gamma * z);
        let rhs = h.evaluate(z) - h.coeffs[n - 1] * z_n1;
        assert_eq!(lhs, rhs, "h̄ identity check");
    }
}
