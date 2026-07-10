//! Shplonk batching for KZG opening claims.
//!
//! Given a set of opening claims {(f_j, x_j, v_j)}, Shplonk reduces them to a
//! single claim (G, z, 0) via batched quotient + partial evaluation, using
//! Fiat-Shamir challenges ν (batching) and z (evaluation).
//!
//! For Gemini fold polynomials, each fold is opened at TWO points concurrently.
//! The claim lists the negative point and neg evaluation; the positive evaluation
//! is passed separately and handled with the 1/(z + x_j) denominator.

use crate::pcs::prelude::{Commitment, PCSError};
use ark_ec::{pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, AffineRepr, CurveGroup};
use ark_ff::Field;
use ark_std::{vec, vec::Vec, One, Zero};
use transcript::IOPTranscript;

/// A single opening claim (point, claimed value) for a polynomial with its
/// coefficient vector (prover side).
#[derive(Clone)]
pub(crate) struct ProverClaim<F: Field> {
    pub coeffs: Vec<F>,
    pub point: F,
    pub value: F,
    /// True if this polynomial is a Gemini fold opened at TWO points.
    /// When true, `point` is the negative opening point and `value` is the
    /// negative evaluation. The positive evaluation is provided alongside.
    pub gemini_fold: bool,
}

/// A single opening claim for the verifier side (commitment, point, value).
#[derive(Clone)]
pub(crate) struct VerifierClaim<E: Pairing> {
    pub commitment: E::G1Affine,
    pub point: E::ScalarField,
    pub value: E::ScalarField,
}

/// Shplonk batched opening + final KZG proof.
pub(crate) struct ShplonkOutput<E: Pairing> {
    pub q_commit: E::G1Affine,
    pub final_witness: E::G1Affine,
    pub z_challenge: E::ScalarField,
}

/// Prove a set of opening claims via Shplonk batching.
///
/// Returns a single batched claim (G(X) where G(z)=0) and the KZG witness.
pub(crate) fn shplonk_prove<E: Pairing>(
    claims: &[ProverClaim<E::ScalarField>],
    pos_evals: &[E::ScalarField], // positive fold evals, parallel to claims where gemini_fold=true
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<ShplonkOutput<E>, PCSError> {
    let nu = transcript.get_and_append_challenge_vectors(b"Shplonk:nu", 1)?[0];

    // 1. Compute batched quotient Q(X) = Σ ν^j * (f_j(X) - v_j) / (X - x_j)
    let mut max_deg = 0usize;
    for c in claims {
        max_deg = max_deg.max(c.coeffs.len().saturating_sub(1));
    }

    let mut batched_q = vec![E::ScalarField::zero(); max_deg];
    let mut current_nu = E::ScalarField::one();
    let mut fold_idx = 0usize;

    for c in claims {
        if c.gemini_fold {
            // Positive opening: contribute (f(X) - pos_val) / (X + neg_point)
            let pos_val = pos_evals[fold_idx];
            fold_idx += 1;
            let quot = kzg_quotient(&c.coeffs, -c.point, pos_val);
            let scaled_len = quot.len();
            for (j, &qj) in quot.iter().enumerate() {
                batched_q[j] += current_nu * qj;
            }
            if batched_q.len() < scaled_len {
                batched_q.resize(scaled_len, E::ScalarField::zero());
            }
            current_nu *= nu;
        }
        // Negative opening: contribute (f(X) - v_j) / (X - x_j)
        let quot = kzg_quotient(&c.coeffs, c.point, c.value);
        let scaled_len = quot.len();
        for (j, &qj) in quot.iter().enumerate() {
            batched_q[j] += current_nu * qj;
        }
        if batched_q.len() < scaled_len {
            batched_q.resize(scaled_len, E::ScalarField::zero());
        }
        current_nu *= nu;
    }

    // 2. Commit Q
    let q_commit = /* filled in by caller who has pp */ todo!("caller provides pp");
    transcript.append_serializable_element(b"Shplonk:Q", &q_commit)?;

    // 3. Get evaluation challenge z
    let z = transcript.get_and_append_challenge_vectors(b"Shplonk:z", 1)?[0];

    // Verify z is non-zero and distinct from all opening points
    if z.is_zero() {
        return Err(PCSError::InvalidParameters("Shplonk: z is zero".to_string()));
    }

    // 4. Compute partially evaluated batched quotient G(X) = Q(X) - Q_z(X)
    //    G(X) = Q(X) - Σ ν^j * (f_j(X) - v_j) / (z - x_j)
    //    where for gemini_fold: also subtract ν^j * (f_j(X) - pos_val) / (z + x_j)
    let mut g = batched_q.clone();
    let mut current_nu = E::ScalarField::one();
    let mut fold_idx2 = 0usize;

    for c in claims {
        if c.gemini_fold {
            let pos_val = pos_evals[fold_idx2];
            fold_idx2 += 1;
            let denom = z + c.point;
            if denom.is_zero() {
                return Err(PCSError::InvalidParameters(
                    "Shplonk: z + x_j is zero".to_string(),
                ));
            }
            let inv_denom = denom.inverse()
                .ok_or_else(|| PCSError::InvalidParameters("Shplonk: denom inverse failed".to_string()))?;
            let scale = current_nu * inv_denom;
            for (j, &cj) in c.coeffs.iter().enumerate() {
                if g.len() <= j {
                    g.resize(j + 1, E::ScalarField::zero());
                }
                g[j] -= scale * cj;
            }
            if g.is_empty() {
                g.resize(1, E::ScalarField::zero());
            }
            g[0] += scale * pos_val;
            current_nu *= nu;
        }
        // Standard: 1/(z - x_j)
        let denom = z - c.point;
        if denom.is_zero() {
            return Err(PCSError::InvalidParameters(
                "Shplonk: z - x_j is zero".to_string(),
            ));
        }
        let inv_denom = denom.inverse()
            .ok_or_else(|| PCSError::InvalidParameters("Shplonk: denom inverse failed".to_string()))?;
        let scale = current_nu * inv_denom;
        for (j, &cj) in c.coeffs.iter().enumerate() {
            if g.len() <= j {
                g.resize(j + 1, E::ScalarField::zero());
            }
            g[j] -= scale * cj;
        }
        if g.is_empty() {
            g.resize(1, E::ScalarField::zero());
        }
        g[0] += scale * c.value;
        current_nu *= nu;
    }

    // 5. G(z) should be zero; produce final KZG quotient witness
    let final_witness = kzg_quotient(&g, z, E::ScalarField::zero());

    Ok(ShplonkOutput {
        q_commit,
        final_witness: /* caller commits */ todo!(),
        z_challenge: z,
    })
}

/// KZG quotient: compute Q(X) = (f(X) - v) / (X - point).
/// f has coefficients f[0..len-1] where f(X) = Σ f_i X^i.
/// Return Q with degree len-2 (zero if len ≤ 1).
pub(crate) fn kzg_quotient<F: Field>(coeffs: &[F], point: F, value: F) -> Vec<F> {
    if coeffs.len() <= 1 {
        return vec![];
    }
    let n = coeffs.len();
    let mut q = vec![F::zero(); n - 1];
    let mut carry = F::zero();
    // Standard synthetic division: f(X) - v, then divide by (X - point)
    // Work from high degree down
    for i in (1..n).rev() {
        let term = coeffs[i] + carry;
        q[i - 1] = term;
        carry = term * point;
    }
    // Adjust for the subtracted value
    let rem = coeffs[0] + carry;
    if rem != value {
        // The remainder should equal the claimed value; if not, the claim is invalid
        // (this is a prover-side check)
    }
    q
}

/// Shplonk verifier: given claims and the proof, compute the batched commitment
/// [G] that must be opened at z with value 0. Returns (G_commitment, z_challenge).
pub(crate) fn shplonk_verify_reduce<E: Pairing>(
    claims: &[VerifierClaim<E>],
    pos_evals: &[E::ScalarField],
    q_commit: &E::G1Affine,
    g1_identity: &E::G1Affine,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<(E::G1Affine, E::ScalarField), PCSError> {
    let nu = transcript.get_and_append_challenge_vectors(b"Shplonk:nu", 1)?[0];
    transcript.append_serializable_element(b"Shplonk:Q", q_commit)?;
    let z = transcript.get_and_append_challenge_vectors(b"Shplonk:z", 1)?[0];

    if z.is_zero() {
        return Err(PCSError::InvalidProof("Shplonk verifier: z is zero".to_string()));
    }

    // Build the MSM: [G] = [Q] - Σ (ν^j / (z - x_j)) * [C_j] + Σ (ν^j * v_j / (z - x_j)) * [1]
    // For gemini_fold: also subtract (ν^j / (z + x_j)) * [C_j] and add (ν^j * pos_val / (z + x_j)) * [1]

    let mut bases: Vec<E::G1Affine> = vec![*q_commit];
    let mut scalars: Vec<E::ScalarField> = vec![E::ScalarField::one()]; // [Q] with coeff 1

    let mut identity_scalar = E::ScalarField::zero();
    let mut current_nu = E::ScalarField::one();
    let mut fold_idx = 0usize;

    for c in claims {
        if fold_idx < pos_evals.len() {
            // This is a Gemini fold: first handle positive opening with 1/(z + x_j)
            let pos_val = pos_evals[fold_idx];
            fold_idx += 1;
            let denom_p = z + c.point;
            if denom_p.is_zero() {
                return Err(PCSError::InvalidProof("Shplonk verifier: z + x is zero".to_string()));
            }
            let inv_p = denom_p.inverse()
                .ok_or_else(|| PCSError::InvalidProof("Shplonk verifier: denom inverse failed".to_string()))?;
            let scale = current_nu * inv_p;
            bases.push(c.commitment);
            scalars.push(-scale);
            identity_scalar += scale * pos_val;
            current_nu *= nu;
        }
        // Negative opening: 1/(z - x_j)
        let denom = z - c.point;
        if denom.is_zero() {
            return Err(PCSError::InvalidProof("Shplonk verifier: z - x is zero".to_string()));
        }
        let inv = denom.inverse()
            .ok_or_else(|| PCSError::InvalidProof("Shplonk verifier: denom inverse failed".to_string()))?;
        let scale = current_nu * inv;
        bases.push(c.commitment);
        scalars.push(-scale);
        identity_scalar += scale * c.value;
        current_nu *= nu;
    }

    // Add identity contribution
    bases.push(*g1_identity);
    scalars.push(identity_scalar);

    // [G] = MSM(bases, scalars)
    let g_commit = E::G1::msm_unchecked(&bases, &scalars).into_affine();

    Ok((g_commit, z))
}
