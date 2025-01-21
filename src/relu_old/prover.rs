//! To prove a ReLU layer:
//! Input: a vector of scalars `y1`.
//! Step 1: Compute `y2 = y1 >> 2^Q`.
//! Step 2: Compute `y3 = relu(y2)`.
//! Output: `y3`.
//!
//! Proof for Step 1: `y1 = 2^Q * y2 + remainder`.
//! - The prover calculates `remainder = y1 - 2^Q * y2` and commits to it.
//! - Then the prover and verifier:
//!   1. Conduct a sumcheck protocol to prove `sum_i(y1_i - 2^Q * y2_i - r_i) = 0`.
//!   2. Conduct a logup protocol to prove `remainder ∈ [0, 2^Q-1]`.
//!
//! Proof for Step 2: `y3 = relu(y2)`.
//! - The prover and verifier:
//!   1. Conduct a logup protocol to prove `relu(y2) = y3`.

use crate::{E, F};
use ark_bn254::FrConfig;
use ark_ec::pairing::Pairing;
use ark_ff::Fp;
use ark_ff::MontBackend;
use ark_ff::{Field, PrimeField}; // for into_bigint
use ark_poly::DenseMultilinearExtension;
use ark_std::rand::Rng;
use ark_std::rc::Rc;
use ark_std::test_rng;
use ark_std::vec::Vec;
use logup::{Logup, LogupProof};
use merlin::Transcript;
use pcs::multilinear_kzg::data_structures::{MultilinearProverParam, MultilinearVerifierParam};

// Sumcheck usage as per snippet
use ark_sumcheck::ml_sumcheck::{
    data_structures::{ListOfProductsOfPolynomials, PolynomialInfo},
    MLSumcheck, Proof as SumcheckProof,
};

pub struct Prover {
    pub Q: u32,
    pub y1: Vec<F>,
    pub y2: Vec<F>,
    pub y3: Vec<F>,
    pub remainder: Vec<F>,
}

impl Prover {
    pub fn new(Q: u32, y1: Vec<F>) -> Self {
        let (y2, remainder) = Self::compute_y2_and_remainder(&y1, Q);
        let y3 = Self::compute_relu(&y2);
        Prover {
            Q,
            y1,
            y2,
            y3,
            remainder,
        }
    }

    pub fn new_real_data(Q: u32, y1: Vec<F>, y2: Vec<F>, y3: Vec<F>, remainder: Vec<F>) -> Self {
        Prover {
            Q,
            y1,
            y2,
            y3,
            remainder,
        }
    }

    fn compute_y2_and_remainder(y1: &Vec<F>, Q: u32) -> (Vec<F>, Vec<F>) {
        let mut y2 = Vec::new();
        let mut remainder = Vec::new();
        for &val in y1.iter() {
            let int_val = Self::field_to_int(val);
            let y2_i_int = int_val >> Q;
            let y2_i = Self::int_to_field(y2_i_int);
            let computed_remainder_int = int_val - (y2_i_int << Q);
            let remainder_i = Self::int_to_field(computed_remainder_int);
            y2.push(y2_i);
            remainder.push(remainder_i);
        }
        (y2, remainder)
    }

    fn compute_relu(y2: &Vec<F>) -> Vec<F> {
        y2.iter()
            .map(|&val| {
                let int_val = Self::field_to_int(val);
                if int_val >= 0 {
                    val
                } else {
                    F::from(0u64)
                }
            })
            .collect()
    }

    fn field_to_int(val: F) -> i64 {
        val.into_bigint().as_ref()[0] as i64
    }

    fn int_to_field(x: i64) -> F {
        F::from(x as u64)
    }

    fn next_power_of_two(n: usize) -> usize {
        let mut p = 1;
        while p < n {
            p <<= 1;
        }
        p
    }

    fn vector_to_mle(evals: &Vec<F>) -> Rc<DenseMultilinearExtension<F>> {
        let length = evals.len();
        let size = Self::next_power_of_two(length);
        let mut padded = evals.clone();
        padded.resize(size, F::from(0u64));
        let nv = (size as f64).log2() as usize;
        Rc::new(DenseMultilinearExtension::from_evaluations_vec(nv, padded))
    }

    // Prove step1 sumcheck: sum_i(y1_i - 2^Q * y2_i - remainder_i) = 0
    // Return (proof, asserted_sum, poly_info)
    pub fn prove_step1_sumcheck<R: Rng>(
        &self,
        rng: &mut R,
    ) -> (SumcheckProof<F>, F, PolynomialInfo) {
        let two_pow_q = F::from(2u64).pow([self.Q as u64]);
        let evals: Vec<F> = self
            .y1
            .iter()
            .zip(self.y2.iter())
            .zip(self.remainder.iter())
            .map(|((&y1_i, &y2_i), &r_i)| y1_i - two_pow_q * y2_i - r_i)
            .collect();

        let mut poly = ListOfProductsOfPolynomials::new((evals.len() as f64).log2() as usize);
        let mle = Self::vector_to_mle(&evals);
        poly.add_product(vec![mle], F::from(1u64));

        let proof = MLSumcheck::prove(&poly).expect("fail to prove sumcheck");
        let asserted_sum = MLSumcheck::extract_sum(&proof);
        let poly_info = poly.info();
        (proof, asserted_sum, poly_info)
    }

    // // Prove step1_logup: remainder in [0, 2^Q-1]
    // // a = remainder
    // // t = merged sorted vector of [0..2^Q-1] and a
    // // Return (commit, proof, a, t)
    // pub fn prove_step1_logup(
    //     &self,
    // ) -> (Vec<<E as Pairing>::G1Affine>, LogupProof<E>, Vec<F>, Vec<F>) {
    //     let a = self.remainder.clone();
    //     let mut base: Vec<F> = (0..(1 << self.Q)).map(|x| F::from(x as u64)).collect();
    //     let mut t = base;

    //     let mut transcript = Transcript::new(b"Logup");
    //     let ((pk, ck), commit) = Logup::process::<E>(20, &a);
    //     let proof = Logup::prove::<E>(&a, &t, &pk, &mut transcript);

    //     (commit, proof, a, t)
    // }

    // // Prove step2_logup: y3 = relu(y2)
    // // a = y2 + alpha*y3
    // // t = a (todo: confirm t range)
    // // Return (commit, proof, a, t)
    // pub fn prove_step2_logup(
    //     &self,
    // ) -> (Vec<<E as Pairing>::G1Affine>, LogupProof<E>, Vec<F>, Vec<F>) {
    //     let mut rng = test_rng();
    //     let alpha = F::rand(&mut rng);

    //     let a: Vec<F> = self
    //         .y2
    //         .iter()
    //         .zip(self.y3.iter())
    //         .map(|(&y2_i, &y3_i)| y2_i + alpha * y3_i)
    //         .collect();
    //     let t = a.clone(); // todo: confirm t range

    //     let mut transcript = Transcript::new(b"Logup");
    //     let ((pk, ck), commit) = Logup::process::<E>(20, &a);
    //     let proof = Logup::prove::<E>(&a, &t, &pk, &mut transcript);

    //     (commit, proof, a, t)
    // }
    // Process function extracted for reuse
    pub fn process_step1_logup<E: Pairing<ScalarField = Fp<MontBackend<FrConfig, 4>, 4>>>(
        &self,
        a: &Vec<F>,
        params: usize,
    ) -> (
        Vec<<E as Pairing>::G1Affine>,
        MultilinearProverParam<E>,
        MultilinearVerifierParam<E>,
        Vec<F>,
    ) {
        let mut base: Vec<F> = (0..(1 << params)).map(|x| F::from(x as u64)).collect();
        let t = base.clone();

        let mut transcript = Transcript::new(b"Logup");
        let ((pk, ck), commit) = Logup::process::<E>(18, a);

        (commit, pk, ck, t)
    }

    // Prove step1_logup: remainder in [0, 2^Q-1]
    // a = remainder
    // t = merged sorted vector of [0..2^Q-1] and a
    // Return (commit, proof, a, t)
    pub fn prove_step1_logup(
        &self,
        commit: Vec<<E as Pairing>::G1Affine>,
        pk: MultilinearProverParam<E>,
        t: Vec<F>,
    ) -> (Vec<<E as Pairing>::G1Affine>, LogupProof<E>, Vec<F>, Vec<F>) {
        let a = self.remainder.clone();
        let mut transcript = Transcript::new(b"Logup");

        // Prove
        let proof = Logup::prove::<E>(&a, &t, &pk, &mut transcript);

        (commit, proof, a, t)
    }

    // Prove step2_logup: y3 = relu(y2)
    // a = y2 + alpha*y3
    // Return (commit, proof, a, t)
    // pub fn prove_step2_logup(
    //     &self,
    //     commit: Vec<<E as Pairing>::G1Affine>,
    //     pk: MultilinearProverParam<E>,
    //     t: Vec<F>,
    // ) -> (Vec<<E as Pairing>::G1Affine>, LogupProof<E>, Vec<F>, Vec<F>) {
    //     let mut rng = test_rng();
    //     let alpha = F::rand(&mut rng);

    //     let a: Vec<F> = self
    //         .y2
    //         .iter()
    //         .zip(self.y3.iter())
    //         .map(|(&y2_i, &y3_i)| y2_i + alpha * y3_i)
    //         .collect();
    //     // t = concate(y2, (1+alpha)*y2)
    //     let mut t = self.y2.clone();
    //     t.extend(self.y2.iter().map(|&y2_i| y2_i + y2_i * alpha));
    //     // sort t
    //     t.sort();

    //     let mut transcript = Transcript::new(b"Logup");

    //     // Prove
    //     let proof = Logup::prove::<E>(&a, &t, &pk, &mut transcript);

    //     (commit, proof, a, t)
    // }
    pub fn compute_a_t<F: Field>(&self, y2: &[F], y3: &[F]) -> (Vec<F>, Vec<F>) {
        let mut rng = test_rng();
        let alpha = F::rand(&mut rng);

        // compute a
        // a = y2 + alpha * y3
        // when y2 > max_possible_value, y3 = 0, a = y2
        // when y2 <= max_possible_value, y3 = y2, a = y2 + alpha * y2
        let mut a: Vec<F> = y2
            .iter()
            .zip(y3.iter())
            .map(|(&y2_i, &y3_i)| y2_i + alpha * y3_i)
            .collect();

        // compute t
        // t = concat(y2, (1 + alpha) * y2)
        let mut t = y2.to_vec();
        t.extend(y2.iter().map(|&y2_i| y2_i + y2_i * alpha));

        a.sort();

        t.sort();
        t.dedup();

        // Ensure t contains a power-of-2 number of unique values
        let unique_count = t.len();
        let next_power_of_2 = unique_count.next_power_of_two();

        while t.len() < next_power_of_2 {
            // Generate random values and add to t
            let random_value = F::rand(&mut rng);
            if !t.contains(&random_value) {
                t.push(random_value);
            }
        }

        t.sort();
        t.dedup(); // Re-deduplicate in case random values introduced duplicates

        (a, t)
    }

    pub fn process_step2_logup<E: Pairing<ScalarField = Fp<MontBackend<FrConfig, 4>, 4>>>(
        &self,
        a: &Vec<F>,
    ) -> (
        Vec<<E as Pairing>::G1Affine>,
        MultilinearProverParam<E>,
        MultilinearVerifierParam<E>,
    ) {
        let ((pk, ck), commit) = Logup::process::<E>(18, &a);

        (commit, pk, ck)
    }

    pub fn prove_step2_logup(
        &self,
        commit: Vec<<E as Pairing>::G1Affine>,
        pk: MultilinearProverParam<E>,
        t: Vec<F>,
        a: Vec<F>,
        transcript: &mut Transcript,
    ) -> (Vec<<E as Pairing>::G1Affine>, LogupProof<E>, Vec<F>, Vec<F>) {
        // let mut transcript = Transcript::new(b"Logup");

        // call prove
        let proof = Logup::prove::<E>(&a, &t, &pk, transcript);

        (commit, proof, a, t)
    }
}
