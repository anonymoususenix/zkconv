pub mod commit;
pub mod conv;
pub mod conv_padding;
pub mod maxpool;
pub mod relu;

use ark_bn254::{Bn254, Fr as ScalarField, FrConfig};
use ark_ec::pairing::Pairing;

pub type E = Bn254;
pub type F = <E as Pairing>::ScalarField;
