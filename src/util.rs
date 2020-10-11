pub mod make_input_feature;
pub mod make_output_label;

use crate::constants::INPUT_CHANNELS;
use anyhow::Result;
use shogiutil::{Board, Piece, Square};
use tch::kind::Kind::Double;
use tch::nn::VarStore;
use tch::Tensor;

pub trait Accuracy {
    fn accuracy(&self, target: &Tensor) -> f64;
}

impl Accuracy for Tensor {
    fn accuracy(&self, target: &Tensor) -> f64 {
        let pred = self.argmax(Some(1), true);
        pred.eq1(&target.view_as(&pred))
            .mean(Double)
            .double_value(&[])
    }
}

pub trait CheckPoint {
    fn load_if_exists(&mut self, filepath: &str) -> Result<()>;
}

impl CheckPoint for VarStore {
    fn load_if_exists(&mut self, filepath: &str) -> Result<()> {
        if std::path::Path::new(filepath).exists() {
            log::info!("Loading {}", filepath);
            self.load(filepath)?;
        }
        Ok(())
    }
}
