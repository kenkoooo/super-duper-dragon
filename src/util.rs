use tch::kind::Kind::Double;
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
