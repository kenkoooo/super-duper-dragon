use anyhow::Result;
use tch::kind::Kind::Double;
use tch::nn::{self, Linear, Module, OptimizerConfig, Path};
use tch::{no_grad, Device, Tensor};

#[derive(Debug)]
struct MLP {
    l1: Linear,
    l2: Linear,
    l3: Linear,
}

impl MLP {
    fn new(vs: &Path, n_unit: i64) -> Self {
        let l1 = tch::nn::linear(vs, 28 * 28 * 1, n_unit, Default::default());
        let l2 = tch::nn::linear(vs, n_unit, n_unit, Default::default());
        let l3 = tch::nn::linear(vs, n_unit, 10, Default::default());
        MLP { l1, l2, l3 }
    }
}

impl tch::nn::Module for MLP {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let h1 = xs.apply(&self.l1).relu();
        let h2 = h1.apply(&self.l2).relu();
        h2.apply(&self.l3)
    }
}

fn main() -> Result<()> {
    let m = tch::vision::mnist::load_dir("./mnist/MNIST/raw")?;
    let vs = tch::nn::VarStore::new(Device::Cuda(0));
    let batchsize = 100;

    let model = MLP::new(&vs.root(), 1000);
    let mut optimizer = nn::Sgd::default().build(&vs, 0.01)?;
    for epoch in 0..20 {
        let mut sum_loss = 0.0;
        let mut itr = 0.0;
        for (data, target) in m.train_iter(batchsize).to_device(vs.device()) {
            optimizer.zero_grad();
            let y = model.forward(&data);
            let loss = y.log_softmax(-1, Double).nll_loss(&target);
            optimizer.backward_step(&loss);

            sum_loss += loss.double_value(&[]);
            itr += 1.0;
        }

        let mut sum_test_loss = 0.0;
        let mut sum_test_accuracy = 0.0;
        let mut test_itr = 0.0;
        no_grad(|| {
            for (data, target) in m.test_iter(batchsize).to_device(vs.device()) {
                let y_test = model.forward(&data);
                sum_test_loss += y_test
                    .log_softmax(-1, Double)
                    .nll_loss(&target)
                    .double_value(&[]);
                let pred = y_test.argmax(Some(1), true);
                sum_test_accuracy += pred
                    .eq1(&target.view_as(&pred))
                    .sum(Double)
                    .double_value(&[]);
                test_itr += 1.0;
            }
        });

        let train_loss = sum_loss / itr;
        let test_loss = sum_test_loss / test_itr;
        let accuracy = sum_test_accuracy / test_itr;
        println!(
            "epoch={} train_loss={} test_loss={} accuracy={}",
            epoch, train_loss, test_loss, accuracy
        );
    }

    Ok(())
}
