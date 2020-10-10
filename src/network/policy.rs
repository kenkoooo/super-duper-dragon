use crate::constants::{INPUT_CHANNELS, MOVE_DIRECTION_LABEL_NUM};
use tch::nn::{Conv2D, ConvConfig, Module, Path};
use tch::Tensor;

#[derive(Debug)]
struct Bias {
    bias: Tensor,
}

impl Bias {
    fn new(vs: &Path, shape: i64) -> Self {
        Self {
            bias: vs.zeros("bias", &[shape]),
        }
    }
}

impl Module for Bias {
    fn forward(&self, input: &Tensor) -> Tensor {
        input + &self.bias
    }
}

#[derive(Debug)]
pub struct PolicyNetwork {
    l1: Conv2D,
    l2: Conv2D,
    l3: Conv2D,
    l4: Conv2D,
    l5: Conv2D,
    l6: Conv2D,
    l7: Conv2D,
    l8: Conv2D,
    l9: Conv2D,
    l10: Conv2D,
    l11: Conv2D,
    l12: Conv2D,
    l13: Conv2D,
    l13_bias: Bias,
}

const CH: i64 = 192;
impl PolicyNetwork {
    pub fn new(vs: &Path) -> Self {
        let conv_config = ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let l1 = tch::nn::conv2d(vs, INPUT_CHANNELS as i64, CH, 3, conv_config);
        let l2 = tch::nn::conv2d(vs, CH, CH, 3, conv_config);
        let l3 = tch::nn::conv2d(vs, CH, CH, 3, conv_config);
        let l4 = tch::nn::conv2d(vs, CH, CH, 3, conv_config);
        let l5 = tch::nn::conv2d(vs, CH, CH, 3, conv_config);
        let l6 = tch::nn::conv2d(vs, CH, CH, 3, conv_config);
        let l7 = tch::nn::conv2d(vs, CH, CH, 3, conv_config);
        let l8 = tch::nn::conv2d(vs, CH, CH, 3, conv_config);
        let l9 = tch::nn::conv2d(vs, CH, CH, 3, conv_config);
        let l10 = tch::nn::conv2d(vs, CH, CH, 3, conv_config);
        let l11 = tch::nn::conv2d(vs, CH, CH, 3, conv_config);
        let l12 = tch::nn::conv2d(vs, CH, CH, 3, conv_config);

        let conv_config = ConvConfig {
            bias: false,
            ..Default::default()
        };
        let l13 = tch::nn::conv2d(vs, CH, MOVE_DIRECTION_LABEL_NUM, 1, conv_config);
        let l13_bias = Bias::new(vs, 9 * 9 * MOVE_DIRECTION_LABEL_NUM);
        Self {
            l1,
            l2,
            l3,
            l4,
            l5,
            l6,
            l7,
            l8,
            l9,
            l10,
            l11,
            l12,
            l13,
            l13_bias,
        }
    }
}

impl Module for PolicyNetwork {
    fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = x.apply(&self.l1).relu();
        let h2 = h1.apply(&self.l2).relu();
        let h3 = h2.apply(&self.l3).relu();
        let h4 = h3.apply(&self.l4).relu();
        let h5 = h4.apply(&self.l5).relu();
        let h6 = h5.apply(&self.l6).relu();
        let h7 = h6.apply(&self.l7).relu();
        let h8 = h7.apply(&self.l8).relu();
        let h9 = h8.apply(&self.l9).relu();
        let h10 = h9.apply(&self.l10).relu();
        let h11 = h10.apply(&self.l11).relu();
        let h12 = h11.apply(&self.l12).relu();
        let h13 = h12.apply(&self.l13);
        let batchsize = h13.size()[0];
        h13.reshape(&[batchsize, 9 * 9 * MOVE_DIRECTION_LABEL_NUM])
            .apply(&self.l13_bias)
    }
}
