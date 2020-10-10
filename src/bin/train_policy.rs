use anyhow::Result;
use rand::prelude::*;
use std::env;
use std::fs::File;
use std::io::Read;
use super_duper_dragon::constants::INPUT_CHANNELS;
use super_duper_dragon::data_loader::DataLoader;
use super_duper_dragon::model::Position;
use super_duper_dragon::network::policy::PolicyNetwork;
use super_duper_dragon::util::Accuracy;
use tch::kind::Kind::{Double, Int64};
use tch::nn::{Module, OptimizerConfig, Sgd, VarStore};
use tch::{no_grad, Device};

fn load_bin_file(filepath: &str) -> Result<Vec<Position>> {
    log::info!("Loading {}", filepath);
    let mut f = File::open(filepath)?;
    let mut buf = vec![];
    f.read_to_end(&mut buf)?;
    let kifu: Vec<Position> = bincode::deserialize(&buf)?;
    Ok(kifu)
}

fn main() -> Result<()> {
    env::set_var("RUST_LOG", "info");
    env_logger::init();

    let mut rng = StdRng::seed_from_u64(717);
    let batchsize = 32;
    let eval_interval = 100.0;

    let train_kifu = load_bin_file("./train_kifu_list.bin")?;
    log::info!("train_data = {}", train_kifu.len());

    let mut test_kifu = load_bin_file("./test_kifu_list.bin")?;
    log::info!("test_data = {}", test_kifu.len());

    let vs = VarStore::new(Device::Cuda(0));
    let model = PolicyNetwork::new(&vs.root());
    let mut optimizer = Sgd::default().build(&vs, 0.01)?;
    for _ in 0..1 {
        let mut sum_loss = 0.0;
        let mut iter = 0.0;
        let mut sum_loss_epoch = 0.0;
        let mut iter_epoch = 0.0;

        let train_loader = DataLoader::new(&train_kifu, position_to_features, batchsize);
        for (x, t) in train_loader {
            let x = x
                .view((batchsize as i64, INPUT_CHANNELS as i64, 9, 9))
                .to_device(vs.device());
            let t = t.totype(Int64).to_device(vs.device());
            optimizer.zero_grad();
            let y = model.forward(&x);
            let loss = y.log_softmax(-1, Double).nll_loss(&t);
            optimizer.backward_step(&loss);

            sum_loss += loss.double_value(&[]);
            iter += 1.0;
            sum_loss_epoch += loss.double_value(&[]);
            iter_epoch += 1.0;

            if iter == eval_interval {
                no_grad(|| {
                    let test_batchsize = 512;
                    test_kifu.shuffle(&mut rng);
                    let mut test_loader =
                        DataLoader::new(&test_kifu, position_to_features, test_batchsize);
                    let (x, t) = test_loader.next().unwrap();
                    let x = x
                        .view((test_batchsize as i64, INPUT_CHANNELS as i64, 9, 9))
                        .to_device(vs.device());
                    let t = t.to_device(vs.device());
                    let y = model.forward(&x);
                    log::info!(
                        "iter_epoch={} loss={} accuracy={}",
                        iter_epoch,
                        sum_loss / iter,
                        y.accuracy(&t)
                    );
                });
                sum_loss = 0.0;
                iter = 0.0;
            }
        }
    }

    Ok(())
}

fn position_to_features(position: &Position) -> (Vec<f32>, u8) {
    let mut features = vec![0.0f32; 9 * 9 * INPUT_CHANNELS];
    for (c, &feature) in position.features.iter().enumerate() {
        assert!(c < INPUT_CHANNELS);
        for i in 0..9 {
            for j in 0..9 {
                let pos = i * 9 + j;
                if feature & (1 << pos) != 0 {
                    features[9 * 9 * c + pos] = 1.0;
                }
            }
        }
    }
    (features, position.move_label)
}
