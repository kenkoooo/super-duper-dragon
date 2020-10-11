use anyhow::Result;
use clap::Clap;
use rand::prelude::*;
use std::env;
use std::fs::File;
use std::io::Read;
use super_duper_dragon::constants::INPUT_CHANNELS;
use super_duper_dragon::data_loader::DataLoader;
use super_duper_dragon::model::Position;
use super_duper_dragon::network::policy::PolicyNetwork;
use super_duper_dragon::progressbar::ProgressBar;
use super_duper_dragon::util::{Accuracy, CheckPoint};
use tch::kind::Kind::{Double, Int64};
use tch::nn::{Module, OptimizerConfig, Sgd, VarStore};
use tch::{no_grad, Device};

#[derive(Clap)]
#[clap(version = "1.0", author = "kenkoooo <kenkou.n@gmail.com>")]
struct Opts {
    #[clap(short, long, default_value = "1024")]
    batchsize: usize,
    #[clap(short, long, default_value = "100")]
    eval_interval: usize,
    #[clap(short, long)]
    save_file_path: String,
    #[clap(long)]
    train: String,
    #[clap(long)]
    test: String,
}

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
    let opts: Opts = Opts::parse();

    let mut rng = StdRng::seed_from_u64(717);
    let batchsize = opts.batchsize;

    let train_kifu = load_bin_file(&opts.train)?;
    log::info!("train_data = {}", train_kifu.len());

    let mut test_kifu = load_bin_file(&opts.test)?;
    log::info!("test_data = {}", test_kifu.len());

    let mut vs = VarStore::new(Device::Cuda(0));
    let model = PolicyNetwork::new(&vs.root());
    vs.load_if_exists(&opts.save_file_path)?;

    let mut optimizer = Sgd::default().build(&vs, 0.01)?;
    for _ in 0..1 {
        let mut sum_loss = 0.0;
        let mut iter = 0.0;
        let mut sum_loss_epoch = 0.0;
        let mut iter_epoch = 0.0;

        let train_loader = DataLoader::new(&train_kifu, position_to_features, batchsize);
        for (x, t) in ProgressBar::new(train_loader, |state| log::info!("{}", state)) {
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

            if iter as usize == opts.eval_interval {
                vs.save(&opts.save_file_path)?;
                no_grad(|| {
                    test_kifu.shuffle(&mut rng);
                    let mut test_loader =
                        DataLoader::new(&test_kifu, position_to_features, batchsize);
                    let (x, t) = test_loader.next().unwrap();
                    let x = x
                        .view((batchsize as i64, INPUT_CHANNELS as i64, 9, 9))
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

    log::info!("Done");
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
