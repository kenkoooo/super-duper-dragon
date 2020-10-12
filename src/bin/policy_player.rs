use anyhow::Result;
use clap::Clap;
use shogiutil::{UsiRequest, UsiResponse};
use super_duper_dragon::constants::INPUT_CHANNELS;
use super_duper_dragon::network::policy::PolicyNetwork;
use super_duper_dragon::usi::UsiPlayer;
use super_duper_dragon::util::board_packer::flatten;
use super_duper_dragon::util::make_output_label::make_output_label;
use super_duper_dragon::util::CheckPoint;
use tch::nn::{Module, VarStore};
use tch::Device;
use tch::Tensor;

#[derive(Clap)]
struct Opts {
    #[clap(short, long)]
    model_filepath: String,
}
struct PolicyPlayer {
    model: PolicyNetwork,
    vs: VarStore,
}

impl PolicyPlayer {
    fn init(&mut self) {}
}

impl UsiPlayer for PolicyPlayer {
    fn play(&mut self, request: UsiRequest) -> Vec<UsiResponse> {
        use UsiResponse::*;
        match request {
            UsiRequest::Usi => vec![
                Id {
                    name: "policy_player".to_string(),
                },
                UsiOk,
            ],
            UsiRequest::IsReady => {
                self.init();
                vec![ReadyOk]
            }
            UsiRequest::SetOption { .. } => vec![],
            UsiRequest::NewGame => vec![],
            UsiRequest::Position { board, next_turn } => {
                let features = make_input_feature_from_board(&board, next_turn);
                let input = flatten(&features);
                let x = Tensor::of_slice(&input)
                    .view((1, INPUT_CHANNELS as i64, 9, 9))
                    .to_device(self.vs.device());
                let output = self.model.forward(&x);

                for mv in board.generate_legal_moves() {
                    // make_output_label()
                }
                todo!()
            }
            UsiRequest::Go => vec![],
            UsiRequest::Quit => vec![],
        }
    }
}

fn main() -> Result<()> {
    let opts: Opts = Opts::parse();
    let mut vs = VarStore::new(Device::Cuda(0));
    let model = PolicyNetwork::new(&vs.root());
    vs.load_if_exists(&opts.model_filepath)?;
    let mut player = PolicyPlayer { model, vs };
    player.usi_play();
    Ok(())
}
