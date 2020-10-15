use anyhow::Result;
use clap::Clap;
use shogiutil::{Board, Color, Move, UsiRequest, UsiResponse};
use std::env;
use super_duper_dragon::constants::INPUT_CHANNELS;
use super_duper_dragon::network::policy::PolicyNetwork;
use super_duper_dragon::usi::UsiPlayer;
use super_duper_dragon::util::board_packer::{BoardPacker, ToFlatVec};
use super_duper_dragon::util::make_output_label::make_output_label;
use super_duper_dragon::util::CheckPoint;
use tch::nn::{Module, VarStore};
use tch::Device;
use tch::Kind::Double;
use tch::Tensor;

#[derive(Clap)]
struct Opts {
    #[clap(short, long)]
    model_filepath: String,
}
struct PolicyPlayer {
    model: PolicyNetwork,
    vs: VarStore,
    board: Option<Board>,
    next_turn: Option<Color>,
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
                self.board = Some(board);
                self.next_turn = Some(next_turn);
                vec![]
            }
            UsiRequest::Go => {
                let board = self.board.take().unwrap();
                let next_turn = self.next_turn.take().unwrap();

                let board = if next_turn == Color::Black {
                    board
                } else {
                    board.rotate180()
                };
                let features = board.encode();
                let x = Tensor::of_slice(&features.to_flat_vec())
                    .view((1, INPUT_CHANNELS as i64, 9, 9))
                    .to_device(self.vs.device());
                let y = self.model.forward(&x);
                let probability = y.softmax(-1, Double);

                let mut moves = vec![];
                for mv in board.generate_legal_moves() {
                    let (mv, promoted) = (mv.mv, mv.promoted);
                    let label = make_output_label(&mv.from, &mv.to, mv.piece, promoted);
                    let logit = y.double_value(&[0, label as i64]);
                    let probability = probability.double_value(&[0, label as i64]);

                    let legal_move = if next_turn == Color::Black {
                        mv
                    } else {
                        Move {
                            from: mv.from.map(|f| f.rotate()),
                            to: mv.to.rotate(),
                            piece: mv.piece,
                            color: next_turn,
                        }
                    };
                    log::info!("{:?} {:.5}", legal_move, probability);
                    moves.push((legal_move, logit, probability, promoted));
                }

                moves.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
                let (best_move, _, _, promoted) = moves.pop().unwrap();
                if let Some(from) = best_move.from {
                    vec![TravelMove {
                        from,
                        to: best_move.to,
                        promoted,
                    }]
                } else {
                    vec![UsiResponse::DropMove {
                        to: best_move.to,
                        piece: best_move.piece,
                    }]
                }
            }
            UsiRequest::Quit => vec![],
        }
    }
}

fn main() -> Result<()> {
    env::set_var("RUST_LOG", "info");
    env_logger::init();

    let opts: Opts = Opts::parse();
    log::info!("Initializing model ...");
    let mut vs = VarStore::new(Device::Cuda(0));
    let model = PolicyNetwork::new(&vs.root());
    vs.load_if_exists(&opts.model_filepath)?;
    log::info!("Model initialized");

    let mut player = PolicyPlayer {
        model,
        vs,
        board: None,
        next_turn: None,
    };
    player.usi_play()?;
    Ok(())
}
