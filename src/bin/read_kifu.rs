use anyhow::Result;
use clap::Clap;
use shogiutil::{parse_csa_string, Bitboard, Board, Color, Move, Piece, Square};
use std::cmp::{max, min};
use std::env;
use std::fs::{read_to_string, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use super_duper_dragon::constants::{INPUT_CHANNELS, MOVE_DIRECTIONS};
use super_duper_dragon::model::{MoveDirection, Position};
use super_duper_dragon::progressbar::ToProgressBar;
use super_duper_dragon::util::make_input_feature::{
    make_input_feature, make_input_feature_from_board,
};
use super_duper_dragon::util::make_output_label::make_output_label;

#[derive(Clap)]
#[clap(version = "1.0", author = "kenkoooo <kenkou.n@gmail.com>")]
struct Opts {
    #[clap(long)]
    train: String,
    #[clap(long)]
    test: String,
}

fn read_single_kifu<P: AsRef<Path>>(filepath: P) -> Result<Vec<Position>> {
    let content = read_to_string(filepath)?;
    let kifu = parse_csa_string(&content)?;
    let winner = kifu.winner.expect("No winner");
    let mut data = vec![];
    let mut board = Board::default();
    for mv in kifu.moves {
        let features = make_input_feature_from_board(&board, mv.color);
        let is_winner_turn = mv.color == winner;
        let result = board.push_move(mv.clone())?;
        let move_label = make_output_label(&mv, mv.color, result.promoted);
        data.push(Position {
            is_winner_turn,
            move_label,
            features: features.to_vec(),
        });
    }
    Ok(data)
}

fn read_and_write<P: AsRef<Path>>(kifu_list_filepath: P, bin_filepath: P) -> Result<()> {
    let mut train_data = vec![];
    let kifu_list = read_to_string(kifu_list_filepath)?;
    let kifu_list = kifu_list.split("\n").collect::<Vec<_>>();
    for filepath in kifu_list.iter().progress(|state| log::info!("{}", state)) {
        if filepath.is_empty() {
            continue;
        }
        let data = read_single_kifu(filepath)?;
        train_data.extend(data);
    }

    let mut file = File::create(bin_filepath)?;
    let bin = bincode::serialize(&train_data)?;
    file.write_all(&bin)?;
    Ok(())
}

fn main() -> Result<()> {
    env::set_var("RUST_LOG", "info");
    env_logger::init();
    let opts: Opts = Opts::parse();

    let train_list = PathBuf::from(opts.train);
    let train_save = train_list.with_extension("bin");
    read_and_write(train_list, train_save)?;

    let test_list = PathBuf::from(opts.test);
    let test_save = test_list.with_extension("bin");
    read_and_write(test_list, test_save)?;
    Ok(())
}
