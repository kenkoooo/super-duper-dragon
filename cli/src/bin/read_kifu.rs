use anyhow::Result;
use cli::state::{GameState, PieceType, Position};
use csa::{parse_csa, Action, Color};
use std::fs::read_to_string;
use std::path::Path;

fn read_kifu<P: AsRef<Path>>(kifu_file_path: P) -> Result<()> {
    let csa_content = read_to_string(kifu_file_path)?;
    let kifu = parse_csa(&csa_content)?;

    let mut state = GameState::new();
    let mut turn = Color::Black;
    let mut lose_color = None;
    eprintln!("{:?}", kifu.moves);
    for one_move in kifu.moves {
        let action = one_move.action;
        match action {
            Action::Toryo | Action::TimeUp | Action::IllegalMove => {
                lose_color = Some(turn);
            }
            Action::IllegalAction(color) => lose_color = Some(color),
            Action::Move(color, from, to, piece) => {
                let from = Position::new(from.rank, from.file);
                let to = Position::new(to.rank, to.file);
                let piece = match piece {
                    csa::PieceType::Pawn => PieceType::Pawn,
                    csa::PieceType::Lance => PieceType::Lance,
                    csa::PieceType::Knight => PieceType::Knight,
                    csa::PieceType::Silver => PieceType::Silver,
                    csa::PieceType::Gold => PieceType::Gold,
                    csa::PieceType::Bishop => PieceType::Bishop,
                    csa::PieceType::Rook => PieceType::Rook,
                    csa::PieceType::King => PieceType::King,
                    csa::PieceType::ProPawn => PieceType::PromPawn,
                    csa::PieceType::ProLance => PieceType::PromLance,
                    csa::PieceType::ProKnight => PieceType::PromKnight,
                    csa::PieceType::ProSilver => PieceType::PromSilver,
                    csa::PieceType::Horse => PieceType::PromBishop,
                    csa::PieceType::Dragon => PieceType::PromRook,
                    csa::PieceType::All => unreachable!(),
                };
                println!("{:?} {:?} {:?}", from, to, piece);
                state.make_move(from, to, piece);
            }
            _ => {
                println!("{:?}", action);
            }
        }

        turn = match turn {
            Color::Black => Color::White,
            Color::White => Color::Black,
        };
    }

    Ok(())
}

fn main() -> Result<()> {
    read_kifu("../floodgate-kifu/wdoor2016/2016/wdoor+floodgate-600-10+gpsfish_normal_1c+Azul_demo_pentium_1c+20160211210002.csa")?;
    Ok(())
}
