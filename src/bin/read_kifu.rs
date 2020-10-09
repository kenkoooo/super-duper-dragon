use anyhow::Result;
use indicatif::ProgressIterator;
use shogiutil::{parse_csa_string, Bitboard, Board, Color, Move, Piece, Square};
use std::cmp::{max, min};
use std::env;
use std::fs::{read_to_string, File};
use std::io::Write;
use std::path::Path;
use super_duper_dragon::constants::{INPUT_CHANNELS, MOVE_DIRECTIONS};
use super_duper_dragon::model::{MoveDirection, Position};

fn read_single_kifu<P: AsRef<Path>>(filepath: P) -> Result<Vec<Position>> {
    let content = read_to_string(filepath)?;
    let kifu = parse_csa_string(&content)?;
    let winner = kifu.winner.expect("No winner");
    let mut data = vec![];
    let mut board = Board::default();
    for mv in kifu.moves {
        let (piece_bb, occupied, pieces_in_hand) = if mv.color == Color::Black {
            (
                board.piece_bb.to_vec(),
                board.occupied,
                board.pieces_in_hand,
            )
        } else {
            (
                board
                    .piece_bb
                    .iter()
                    .map(|b| b.rotate180())
                    .collect::<Vec<_>>(),
                [board.occupied[1].rotate180(), board.occupied[0].rotate180()],
                [board.pieces_in_hand[1], board.pieces_in_hand[0]],
            )
        };

        let is_winner_turn = mv.color == winner;
        let result = board.push_move(mv.clone())?;
        let move_label = make_output_label(&mv, mv.color, result.promoted);
        let features = make_input_feature(piece_bb, occupied, pieces_in_hand);
        data.push(Position {
            is_winner_turn,
            move_label,
            features: features.to_vec(),
        });
    }
    Ok(data)
}

fn make_input_feature(
    piece_bb: Vec<Bitboard>,
    occupied: [Bitboard; 2],
    pieces_in_hand: [[u8; 15]; 2],
) -> [u128; INPUT_CHANNELS] {
    let mut features = [0; INPUT_CHANNELS];
    let mut pos = 0;
    for color_id in 0..2 {
        for piece_id in 1..15 {
            let bb = piece_bb[piece_id] & occupied[color_id];
            let mut feature = 0;
            for rank in 1..=9 {
                for file in 1..=9 {
                    if bb.is_filled(&Square { file, rank }) {
                        let pos_i = rank - 1;
                        let pos_j = 9 - file;
                        let pos = pos_i * 9 + pos_j;
                        feature |= 1 << pos;
                    }
                }
            }
            features[pos] = feature;
            pos += 1;
        }

        for &piece in HANDY_PIECES.iter() {
            for i in 0..piece.max_piece_in_hand() {
                if i < pieces_in_hand[color_id][piece.to_usize()].into() {
                    features[pos] = 0b_111111111_111111111_111111111_111111111_111111111_111111111_111111111_111111111_111111111;
                } else {
                    features[pos] = 0;
                }
                pos += 1;
            }
        }
    }
    features
}

const HANDY_PIECES: [Piece; 7] = [
    Piece::Pawn,
    Piece::Lance,
    Piece::Knight,
    Piece::Silver,
    Piece::Gold,
    Piece::Bishop,
    Piece::Rook,
];
const MOVE_DIRECTIONS_MAP: [[Option<MoveDirection>; 3]; 3] = [
    [
        Some(MoveDirection::UpLeft),
        Some(MoveDirection::Up),
        Some(MoveDirection::UpRight),
    ],
    [Some(MoveDirection::Left), None, Some(MoveDirection::Right)],
    [
        Some(MoveDirection::DownLeft),
        Some(MoveDirection::Down),
        Some(MoveDirection::DownRight),
    ],
];

fn make_output_label(mv: &Move, color: Color, promoted: bool) -> u8 {
    let direction = if let Some(from) = mv.from.as_ref() {
        let dy = (mv.to.rank as i32) - (from.rank as i32);
        let dx = (from.file as i32) - (mv.to.file as i32);
        let direction = if dy == -2 && dx == 1 {
            MoveDirection::Up2Right
        } else if dy == -2 && dx == -1 {
            MoveDirection::Up2Left
        } else {
            let dx = min(max(dx, -1), 1);
            let dy = min(max(dy, -1), 1);
            MOVE_DIRECTIONS_MAP[(dy + 1) as usize][(dx + 1) as usize]
                .expect("Invalid move direction")
        };
        if promoted {
            direction.promote().to_byte()
        } else {
            direction.to_byte()
        }
    } else {
        let drop_id = mv.piece.to_usize() - 1;
        let direction_id = MOVE_DIRECTIONS.len() + drop_id;
        direction_id as u8
    };
    let move_to = if color == Color::Black {
        (mv.to.rank - 1) * 9 + 9 - mv.to.file
    } else {
        (9 - mv.to.rank) * 9 + mv.to.file - 1
    };
    assert!(direction < 27);
    assert!(move_to < 81);
    9 * 9 * direction + move_to
}

fn read_and_write(kifu_list_filepath: &str, bin_filepath: &str) -> Result<()> {
    let mut train_data = vec![];
    let kifu_list = read_to_string(kifu_list_filepath)?;
    let kifu_list = kifu_list.split("\n").collect::<Vec<_>>();
    for filepath in kifu_list.iter().progress() {
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

    read_and_write("./train_kifu_list.txt", "./train_kifu_list.bin")?;
    read_and_write("./test_kifu_list.txt", "./test_kifu_list.bin")?;
    Ok(())
}
