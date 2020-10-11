use crate::constants::INPUT_CHANNELS;
use shogiutil::{Bitboard, Board, Color, Piece, Square};

pub fn make_input_feature(
    piece_bb: &[Bitboard],
    occupied: &[Bitboard],
    pieces_in_hand: &[[u8; 15]; 2],
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

pub fn make_input_feature_from_board(board: &Board, turn: Color) -> [u128; 104] {
    let (piece_bb, occupied, pieces_in_hand) = if turn == Color::Black {
        (
            board.piece_bb.to_vec(),
            board.occupied,
            board.pieces_in_hand,
        )
    } else {
        let board = board.rotate180();
        (
            board.piece_bb.to_vec(),
            board.occupied,
            board.pieces_in_hand,
        )
    };
    make_input_feature(&piece_bb, &occupied, &pieces_in_hand)
}

pub fn unflatten(f: &[u128]) -> Vec<f32> {
    let mut features = vec![0.0f32; 9 * 9 * INPUT_CHANNELS];
    for (c, &feature) in f.iter().enumerate() {
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
    features
}
