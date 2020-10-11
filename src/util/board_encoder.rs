use crate::constants::INPUT_CHANNELS;
use shogiutil::{Bitboard, Board, Color, Piece, Square};

pub trait BoardPacker {
    fn encode(&self) -> [u128; INPUT_CHANNELS];
}

impl BoardPacker for Board {
    fn encode(&self) -> [u128; 104] {
        let mut features = [0; INPUT_CHANNELS];
        let mut pos = 0;
        for color_id in 0..2 {
            for piece_id in 1..15 {
                let bb = self.piece_bb[piece_id] & self.occupied[color_id];
                features[pos] = bb.0;
                pos += 1;
            }

            for &piece in HANDY_PIECES.iter() {
                for i in 0..piece.max_piece_in_hand() {
                    if i < self.pieces_in_hand[color_id][piece.to_usize()].into() {
                        features[pos] = Bitboard::full().0;
                    } else {
                        features[pos] = 0;
                    }
                    pos += 1;
                }
            }
        }
        assert_eq!(pos, INPUT_CHANNELS);
        features
    }
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

pub trait ToFlatVec {
    fn to_flat_vec(&self) -> Vec<f32>;
}

impl ToFlatVec for [u128] {
    fn to_flat_vec(&self) -> Vec<f32> {
        let mut features = vec![vec![vec![0.0f32; 9]; 9]; self.len()];
        for (channel, &board) in self.iter().enumerate() {
            for i in 0..9 {
                for j in 0..9 {
                    let pos = i * 9 + j;
                    if board & (1 << pos) != 0 {
                        features[channel][i][j] = 1.0;
                    }
                }
            }
        }
        features
            .into_iter()
            .flat_map(|board| board.into_iter().flat_map(|row| row))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_flat_vec() {
        let v: Vec<u128> = vec![
            0b_111100000_000000000_101010101_000000000_000000000_000000000_000000000_000000000_000000000,
            0b_000000000_000000000_000000000_000000000_000000000_101010101_000000000_000000000_000000000,
        ];
        assert_eq!(
            v.to_flat_vec(),
            [
                // board 1
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, //
                // board 2
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            ]
        )
    }
}
