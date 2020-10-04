use crate::constants::*;

type Bitboard = u128;
pub struct GameState {
    piece_bb: [Bitboard; 15],
    pieces_in_hand: [[usize; 15]; 2],
    occupied: [Bitboard; 2],
    turn: usize,
}

impl GameState {
    pub fn new() -> Self {
        Self {
            occupied: [
                BB_RANK_G | BB_H2 | BB_H8 | BB_RANK_I,
                BB_RANK_A | BB_B2 | BB_B8 | BB_RANK_C,
            ],
            piece_bb: [
                BB_VOID,                       // NONE
                BB_RANK_C | BB_RANK_G,         // PAWN
                BB_A1 | BB_I1 | BB_A9 | BB_I9, // LANCE
                BB_A2 | BB_A8 | BB_I2 | BB_I8, // KNIGHT
                BB_A3 | BB_A7 | BB_I3 | BB_I7, // SILVER
                BB_A4 | BB_A6 | BB_I4 | BB_I6, // GOLD
                BB_B2 | BB_H8,                 // BISHOP
                BB_B8 | BB_H2,                 // ROOK
                BB_A5 | BB_I5,                 // KING
                BB_VOID,                       // PROM_PAWN
                BB_VOID,                       // PROM_LANCE
                BB_VOID,                       // PROM_KNIGHT
                BB_VOID,                       // PROM_SILVER
                BB_VOID,                       // PROM_BISHOP
                BB_VOID,                       // PROM_ROOK
            ],
            turn: BLACK,
            pieces_in_hand: [[0; 15]; 2],
        }
    }

    pub fn make_move(&mut self, from: Position, to: Position, piece: PieceType) {
        assert_ne!(to.file, 0);
        assert_ne!(to.rank, 0);

        let my_turn = self.turn;
        let opponent_turn = my_turn ^ 1;

        let mut from_piece = None;
        if from.file == 0 && from.rank == 0 {
            assert!(self.pieces_in_hand[my_turn][piece.to_usize()] > 0);
            self.pieces_in_hand[my_turn][piece.to_usize()] -= 1;
            from_piece = Some(piece);
        } else {
            let from_bit = from.to_bit();
            assert_ne!(self.occupied[my_turn] & from_bit, 0);
            assert_eq!(self.occupied[opponent_turn] & from_bit, 0);
            self.occupied[my_turn] ^= from_bit;
            for piece_id in 1..=14 {
                if self.piece_bb[piece_id] & from_bit != 0 {
                    assert!(from_piece.is_none());
                    from_piece = Some(PieceType::from(piece_id));
                    self.piece_bb[piece_id] ^= from_bit;
                }
            }
        }
        assert!(from_piece.is_some());

        let to_bit = to.to_bit();
        assert_eq!(
            self.occupied[my_turn] & to_bit,
            0,
            "{:b} {:?} {:b}",
            self.occupied[my_turn],
            to,
            to.to_bit()
        );
        if self.occupied[opponent_turn] & to_bit != 0 {
            self.occupied[opponent_turn] ^= to_bit;
            let mut captured_piece = None;
            for piece_id in 1..=14 {
                if self.piece_bb[piece_id] & to_bit != 0 {
                    assert!(captured_piece.is_none());
                    captured_piece = Some(PieceType::from(piece_id));
                    self.piece_bb[piece_id] ^= to_bit;
                }
            }

            let primitive = captured_piece.unwrap().primitive_piece();
            assert!(1 <= primitive.to_usize() && primitive.to_usize() <= 7);
            self.pieces_in_hand[my_turn][primitive.to_usize()] += 1;
        }

        self.occupied[my_turn] |= to_bit;
        self.piece_bb[piece.to_usize()] |= to_bit;

        self.turn ^= 1;
    }
}

#[derive(Debug)]
pub struct Position {
    rank: u8,
    file: u8,
}

impl Position {
    pub fn new(rank: u8, file: u8) -> Self {
        Self { file, rank }
    }
    fn to_pos(&self) -> u8 {
        assert_ne!(self.rank, 0);
        assert_ne!(self.file, 0);
        (self.rank - 1) * 9 + 9 - self.file
    }
    fn to_bit(&self) -> Bitboard {
        1u128 << (self.to_pos() as u128)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PieceType {
    None,
    Pawn,
    Lance,
    Knight,
    Silver,
    Gold,
    Bishop,
    Rook,
    King,
    PromPawn,
    PromLance,
    PromKnight,
    PromSilver,
    PromBishop,
    PromRook,
}

impl PieceType {
    fn to_usize(&self) -> usize {
        match self {
            PieceType::None => 0,
            PieceType::Pawn => 1,
            PieceType::Lance => 2,
            PieceType::Knight => 3,
            PieceType::Silver => 4,
            PieceType::Gold => 5,
            PieceType::Bishop => 6,
            PieceType::Rook => 7,
            PieceType::King => 8,
            PieceType::PromPawn => 9,
            PieceType::PromLance => 10,
            PieceType::PromKnight => 11,
            PieceType::PromSilver => 12,
            PieceType::PromBishop => 13,
            PieceType::PromRook => 14,
        }
    }

    fn primitive_piece(&self) -> PieceType {
        match self {
            PieceType::None => PieceType::None,
            PieceType::Pawn => PieceType::Pawn,
            PieceType::Lance => PieceType::Lance,
            PieceType::Knight => PieceType::Knight,
            PieceType::Silver => PieceType::Silver,
            PieceType::Gold => PieceType::Gold,
            PieceType::Bishop => PieceType::Bishop,
            PieceType::Rook => PieceType::Rook,
            PieceType::King => PieceType::King,
            PieceType::PromPawn => PieceType::Pawn,
            PieceType::PromLance => PieceType::Lance,
            PieceType::PromKnight => PieceType::Knight,
            PieceType::PromSilver => PieceType::Silver,
            PieceType::PromBishop => PieceType::Bishop,
            PieceType::PromRook => PieceType::Rook,
        }
    }
}

impl From<usize> for PieceType {
    fn from(index: usize) -> Self {
        match index {
            0 => PieceType::None,
            1 => PieceType::Pawn,
            2 => PieceType::Lance,
            3 => PieceType::Knight,
            4 => PieceType::Silver,
            5 => PieceType::Gold,
            6 => PieceType::Bishop,
            7 => PieceType::Rook,
            8 => PieceType::King,
            9 => PieceType::PromPawn,
            10 => PieceType::PromLance,
            11 => PieceType::PromKnight,
            12 => PieceType::PromSilver,
            13 => PieceType::PromBishop,
            14 => PieceType::PromRook,
            _ => unreachable!(),
        }
    }
}
