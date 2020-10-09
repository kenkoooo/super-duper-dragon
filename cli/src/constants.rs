use crate::model::MoveDirection;
use shogiutil::Piece;

const PIECES: [Piece; 14] = [
    Piece::Pawn,
    Piece::Lance,
    Piece::Knight,
    Piece::Silver,
    Piece::Gold,
    Piece::Bishop,
    Piece::Rook,
    Piece::King,
    Piece::ProPawn,
    Piece::ProLance,
    Piece::ProKnight,
    Piece::ProSilver,
    Piece::ProBishop,
    Piece::ProRook,
];

pub const INPUT_CHANNELS: usize = PIECES.len()
    + Piece::Pawn.max_piece_in_hand()
    + Piece::Lance.max_piece_in_hand()
    + Piece::Knight.max_piece_in_hand()
    + Piece::Silver.max_piece_in_hand()
    + Piece::Gold.max_piece_in_hand()
    + Piece::Bishop.max_piece_in_hand()
    + Piece::Rook.max_piece_in_hand()
    + Piece::King.max_piece_in_hand()
    + Piece::ProPawn.max_piece_in_hand()
    + Piece::ProLance.max_piece_in_hand()
    + Piece::ProKnight.max_piece_in_hand()
    + Piece::ProSilver.max_piece_in_hand()
    + Piece::ProBishop.max_piece_in_hand()
    + Piece::ProRook.max_piece_in_hand();

pub const MOVE_DIRECTIONS: [MoveDirection; 20] = [
    MoveDirection::Up,
    MoveDirection::Down,
    MoveDirection::Left,
    MoveDirection::Right,
    MoveDirection::UpLeft,
    MoveDirection::UpRight,
    MoveDirection::DownLeft,
    MoveDirection::DownRight,
    MoveDirection::Up2Left,
    MoveDirection::Up2Right,
    MoveDirection::UpPromote,
    MoveDirection::DownPromote,
    MoveDirection::LeftPromote,
    MoveDirection::RightPromote,
    MoveDirection::UpLeftPromote,
    MoveDirection::UpRightPromote,
    MoveDirection::DownLeftPromote,
    MoveDirection::DownRightPromote,
    MoveDirection::Up2LeftPromote,
    MoveDirection::Up2RightPromote,
];

// directions + drops
pub const MOVE_DIRECTION_LABEL_NUM: usize = MOVE_DIRECTIONS.len() + 7;
