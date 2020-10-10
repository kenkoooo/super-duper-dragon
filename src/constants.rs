use crate::model::MoveDirection;

/// Pieces = 14
/// Pieces in hand
///     - Pawn: 18
///     - Lance: 4
///     - Knight: 4
///     - Silver: 4
///     - Gold: 4
///     - Bishop: 2
///     - Rook: 2
pub const INPUT_CHANNELS: usize = 104;

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
pub const MOVE_DIRECTION_LABEL_NUM: i64 = MOVE_DIRECTIONS.len() as i64 + 7;
