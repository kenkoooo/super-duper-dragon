use crate::constants::INPUT_CHANNELS;
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MoveDirection {
    Up,
    Down,
    Left,
    Right,
    UpLeft,
    UpRight,
    DownLeft,
    DownRight,
    Up2Left,
    Up2Right,
    UpPromote,
    DownPromote,
    LeftPromote,
    RightPromote,
    UpLeftPromote,
    UpRightPromote,
    DownLeftPromote,
    DownRightPromote,
    Up2LeftPromote,
    Up2RightPromote,
}

impl MoveDirection {
    pub fn to_byte(&self) -> u8 {
        match self {
            MoveDirection::Up => 0,
            MoveDirection::Down => 1,
            MoveDirection::Left => 2,
            MoveDirection::Right => 3,
            MoveDirection::UpLeft => 4,
            MoveDirection::UpRight => 5,
            MoveDirection::DownLeft => 6,
            MoveDirection::DownRight => 7,
            MoveDirection::Up2Left => 8,
            MoveDirection::Up2Right => 9,
            MoveDirection::UpPromote => 10,
            MoveDirection::DownPromote => 11,
            MoveDirection::LeftPromote => 12,
            MoveDirection::RightPromote => 13,
            MoveDirection::UpLeftPromote => 14,
            MoveDirection::UpRightPromote => 15,
            MoveDirection::DownLeftPromote => 16,
            MoveDirection::DownRightPromote => 17,
            MoveDirection::Up2LeftPromote => 18,
            MoveDirection::Up2RightPromote => 19,
        }
    }
    pub fn promote(&self) -> MoveDirection {
        match self {
            MoveDirection::Up => MoveDirection::UpPromote,
            MoveDirection::Down => MoveDirection::DownPromote,
            MoveDirection::Left => MoveDirection::LeftPromote,
            MoveDirection::Right => MoveDirection::RightPromote,
            MoveDirection::UpLeft => MoveDirection::UpLeftPromote,
            MoveDirection::UpRight => MoveDirection::UpRightPromote,
            MoveDirection::DownLeft => MoveDirection::DownLeftPromote,
            MoveDirection::DownRight => MoveDirection::DownRightPromote,
            MoveDirection::Up2Left => MoveDirection::Up2LeftPromote,
            MoveDirection::Up2Right => MoveDirection::Up2RightPromote,
            _ => unreachable!(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct DataRow {
    pub features: [u128; INPUT_CHANNELS],
    pub is_winner_turn: bool,
    pub move_label: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::MOVE_DIRECTIONS;

    #[test]
    fn test_to_byte() {
        for (i, dir) in MOVE_DIRECTIONS.iter().enumerate() {
            assert_eq!(i, dir.to_byte() as usize);
        }
    }
}
