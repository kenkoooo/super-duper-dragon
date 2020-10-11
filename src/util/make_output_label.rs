use crate::constants::MOVE_DIRECTIONS;
use crate::model::MoveDirection;
use shogiutil::{Color, Move, Piece, Square};
use std::cmp::{max, min};

pub fn make_output_label(from: &Option<Square>, to: &Square, piece: Piece, promoted: bool) -> u8 {
    let direction = if let Some(from) = from.as_ref() {
        let dy = (to.rank as i32) - (from.rank as i32);
        let dx = (from.file as i32) - (to.file as i32);
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
        let drop_id = piece.to_usize() - 1;
        let direction_id = MOVE_DIRECTIONS.len() + drop_id;
        direction_id as u8
    };
    let (i, j) = to.to_pos();
    let move_to = i * 9 + j;
    assert!(direction < 27);
    assert!(move_to < 81);
    9 * 9 * direction + move_to as u8
}

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
