use crate::constants::MOVE_DIRECTIONS;
use crate::model::MoveDirection;
use shogiutil::{Color, Move};
use std::cmp::{max, min};

pub fn make_output_label(mv: &Move, color: Color, promoted: bool) -> u8 {
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
