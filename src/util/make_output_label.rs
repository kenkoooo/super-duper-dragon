use crate::constants::MOVE_DIRECTIONS;
use crate::model::MoveDirection;
use shogiutil::{Piece, Square};
use std::cmp::{max, min};

pub fn make_output_label(from: &Option<Square>, to: &Square, piece: Piece, promoted: bool) -> i16 {
    let direction = if let Some(from) = from.as_ref() {
        let dy = (to.rank as i16) - (from.rank as i16);
        let dx = (from.file as i16) - (to.file as i16);
        let direction = if dy == -2 && dx == 1 {
            MoveDirection::Up2Right
        } else if dy == -2 && dx == -1 {
            MoveDirection::Up2Left
        } else {
            assert!(dx.abs() == dy.abs() || dx == 0 || dy == 0);
            let dx = min(max(dx, -1), 1);
            let dy = min(max(dy, -1), 1);
            MOVE_DIRECTIONS_MAP[(dy + 1) as usize][(dx + 1) as usize]
                .expect("Invalid move direction")
        };
        if promoted {
            direction.promote().to_byte() as i16
        } else {
            direction.to_byte() as i16
        }
    } else {
        let drop_id = piece.to_usize() - 1;
        let direction_id = MOVE_DIRECTIONS.len() + drop_id;
        direction_id as i16
    };
    let (i, j) = to.to_pos();
    let move_to = (i * 9 + j) as i16;
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

#[cfg(test)]
mod tests {
    use crate::util::make_output_label::make_output_label;
    use shogiutil::{Piece, Square};

    #[test]
    fn test_make_output_label() {
        let label = make_output_label(
            &Some(Square { file: 1, rank: 2 }),
            &Square { file: 3, rank: 4 },
            Piece::Bishop,
            true,
        );
        assert_eq!(9 * 9 * 16 + 33, label);

        let label = make_output_label(
            &Some(Square { file: 6, rank: 7 }),
            &Square { file: 6, rank: 6 },
            Piece::Pawn,
            false,
        );
        assert_eq!(48, label);

        let label = make_output_label(&None, &Square { file: 2, rank: 4 }, Piece::Knight, false);
        assert_eq!(9 * 9 * 22 + 34, label);
    }
}
