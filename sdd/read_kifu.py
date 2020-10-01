import copy
from typing import List, Tuple, Counter

import shogi
from shogi import BLACK, WHITE
from shogi.CSA import Parser
from tqdm import tqdm

from .constants import *


def read_kifu(kifu_list_file: str) -> List[Tuple[List[int], Tuple[int, int], Tuple[Counter, Counter], int, int]]:
    positions = []

    with open(kifu_list_file, 'r') as f:
        for line in tqdm(f.readlines()):
            filepath = line.strip()
            kifu = Parser.parse_file(filepath)[0]
            win_color = BLACK if kifu["win"] == "b" else WHITE
            board = shogi.Board()
            for move in kifu["moves"]:
                if board.turn == BLACK:
                    piece_bb = copy.deepcopy(board.piece_bb)
                    occupied = copy.deepcopy((board.occupied[BLACK], board.occupied[WHITE]))
                    pieces_in_hand = copy.deepcopy((board.pieces_in_hand[BLACK], board.pieces_in_hand[WHITE]))
                else:
                    piece_bb = [bb_rotate_180(bb) for bb in board.piece_bb]
                    occupied = (bb_rotate_180(board.occupied[WHITE]), bb_rotate_180(board.occupied[BLACK]))
                    pieces_in_hand = copy.deepcopy((board.pieces_in_hand[WHITE], board.pieces_in_hand[BLACK]))

                move_label = make_output_label(shogi.Move.from_usi(move), board.turn)
                win = 1 if win_color == board.turn else 0

                positions.append((piece_bb, occupied, pieces_in_hand, move_label, win))
                board.push_usi(move)
    return positions


SQUARES_R180 = [
    shogi.I1, shogi.I2, shogi.I3, shogi.I4, shogi.I5, shogi.I6, shogi.I7, shogi.I8, shogi.I9,
    shogi.H1, shogi.H2, shogi.H3, shogi.H4, shogi.H5, shogi.H6, shogi.H7, shogi.H8, shogi.H9,
    shogi.G1, shogi.G2, shogi.G3, shogi.G4, shogi.G5, shogi.G6, shogi.G7, shogi.G8, shogi.G9,
    shogi.F1, shogi.F2, shogi.F3, shogi.F4, shogi.F5, shogi.F6, shogi.F7, shogi.F8, shogi.F9,
    shogi.E1, shogi.E2, shogi.E3, shogi.E4, shogi.E5, shogi.E6, shogi.E7, shogi.E8, shogi.E9,
    shogi.D1, shogi.D2, shogi.D3, shogi.D4, shogi.D5, shogi.D6, shogi.D7, shogi.D8, shogi.D9,
    shogi.C1, shogi.C2, shogi.C3, shogi.C4, shogi.C5, shogi.C6, shogi.C7, shogi.C8, shogi.C9,
    shogi.B1, shogi.B2, shogi.B3, shogi.B4, shogi.B5, shogi.B6, shogi.B7, shogi.B8, shogi.B9,
    shogi.A1, shogi.A2, shogi.A3, shogi.A4, shogi.A5, shogi.A6, shogi.A7, shogi.A8, shogi.A9,
]


def bb_rotate_180(bb: int) -> int:
    bb_r180 = 0
    for pos in shogi.SQUARES:
        if bb & shogi.BB_SQUARES[pos] > 0:
            bb_r180 += 1 << SQUARES_R180[pos]
    return bb_r180


POS = [
    [UP_LEFT, UP, UP_RIGHT],
    [LEFT, None, RIGHT],
    [DOWN_LEFT, DOWN, DOWN_RIGHT]
]


def make_output_label(move: shogi.Move, color: int):
    move_to = move.to_square
    move_from = move.from_square

    if color == WHITE:
        move_to = SQUARES_R180[move_to]
        if move_from is not None:
            move_from = SQUARES_R180[move_from]

    if move_from is not None:
        to_y, to_x = divmod(move_to, 9)
        from_y, from_x = divmod(move_from, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y
        if dir_y == -2 and dir_x == -1:
            move_direction = UP2_LEFT
        elif dir_y == -2 and dir_x == 1:
            move_direction = UP2_RIGHT
        else:
            dx = min(max(dir_x, -1), 1)
            dy = min(max(dir_y, -1), 1)
            move_direction = POS[dx + 1][dy + 1]
            if move_direction is None:
                raise RuntimeError(f"move_direction could not be parsed: dir_y={dir_y} dir_x={dir_x}")

        if move.promotion:
            move_direction = MOVE_DIRECTION_PROMOTED[move_direction]
    else:
        move_direction = len(MOVE_DIRECTION) + move.drop_piece_type - 1

    return 9 * 9 * move_direction + move_to


if __name__ == '__main__':
    read_kifu("./test_kifu_list.txt")
