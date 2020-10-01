import argparse
import logging
import os
import pickle
import random
from typing import Counter, Tuple, List

import numpy as np
import shogi
import torch
from torch import nn
from tqdm import tqdm

from sdd.policy import PolicyNetwork
from sdd.read_kifu import read_kifu

Position = Tuple[List[int], Tuple[int, int], Tuple[Counter, Counter], int, int]


def load_or_read_kifu(kifu_list_path: str):
    pickle_path = kifu_list_path.replace("txt", "pickle")
    if os.path.exists(pickle_path):
        logging.info(f"Loading {pickle_path} ...")
        with open(pickle_path, "rb") as f:
            positions_train = pickle.load(f)
    else:
        logging.info(f"Loading {kifu_list_path} ...")
        positions_train = read_kifu(kifu_list_path)
        logging.info(f"Saving {pickle_path} ...")
        with open(pickle_path, "wb") as f:
            pickle.dump(positions_train, f, pickle.HIGHEST_PROTOCOL)
    return positions_train


def make_feature(position: Position):
    piece_bb, occupied, pieces_in_hand, move, win = position
    features = make_input_features(piece_bb, occupied, pieces_in_hand)
    return features, move, win


def make_input_features(piece_bb: List[int], occupied: Tuple[int, int], piece_in_hand: Tuple[Counter, Counter]):
    features: List[np.ndarray] = []
    for color in shogi.COLORS:
        for piece_type in shogi.PIECE_TYPES_WITH_NONE[1:]:
            bb = piece_bb[piece_type] & occupied[color]
            feature = np.zeros(9 * 9)
            for pos in shogi.SQUARES:
                if bb & shogi.BB_SQUARES[pos] > 0:
                    feature[pos] = 1
            features.append(feature.reshape((9, 9)))

        for piece_type in range(1, 8):
            for n in range(shogi.MAX_PIECES_IN_HAND[piece_type]):
                if piece_type in piece_in_hand[color] and n < piece_in_hand[color][piece_type]:
                    feature = np.ones(9 * 9)
                else:
                    feature = np.zeros(9 * 9)
                features.append(feature.reshape((9, 9)))

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", "-b", type=int, default=32)
    parser.add_argument("--test_batchsize", type=int, default=512)
    parser.add_argument("--epoch", "-e", type=int, default=1)
    parser.add_argument("--initmodel", "-m", default="", help="model state file")
    parser.add_argument("--resume", "-r", default="", help="optimizer state file")
    parser.add_argument("--log", default=None, help="log file path")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--eval_interval", "-i", default=1000, type=int, help="eval interval")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s\t%(levelname)s\t%(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        filename=args.log,
        level=logging.DEBUG
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PolicyNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.initmodel:
        logging.info(f"Loading {args.initmodel}")
        model.load_state_dict(torch.load(args.initmodel))
    if args.resume:
        logging.info(f"Loading {args.resume}")
        optimizer.load_state_dict(torch.load(args.resume))

    positions_train = load_or_read_kifu("./train_kifu_list.txt")
    positions_test = load_or_read_kifu("./test_kifu_list.txt")

    for epoch in range(args.epoch):
        model.train()
        sum_loss = 0
        itr = 0
        for i in tqdm(range(0, len(positions_train) - args.batchsize, args.batchsize)):
            x, t = mini_batch(positions_train, i, args.batchsize, device)
            optimizer.zero_grad()
            y = model(x)
            loss = nn.CrossEntropyLoss()(y, t)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            itr += 1

            if itr % args.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    x, t = mini_batch_for_test(positions_test, args.test_batchsize, device)
                    y = model(x)

                    pred = y.argmax(dim=1, keepdim=True)
                    accuracy = pred.eq(t.view_as(pred)).sum().item()
                    logging.info(f"iteration={itr} loss={sum_loss / itr} accuracy={accuracy}")

                model.train()

        logging.info("validate test data")
        model.eval()
        itr_test = 0
        sum_test_accuracy = 0
        with torch.no_grad():
            for i in tqdm(range(0, len(positions_test) - args.batchsize, args.batchsize)):
                x, t = mini_batch(positions_test, i, args.batchsize, device)
                y = model(x)

                itr_test += 1
                pred = y.argmax(dim=1, keepdim=True)
                sum_test_accuracy += pred.eq(t.view_as(pred)).sum().item()

        accuracy = sum_test_accuracy / itr_test
        logging.info(f"epoch={epoch} itr={itr} accuracy={accuracy}")

    torch.save(model.state_dict(), "./policy.model")
    torch.save(optimizer.state_dict(), "./policy.state")


def mini_batch(positions: List[Position], offset: int, batchsize: int, device="cuda"):
    mini_batch_data = []
    mini_batch_move = []
    for b in range(batchsize):
        features, move, win = make_feature(positions[offset + b])
        mini_batch_data.append(features)
        mini_batch_move.append(move)

    data = torch.tensor(np.array(mini_batch_data), dtype=torch.float32, device=device)
    move = torch.tensor(np.array(mini_batch_move), dtype=torch.int64, device=device)
    return data, move


def mini_batch_for_test(positions: List[Position], batchsize: int, device="cuda"):
    mini_batch_data = []
    mini_batch_move = []
    for b in range(batchsize):
        features, move, win = make_feature(random.choice(positions))
        mini_batch_data.append(features)
        mini_batch_move.append(move)

    data = torch.tensor(np.array(mini_batch_data), dtype=torch.float32, device=device)
    move = torch.tensor(np.array(mini_batch_move), dtype=torch.int64, device=device)
    return data, move


if __name__ == '__main__':
    main()
