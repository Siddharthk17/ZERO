#!/usr/bin/env python
"""Run perft against the local rules engine."""

from __future__ import annotations

import argparse

from zero_chess import Board


def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves():
        board.push(move)
        total += perft(board, depth - 1)
        board.pop()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fen", nargs="?", default=None)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()
    board = Board.from_fen(args.fen) if args.fen else Board.starting_position()
    for depth in range(1, args.depth + 1):
        print(depth, perft(board, depth))


if __name__ == "__main__":
    main()
