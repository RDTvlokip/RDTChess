"""A tiny opening book, keyed by the exact move sequence played so far.

Its only job is to spread self-play games over a few sensible openings instead
of letting an untrained network wander into the same junk position every time.
Book moves are *not* used as training targets: they come from the book, not
from the policy, so they carry no gradient information (see `Trainer`).
"""

import random
from typing import Dict, List, Optional

import chess

# Generated from mainline theory written in SAN, then converted and checked
# move by move against python-chess: every key is reachable and every listed
# move legal in it. A silently wrong entry here would be invisible -- the book
# would just stop matching and self-play would quietly lose its variety.
OPENINGS: Dict[str, List[str]] = {
    "": ["e2e4", "d2d4", "c2c4", "g1f3"],
    "c2c4": ["e7e5", "g8f6", "c7c5"],
    "d2d4": ["d7d5", "g8f6", "f7f5"],
    "e2e4": ["e7e5", "c7c5", "e7e6", "c7c6", "d7d5", "g8f6", "d7d6", "g7g6"],
    "g1f3": ["d7d5", "g8f6"],
    "c2c4 c7c5": ["g1f3"],
    "c2c4 e7e5": ["b1c3"],
    "c2c4 g8f6": ["b1c3"],
    "d2d4 d7d5": ["c2c4"],
    "d2d4 f7f5": ["g2g3"],
    "d2d4 g8f6": ["c2c4", "c1f4", "g1f3"],
    "e2e4 c7c5": ["g1f3", "b1c3"],
    "e2e4 c7c6": ["d2d4"],
    "e2e4 d7d5": ["e4d5"],
    "e2e4 d7d6": ["d2d4"],
    "e2e4 e7e5": ["g1f3", "b1c3"],
    "e2e4 e7e6": ["d2d4"],
    "e2e4 g7g6": ["d2d4"],
    "e2e4 g8f6": ["e4e5"],
    "g1f3 d7d5": ["d2d4"],
    "g1f3 g8f6": ["c2c4"],
    "c2c4 c7c5 g1f3": ["g8f6"],
    "c2c4 e7e5 b1c3": ["g8f6"],
    "c2c4 g8f6 b1c3": ["e7e6"],
    "d2d4 d7d5 c2c4": ["e7e6", "c7c6", "d5c4"],
    "d2d4 f7f5 g2g3": ["g8f6"],
    "d2d4 g8f6 c1f4": ["e7e6"],
    "d2d4 g8f6 c2c4": ["e7e6", "g7g6"],
    "d2d4 g8f6 g1f3": ["e7e6"],
    "e2e4 c7c5 b1c3": ["b8c6"],
    "e2e4 c7c5 g1f3": ["d7d6", "e7e6", "b8c6"],
    "e2e4 c7c6 d2d4": ["d7d5"],
    "e2e4 d7d5 e4d5": ["d8d5"],
    "e2e4 d7d6 d2d4": ["g8f6"],
    "e2e4 e7e5 b1c3": ["g8f6"],
    "e2e4 e7e5 g1f3": ["b8c6", "g8f6"],
    "e2e4 e7e6 d2d4": ["d7d5"],
    "e2e4 g7g6 d2d4": ["f8g7"],
    "e2e4 g8f6 e4e5": ["f6d5"],
    "g1f3 d7d5 d2d4": ["g8f6"],
    "g1f3 g8f6 c2c4": ["g7g6"],
    "c2c4 c7c5 g1f3 g8f6": ["b1c3"],
    "c2c4 e7e5 b1c3 g8f6": ["g1f3"],
    "c2c4 g8f6 b1c3 e7e6": ["g1f3"],
    "d2d4 d7d5 c2c4 c7c6": ["g1f3"],
    "d2d4 d7d5 c2c4 d5c4": ["g1f3"],
    "d2d4 d7d5 c2c4 e7e6": ["b1c3"],
    "d2d4 f7f5 g2g3 g8f6": ["f1g2"],
    "d2d4 g8f6 c1f4 e7e6": ["e2e3"],
    "d2d4 g8f6 c2c4 e7e6": ["b1c3", "g1f3"],
    "d2d4 g8f6 c2c4 g7g6": ["b1c3"],
    "d2d4 g8f6 g1f3 e7e6": ["c2c4"],
    "e2e4 c7c5 b1c3 b8c6": ["g2g3"],
    "e2e4 c7c5 g1f3 b8c6": ["d2d4"],
    "e2e4 c7c5 g1f3 d7d6": ["d2d4"],
    "e2e4 c7c5 g1f3 e7e6": ["d2d4"],
    "e2e4 c7c6 d2d4 d7d5": ["e4e5", "b1c3", "e4d5"],
    "e2e4 d7d5 e4d5 d8d5": ["b1c3"],
    "e2e4 d7d6 d2d4 g8f6": ["b1c3"],
    "e2e4 e7e5 b1c3 g8f6": ["f2f4"],
    "e2e4 e7e5 g1f3 b8c6": ["f1b5", "f1c4", "d2d4"],
    "e2e4 e7e5 g1f3 g8f6": ["f3e5"],
    "e2e4 e7e6 d2d4 d7d5": ["e4e5", "b1d2", "b1c3"],
    "e2e4 g7g6 d2d4 f8g7": ["b1c3"],
    "e2e4 g8f6 e4e5 f6d5": ["d2d4"],
    "g1f3 d7d5 d2d4 g8f6": ["c2c4"],
    "g1f3 g8f6 c2c4 g7g6": ["b1c3"],
    "c2c4 c7c5 g1f3 g8f6 b1c3": ["d7d5"],
    "c2c4 e7e5 b1c3 g8f6 g1f3": ["b8c6"],
    "c2c4 g8f6 b1c3 e7e6 g1f3": ["d7d5"],
    "d2d4 d7d5 c2c4 c7c6 g1f3": ["g8f6"],
    "d2d4 d7d5 c2c4 d5c4 g1f3": ["g8f6"],
    "d2d4 d7d5 c2c4 e7e6 b1c3": ["g8f6", "c7c6"],
    "d2d4 f7f5 g2g3 g8f6 f1g2": ["e7e6"],
    "d2d4 g8f6 c1f4 e7e6 e2e3": ["c7c5"],
    "d2d4 g8f6 c2c4 e7e6 b1c3": ["f8b4"],
    "d2d4 g8f6 c2c4 e7e6 g1f3": ["b7b6"],
    "d2d4 g8f6 c2c4 g7g6 b1c3": ["f8g7", "d7d5"],
    "d2d4 g8f6 g1f3 e7e6 c2c4": ["c7c5"],
    "e2e4 c7c5 b1c3 b8c6 g2g3": ["g7g6"],
    "e2e4 c7c5 g1f3 b8c6 d2d4": ["c5d4"],
    "e2e4 c7c5 g1f3 d7d6 d2d4": ["c5d4"],
    "e2e4 c7c5 g1f3 e7e6 d2d4": ["c5d4"],
    "e2e4 c7c6 d2d4 d7d5 b1c3": ["d5e4"],
    "e2e4 c7c6 d2d4 d7d5 e4d5": ["c6d5"],
    "e2e4 c7c6 d2d4 d7d5 e4e5": ["c8f5"],
    "e2e4 d7d5 e4d5 d8d5 b1c3": ["d5a5"],
    "e2e4 d7d6 d2d4 g8f6 b1c3": ["g7g6"],
    "e2e4 e7e5 b1c3 g8f6 f2f4": ["d7d5"],
    "e2e4 e7e5 g1f3 b8c6 d2d4": ["e5d4"],
    "e2e4 e7e5 g1f3 b8c6 f1b5": ["a7a6", "g8f6"],
    "e2e4 e7e5 g1f3 b8c6 f1c4": ["f8c5", "g8f6"],
    "e2e4 e7e5 g1f3 g8f6 f3e5": ["d7d6"],
    "e2e4 e7e6 d2d4 d7d5 b1c3": ["f8b4", "g8f6"],
    "e2e4 e7e6 d2d4 d7d5 b1d2": ["g8f6"],
    "e2e4 e7e6 d2d4 d7d5 e4e5": ["c7c5"],
    "e2e4 g7g6 d2d4 f8g7 b1c3": ["d7d6"],
    "e2e4 g8f6 e4e5 f6d5 d2d4": ["d7d6"],
    "g1f3 d7d5 d2d4 g8f6 c2c4": ["e7e6"],
    "g1f3 g8f6 c2c4 g7g6 b1c3": ["d7d5"],
    "c2c4 c7c5 g1f3 g8f6 b1c3 d7d5": ["c4d5"],
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6": ["g2g3"],
    "c2c4 g8f6 b1c3 e7e6 g1f3 d7d5": ["d2d4"],
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6": ["b1c3"],
    "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6": ["e2e3"],
    "d2d4 d7d5 c2c4 e7e6 b1c3 c7c6": ["g1f3"],
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6": ["c1g5"],
    "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6": ["g1f3"],
    "d2d4 g8f6 c1f4 e7e6 e2e3 c7c5": ["c2c3"],
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4": ["e2e3"],
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6": ["g2g3"],
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5": ["c4d5"],
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7": ["e2e4"],
    "d2d4 g8f6 g1f3 e7e6 c2c4 c7c5": ["d4d5"],
    "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6": ["f1g2"],
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4": ["f3d4"],
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4": ["f3d4"],
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4": ["f3d4"],
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4": ["c3e4"],
    "e2e4 c7c6 d2d4 d7d5 e4d5 c6d5": ["f1d3"],
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5": ["g1f3"],
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5": ["d2d4"],
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6": ["g1f3"],
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4": ["f3d4"],
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6": ["b5a4", "b5c6"],
    "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6": ["e1g1"],
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5": ["c2c3"],
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6": ["f3g5"],
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6": ["e5f3"],
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4": ["e4e5"],
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6": ["c1g5"],
    "e2e4 e7e6 d2d4 d7d5 b1d2 g8f6": ["e4e5"],
    "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5": ["c2c3"],
    "e2e4 g7g6 d2d4 f8g7 b1c3 d7d6": ["f2f4"],
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6": ["g1f3"],
    "g1f3 d7d5 d2d4 g8f6 c2c4 e7e6": ["b1c3"],
    "g1f3 g8f6 c2c4 g7g6 b1c3 d7d5": ["c4d5"],
    "c2c4 c7c5 g1f3 g8f6 b1c3 d7d5 c4d5": ["f6d5"],
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3": ["d7d5"],
    "c2c4 g8f6 b1c3 e7e6 g1f3 d7d5 d2d4": ["f8e7"],
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3": ["d5c4"],
    "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6 e2e3": ["e7e6"],
    "d2d4 d7d5 c2c4 e7e6 b1c3 c7c6 g1f3": ["g8f6"],
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5": ["f8e7"],
    "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6 g1f3": ["f8e7"],
    "d2d4 g8f6 c1f4 e7e6 e2e3 c7c5 c2c3": ["b8c6"],
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3": ["e8g8"],
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3": ["c8b7"],
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5": ["f6d5"],
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4": ["d7d6"],
    "d2d4 g8f6 g1f3 e7e6 c2c4 c7c5 d4d5": ["e6d5"],
    "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2": ["f8g7"],
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4": ["g8f6"],
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4": ["g8f6"],
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4": ["b8c6"],
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4": ["c8f5"],
    "e2e4 c7c6 d2d4 d7d5 e4d5 c6d5 f1d3": ["b8c6"],
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3": ["e7e6"],
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4": ["g8f6"],
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 g1f3": ["f8g7"],
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4": ["f8c5"],
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4": ["g8f6"],
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5c6": ["d7c6"],
    "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1": ["f6e4"],
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3": ["g8f6"],
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 f3g5": ["d7d5"],
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3": ["f6e4"],
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5": ["c7c5"],
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5": ["f8e7"],
    "e2e4 e7e6 d2d4 d7d5 b1d2 g8f6 e4e5": ["f6d7"],
    "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3": ["b8c6"],
    "e2e4 g7g6 d2d4 f8g7 b1c3 d7d6 f2f4": ["g8f6"],
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6 g1f3": ["g7g6"],
    "g1f3 d7d5 d2d4 g8f6 c2c4 e7e6 b1c3": ["f8e7"],
    "g1f3 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5": ["f6d5"],
}


class OpeningBook:
    def __init__(self, max_book_moves: int = 8):
        self.max_book_moves = max_book_moves

    @staticmethod
    def _key(board: chess.Board) -> str:
        return " ".join(move.uci() for move in board.move_stack)

    def candidates(self, board: chess.Board) -> List[chess.Move]:
        """Legal book moves for this position, in book order."""
        if len(board.move_stack) >= self.max_book_moves:
            return []

        moves = []
        for uci in OPENINGS.get(self._key(board), ()):
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if board.is_legal(move):
                moves.append(move)
        return moves

    def get_move(
        self, board: chess.Board, rng: Optional[random.Random] = None
    ) -> Optional[chess.Move]:
        moves = self.candidates(board)
        if not moves:
            return None
        chooser = rng if rng is not None else random
        return chooser.choice(moves)
