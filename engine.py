"""Thin, fast wrapper around ``chess.Board``.

Adds three things the rest of the package needs: canonical tensor encoding,
game-outcome values expressed *from the side to move*, and a hand-written
material + piece-square evaluation used by the minimax opponent.
"""

from typing import List, Optional, Tuple

import chess
import numpy as np

from encoding import (
    board_to_planes,
    index_to_move,
    legal_move_indices,
    move_to_index,
)

# Module-level tables: the previous version rebuilt six 64-element lists inside
# ``__init__``, which ran on every board copy made by the search.
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

MATE_SCORE = 100_000

# Tables are written from White's point of view, rank 8 first (top row).
# Index them with ``square ^ 56`` for White and ``square`` for Black.
PAWN_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
]

# Symmetric on the file axis, unlike the classic table which has a typo on
# rank 4 -- an asymmetry biases the engine towards one wing for no reason.
QUEEN_TABLE = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

KING_MIDDLEGAME_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

# Without this the minimax opponent cannot convert K+Q vs K: it shuffles until
# the move limit, and the RL agent never learns what losing feels like.
KING_ENDGAME_TABLE = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
}

# Non-pawn, non-king material below which the king table switches to endgame.
ENDGAME_MATERIAL_THRESHOLD = 1300

# Material edge past which the game is about delivering mate, not winning more.
MOP_UP_THRESHOLD = 500
MOP_UP_WEIGHT = 250


class ChessEngine:
    """A chess position plus the encodings the learner needs."""

    __slots__ = ("board",)

    def __init__(self, board: Optional[chess.Board] = None):
        self.board = board if board is not None else chess.Board()

    # basic state

    def copy(self) -> "ChessEngine":
        return ChessEngine(self.board.copy())

    def get_turn(self) -> bool:
        return self.board.turn

    def get_move_count(self) -> int:
        return len(self.board.move_stack)

    def get_board_ascii(self) -> str:
        return str(self.board)

    def make_move(self, move: chess.Move) -> None:
        self.board.push(move)

    # encoding

    def get_state_planes(self) -> np.ndarray:
        return board_to_planes(self.board)

    def legal_move_indices(self) -> Tuple[List[chess.Move], np.ndarray]:
        return legal_move_indices(self.board)

    def legal_or_terminal(self) -> Tuple[Optional[List[chess.Move]], Optional[np.ndarray]]:
        """Legal moves and their indices, or ``(None, None)`` if the game ended.

        Generating moves is the most expensive board operation, and
        ``is_game_over`` generates them internally -- asking it first and then
        asking for the move list did the work two or three times per position.
        An empty move list already means checkmate or stalemate, so one
        generation answers both questions.
        """
        moves, indices = legal_move_indices(self.board)
        if not moves:
            return None, None
        # Fivefold repetition and the seventy-five-move rule are subsumed by
        # the threefold / fifty-move claims checked here.
        if self.board.is_insufficient_material() or self._is_claimed_draw():
            return None, None
        return moves, indices

    def move_to_index(self, move: chess.Move) -> int:
        return move_to_index(move, self.board.turn)

    def index_to_move(self, index: int) -> chess.Move:
        return index_to_move(index, self.board)

    # outcome

    def _is_claimed_draw(self) -> bool:
        """Threefold repetition or the fifty-move rule.

        Deliberately not ``claim_draw=True``: python-chess implements that by
        replaying the whole move stack *and* pushing every legal move to see
        whether a claim becomes available next ply. It costs 296 us per call
        against 2 us here, which made it the single most expensive operation in
        self-play -- more than move generation and the network combined.
        """
        return self.board.is_repetition(3) or self.board.halfmove_clock >= 100

    def is_game_over(self) -> bool:
        """Includes claimable draws.

        Without them, self-play games degenerate into 200 plies of shuffling
        and every training target ends up being 0.
        """
        return self.board.is_game_over() or self._is_claimed_draw()

    def get_result(self) -> Optional[str]:
        """``"white"``, ``"black"``, ``"draw"``, or ``None`` if still running."""
        outcome = self.board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return "draw"
            return "white" if outcome.winner == chess.WHITE else "black"
        return "draw" if self._is_claimed_draw() else None

    def terminal_value(self) -> Optional[float]:
        """Game value in ``[-1, 1]`` **from the side to move**.

        A checkmated side is the side to move, so this is ``-1.0`` there. The
        old code had this backwards and the search actively walked into mate.
        """
        outcome = self.board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return 0.0
            return 1.0 if outcome.winner == self.board.turn else -1.0
        return 0.0 if self._is_claimed_draw() else None

    # hand-written evaluation (used by the minimax opponent)

    def mating_progress(self) -> float:
        """Progress towards forcing mate, White's side, in ``[-1, 1]``.

        Zero until one side is decisively ahead. Past that, material stops
        being informative -- an agent up thirty pawns is up thirty pawns
        whatever it plays -- and the only thing that still distinguishes moves
        is the mating technique itself: drive the bare king to the edge, and
        walk your own king up to it.

        Without this, a policy trained on material alone reaches +3000 and then
        shuffles, because every legal move looks exactly as good as every other.
        """
        material = self.evaluate(mop_up=False)
        if abs(material) < MOP_UP_THRESHOLD:
            return 0.0

        board = self.board
        winner = chess.WHITE if material > 0 else chess.BLACK
        winner_king = board.king(winner)
        loser_king = board.king(not winner)
        if winner_king is None or loser_king is None:
            return 0.0

        # How far the bare king sits from the centre: 0 in the middle, 1 in a corner.
        file_gap = max(3 - chess.square_file(loser_king), chess.square_file(loser_king) - 4)
        rank_gap = max(3 - chess.square_rank(loser_king), chess.square_rank(loser_king) - 4)
        cornered = (file_gap + rank_gap) / 6.0

        # How close the attacking king has walked up: 0 far away, 1 adjacent.
        closed_in = (7 - chess.square_distance(winner_king, loser_king)) / 6.0

        progress = 0.7 * cornered + 0.3 * closed_in
        return progress if winner == chess.WHITE else -progress

    def _forces_mate(self, move: chess.Move, plies: int) -> bool:
        """True when `move` forces mate within `plies`, whatever the defence.

        Only ever called on bare-king endgames, where the defender has at most
        eight replies, so the tree stays small enough to enumerate honestly.
        """
        board = self.board
        board.push(move)
        try:
            if board.is_checkmate():
                return True
            if plies < 3 or board.is_stalemate() or board.is_insufficient_material():
                return False
            for reply in list(board.legal_moves):
                board.push(reply)
                try:
                    escaped = not any(
                        self._forces_mate(follow_up, plies - 2)
                        for follow_up in list(board.legal_moves)
                    )
                finally:
                    board.pop()
                if escaped:
                    return False  # one defence survives, so nothing is forced
            return True
        finally:
            board.pop()

    def finishing_moves(
        self, plies: int = 1
    ) -> Optional[Tuple[List[chess.Move], List[chess.Move]]]:
        """``(mates, stalemates)`` among the legal moves, or None if not worth it.

        A policy network gets no lookahead, and it shows: measured over 120
        games, 51 ended in stalemate and in 43 of them a mate in one was
        available and simply not seen. Detecting that from the board alone
        would mean internally simulating sixty-odd moves, which is what search
        is for.

        Only computed once the defender is down to a bare king -- the endgames
        where these two outcomes are one move away and the whole game hangs on
        telling them apart. Everywhere else this returns None and costs a
        popcount.
        """
        board = self.board
        defender = not board.turn
        if board.occupied_co[defender] != board.kings & board.occupied_co[defender]:
            return None

        mates: List[chess.Move] = []
        stalemates: List[chess.Move] = []
        for move in list(board.legal_moves):
            board.push(move)
            # Full detection on purpose: a king blocks its own escape ray, so
            # bitboard attack tests call trapped kings free and miss real mates.
            immediate_mate = board.is_checkmate()
            stalemated = board.is_stalemate()
            board.pop()

            if immediate_mate:
                mates.append(move)
            elif stalemated:
                stalemates.append(move)
            elif plies >= 3 and self._forces_mate(move, plies):
                mates.append(move)
        return mates, stalemates

    def steer_to_finish(
        self, moves: List[chess.Move], probabilities: np.ndarray, plies: int = 1
    ) -> np.ndarray:
        """Reweight a move distribution to take mates and refuse stalemates.

        One ply of search, used as a policy improvement operator: the network
        keeps choosing, but it no longer gets to throw a won game away on a
        move whose consequence is decided and visible. In self-play the choice
        is recorded as the behaviour distribution, so PPO's ratio accounts for
        it and the network is pulled towards making the same call unaided.
        """
        finishing = self.finishing_moves(plies)
        if finishing is None:
            return probabilities
        mates, stalemates = finishing
        if not mates and not stalemates:
            return probabilities

        adjusted = probabilities.astype(np.float64, copy=True)
        if mates:
            wanted = {move.uci() for move in mates}
            mask = np.array([move.uci() in wanted for move in moves])
        else:
            refused = {move.uci() for move in stalemates}
            mask = np.array([move.uci() not in refused for move in moves])

        adjusted[~mask] = 0.0
        total = adjusted.sum()
        if total <= 0:
            # Every remaining move is refused: keep the original rather than
            # returning something that cannot be sampled from.
            return probabilities
        return adjusted / total

    def evaluate(self, mop_up: bool = True) -> float:
        """Static evaluation in centipawns, **from White's point of view**.

        `mop_up` adds the endgame mating term; pass False for plain material,
        which is what `mating_progress` needs to avoid recursing.
        """
        board = self.board

        if board.is_checkmate():
            return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        score = 0
        non_pawn_material = 0

        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            value = PIECE_VALUES[piece_type]
            table = PIECE_SQUARE_TABLES[piece_type]

            for square in board.pieces(piece_type, chess.WHITE):
                score += value + table[square ^ 56]
            for square in board.pieces(piece_type, chess.BLACK):
                score -= value + table[square]

            if piece_type != chess.PAWN:
                count = len(board.pieces(piece_type, chess.WHITE)) + len(
                    board.pieces(piece_type, chess.BLACK)
                )
                non_pawn_material += value * count

        king_table = (
            KING_ENDGAME_TABLE
            if non_pawn_material <= ENDGAME_MATERIAL_THRESHOLD
            else KING_MIDDLEGAME_TABLE
        )
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is not None:
            score += king_table[white_king ^ 56]
        if black_king is not None:
            score -= king_table[black_king]

        score = float(score)
        if mop_up and abs(score) >= MOP_UP_THRESHOLD:
            # Gives the minimax a reason to actually convert, too: without it
            # a depth-3 search up a queen shuffles until the move limit.
            score += MOP_UP_WEIGHT * self.mating_progress()

        return score

    def evaluate_for_side_to_move(self) -> float:
        score = self.evaluate()
        return score if self.board.turn == chess.WHITE else -score
