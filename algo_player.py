"""The algorithmic opponent: alpha-beta minimax over a hand-written evaluation.

Improvements over the first version, all of which matter for training quality:

* **mate distance** -- a mate in 1 now scores higher than a mate in 5. Without
  it the engine happily shuffles pieces while "winning", never converts, and
  the RL agent never sees a loss;
* **iterative deepening + transposition table**, so the same depth costs much
  less time;
* **check evasions inside quiescence**, so tactical lines are not cut off
  mid-capture-sequence;
* draws by repetition / fifty-move are scored as draws instead of being
  evaluated statically.
"""

from typing import Dict, List, Optional, Tuple

import chess
import chess.polyglot

from engine import MATE_SCORE, PIECE_VALUES, ChessEngine

# Values above this are mate scores and must not be stored verbatim in the
# transposition table (they are relative to the ply they were found at).
MATE_THRESHOLD = MATE_SCORE - 1000

EXACT, LOWER_BOUND, UPPER_BOUND = 0, 1, 2


class AlgoPlayer:
    def __init__(self, depth: int = 3, quiescence_depth: int = 4, use_tt: bool = True):
        self.depth = max(1, depth)
        self.quiescence_depth = max(0, quiescence_depth)
        self.use_tt = use_tt
        self.nodes_searched = 0
        self._tt: Dict[int, Tuple[int, int, float, Optional[chess.Move]]] = {}
        self._killers: Dict[int, chess.Move] = {}

    # public API

    def get_move(self, engine: ChessEngine) -> Optional[chess.Move]:
        board = engine.board
        moves = list(board.legal_moves)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        self.nodes_searched = 0
        self._tt.clear()
        self._killers.clear()

        best_move = moves[0]
        # Iterative deepening: each pass seeds the next one's move ordering,
        # which is where most of the alpha-beta cutoffs come from.
        for depth in range(1, self.depth + 1):
            value, move = self._search_root(engine, depth, best_move)
            if move is not None:
                best_move = move
            if value >= MATE_THRESHOLD:
                break  # forced mate found, deeper search cannot improve on it

        return best_move

    # search

    def _search_root(
        self, engine: ChessEngine, depth: int, previous_best: Optional[chess.Move]
    ) -> Tuple[float, Optional[chess.Move]]:
        board = engine.board
        alpha, beta = -float("inf"), float("inf")
        best_value, best_move = -float("inf"), None

        for move in self._order_moves(board, previous_best, ply=0):
            board.push(move)
            value = -self._negamax(engine, depth - 1, -beta, -alpha, ply=1)
            board.pop()

            if value > best_value:
                best_value, best_move = value, move
            alpha = max(alpha, value)

        return best_value, best_move

    def _negamax(
        self, engine: ChessEngine, depth: int, alpha: float, beta: float, ply: int
    ) -> float:
        self.nodes_searched += 1
        board = engine.board

        if board.is_repetition(3) or board.halfmove_clock >= 100:
            return 0.0
        if board.is_insufficient_material():
            return 0.0

        original_alpha = alpha
        key = chess.polyglot.zobrist_hash(board) if self.use_tt else 0
        tt_move: Optional[chess.Move] = None

        if self.use_tt:
            entry = self._tt.get(key)
            if entry is not None:
                entry_depth, flag, value, tt_move = entry
                if entry_depth >= depth:
                    if flag == EXACT:
                        return value
                    if flag == LOWER_BOUND:
                        alpha = max(alpha, value)
                    elif flag == UPPER_BOUND:
                        beta = min(beta, value)
                    if alpha >= beta:
                        return value

        moves = list(board.legal_moves)
        if not moves:
            # Mate scores shrink with distance so shorter mates are preferred.
            return -MATE_SCORE + ply if board.is_check() else 0.0

        if depth <= 0:
            return self._quiesce(engine, alpha, beta, self.quiescence_depth, ply)

        best_value = -float("inf")
        best_move = None

        for move in self._order_moves(board, tt_move, ply, moves):
            board.push(move)
            value = -self._negamax(engine, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if value > best_value:
                best_value, best_move = value, move
            if best_value > alpha:
                alpha = best_value
            if alpha >= beta:
                if not board.is_capture(move):
                    self._killers[ply] = move
                break

        if self.use_tt and abs(best_value) < MATE_THRESHOLD:
            if best_value <= original_alpha:
                flag = UPPER_BOUND
            elif best_value >= beta:
                flag = LOWER_BOUND
            else:
                flag = EXACT
            self._tt[key] = (depth, flag, best_value, best_move)

        return best_value

    def _quiesce(
        self, engine: ChessEngine, alpha: float, beta: float, depth: int, ply: int
    ) -> float:
        """Search only forcing moves until the position is quiet."""
        self.nodes_searched += 1
        board = engine.board

        if board.is_check():
            # Never stand pat while in check: search every evasion.
            evasions = list(board.legal_moves)
            if not evasions:
                return -MATE_SCORE + ply
            if depth <= 0:
                return engine.evaluate_for_side_to_move()
            moves = self._order_moves(board, None, ply, evasions)
        else:
            stand_pat = engine.evaluate_for_side_to_move()
            if depth <= 0 or stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
            moves = self._order_moves(board, None, ply, moves)
            if not moves:
                return stand_pat

        best_value = -float("inf") if board.is_check() else alpha

        for move in moves:
            board.push(move)
            value = -self._quiesce(engine, -beta, -alpha, depth - 1, ply + 1)
            board.pop()

            if value > best_value:
                best_value = value
            if best_value > alpha:
                alpha = best_value
            if alpha >= beta:
                break

        return best_value

    # move ordering

    def _order_moves(
        self,
        board: chess.Board,
        priority_move: Optional[chess.Move],
        ply: int,
        moves: Optional[List[chess.Move]] = None,
    ) -> List[chess.Move]:
        if moves is None:
            moves = list(board.legal_moves)
        killer = self._killers.get(ply)

        def score(move: chess.Move) -> int:
            if move == priority_move:
                return 1_000_000
            value = 0
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                # En passant captures have no piece on the target square.
                victim_value = PIECE_VALUES[chess.PAWN] if victim is None else PIECE_VALUES[victim.piece_type]
                attacker = board.piece_at(move.from_square)
                attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
                value += 100_000 + 10 * victim_value - attacker_value
            if move.promotion:
                value += 90_000 + PIECE_VALUES.get(move.promotion, 0)
            if killer is not None and move == killer:
                value += 50_000
            return value

        return sorted(moves, key=score, reverse=True)
