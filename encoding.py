"""Board <-> tensor and move <-> index conversions.

Everything here lives in the **canonical frame**: the side to move is always
"white, playing up the board". When black is to move, every square is mirrored
vertically (``square ^ 56``) and the piece colours are swapped.

Two consequences, both wanted:

* the value head always predicts "how good is this for the player to move",
  which is a well defined question -- unlike the previous absolute encoding
  which did not even tell the network whose turn it was;
* a position and its colour-reversed twin share the same representation, so
  every self-play game trains both colours at once.
"""

from typing import List, Tuple

import chess
import numpy as np

# 12 piece planes + 4 castling + 1 en-passant + 1 fifty-move + 1 repetition
N_PLANES = 19
ACTION_SIZE = 64 * 64  # from-square * 64 + to-square, canonical frame

PIECE_ORDER: Tuple[int, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

# Plane layout, for reference / debugging.
PLANE_NAMES: Tuple[str, ...] = (
    "us_pawn", "us_knight", "us_bishop", "us_rook", "us_queen", "us_king",
    "them_pawn", "them_knight", "them_bishop", "them_rook", "them_queen", "them_king",
    "us_castle_k", "us_castle_q", "them_castle_k", "them_castle_q",
    "en_passant", "fifty_move", "repetition",
)


def canonical_square(square: int, turn: bool) -> int:
    """Map a board square into the frame of the side to move.

    ``x ^ 56`` flips the rank and keeps the file, and is its own inverse, so the
    same function encodes and decodes.
    """
    return square if turn == chess.WHITE else square ^ 56


def move_to_index(move: chess.Move, turn: bool) -> int:
    """Flatten a move to ``[0, 4096)`` in the canonical frame.

    Promotions collapse onto the same index as the plain from/to move; decoding
    resolves the ambiguity in favour of the queen (see :func:`index_to_move`).
    Under-promotions are therefore not representable -- a deliberate trade-off,
    they are worth well under 0.1% of moves in practice.
    """
    return canonical_square(move.from_square, turn) * 64 + canonical_square(move.to_square, turn)


def index_to_squares(index: int, turn: bool) -> Tuple[int, int]:
    """Inverse of :func:`move_to_index`, returning real (non-canonical) squares."""
    from_canonical, to_canonical = divmod(int(index), 64)
    return canonical_square(from_canonical, turn), canonical_square(to_canonical, turn)


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Resolve a policy index against the legal moves of `board`.

    Returns ``chess.Move.null()`` when the index matches no legal move, which
    lets callers fail loudly instead of silently playing something else.
    """
    from_square, to_square = index_to_squares(index, board.turn)

    # Try the queen promotion first, then the plain move: both are cheap
    # membership tests and cover every legal case except under-promotions.
    for promotion in (None, chess.QUEEN):
        move = chess.Move(from_square, to_square, promotion=promotion)
        if board.is_legal(move):
            return move

    for move in board.legal_moves:
        if move.from_square == from_square and move.to_square == to_square:
            return move

    return chess.Move.null()


def legal_move_indices(board: chess.Board) -> Tuple[List[chess.Move], np.ndarray]:
    """Return the legal moves and their canonical policy indices.

    Duplicate indices (under-promotions sharing a from/to pair) are dropped so
    that the index array can be used directly as a legality mask. The common
    case -- no promotion available -- takes the fast path and never builds the
    deduplication set.
    """
    moves = list(board.legal_moves)
    if board.turn == chess.WHITE:
        indices = [m.from_square * 64 + m.to_square for m in moves]
    else:
        indices = [(m.from_square ^ 56) * 64 + (m.to_square ^ 56) for m in moves]

    if len(set(indices)) != len(indices):
        unique: dict = {}
        for move, index in zip(moves, indices):
            # Keep the queen promotion, which is what index_to_move decodes to.
            if index not in unique or move.promotion == chess.QUEEN:
                unique[index] = move
        indices = list(unique.keys())
        moves = list(unique.values())

    return moves, np.asarray(indices, dtype=np.int16)


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode `board` as a ``(N_PLANES, 8, 8)`` float32 tensor."""
    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    turn = board.turn

    # Unpacking the twelve bitboards in one go is about four times faster than
    # walking the occupied squares in Python, and this runs once per position
    # of every game.
    raw = (
        np.array(
            [board.pieces_mask(piece, color) for color in (turn, not turn) for piece in PIECE_ORDER],
            dtype=">u8",
        )
        .view(np.uint8)
        .reshape(12, 8)
    )
    bits = np.unpackbits(raw, axis=1).reshape(12, 8, 8)
    # Big-endian bytes come out rank 7 first, and file 7 first within a rank.
    # Reversing both axes gives White's frame; for Black the rank reversal is
    # exactly the ``square ^ 56`` mirror, so it cancels out.
    planes[:12] = bits[:, ::-1, ::-1] if turn == chess.WHITE else bits[:, :, ::-1]

    planes[12] = float(board.has_kingside_castling_rights(turn))
    planes[13] = float(board.has_queenside_castling_rights(turn))
    planes[14] = float(board.has_kingside_castling_rights(not turn))
    planes[15] = float(board.has_queenside_castling_rights(not turn))

    if board.ep_square is not None:
        canonical = canonical_square(board.ep_square, turn)
        planes[16, canonical >> 3, canonical & 7] = 1.0

    planes[17] = min(board.halfmove_clock, 100) / 100.0
    planes[18] = 1.0 if board.is_repetition(2) else 0.0

    return planes
