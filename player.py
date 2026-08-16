"""The network as a chess player: one forward pass, then pick a move.

This is the inference half of the original training class. Everything the
learner needed -- the optimiser, the PPO clipping, the advantage bookkeeping --
is gone; what is left is the path a move actually takes at play time.
"""

from typing import List, Optional, Tuple

import chess
import numpy as np
import torch

from engine import ChessEngine
from network import ChessNetwork


class ModelPlayer:
    """Greedy policy play, with one optional half-move of terminal search.

    `finish_plies` is a policy improvement operator, not a search engine. It
    looks one ply ahead **only** to spot moves that mate immediately and moves
    that stalemate, then zeroes the rest or the stalemates respectively. The
    network still chooses; it simply no longer gets to throw a won game away on
    a move whose consequence is decided and visible. Set it to 0 for the raw
    policy.
    """

    def __init__(self, checkpoint: str, device: str = "auto", finish_plies: int = 1):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.network = ChessNetwork.from_checkpoint(checkpoint, device=self.device)
        self.finish_plies = finish_plies

    @torch.no_grad()
    def evaluate_position(
        self, engine: ChessEngine, planes: Optional[np.ndarray] = None
    ) -> Tuple[List[chess.Move], np.ndarray, float]:
        """Return the legal moves, their probabilities, and the value.

        The policy is renormalised **over the legal moves only**. The head emits
        all 4096 from/to pairs and about 99% of them are illegal in any given
        position, so reading the full vector is the standard way to get nonsense
        out of this model.

        Probabilities come back in float64: float32 policies routinely fail
        `numpy.random.choice`'s tolerance check on summing to one.
        """
        moves, indices = engine.legal_move_indices()
        if not moves:
            return [], np.empty(0, dtype=np.float64), 0.0

        if planes is None:
            planes = engine.get_state_planes()
        tensor = torch.from_numpy(planes).unsqueeze(0).to(self.device)

        logits, value = self.network(tensor)
        logits = logits[0, torch.as_tensor(indices.astype(np.int64), device=self.device)]
        probabilities = torch.softmax(logits.float(), dim=0).cpu().numpy().astype(np.float64)

        total = probabilities.sum()
        if not np.isfinite(total) or total <= 0.0:
            probabilities = np.full(len(moves), 1.0 / len(moves))
        else:
            probabilities /= total

        return moves, probabilities, float(value.item())

    def get_move(self, engine: ChessEngine) -> Optional[chess.Move]:
        """The move the model plays. This is what the GUI calls."""
        planes = engine.get_state_planes()
        moves, probabilities, _ = self.evaluate_position(engine, planes=planes)
        if not moves:
            return None
        if self.finish_plies:
            probabilities = engine.steer_to_finish(moves, probabilities, self.finish_plies)
        return moves[int(np.argmax(probabilities))]
