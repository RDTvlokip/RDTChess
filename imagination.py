"""RDTChess3: the network imagines before it plays.

RDTChess3 is RDTChess2 after self-distillation (RD2N7), plus a small module g
that predicts the latent state of the position *after* a move, without playing
the move on a board. When the policy hesitates, the three most likely moves are
imagined and the network's own value head judges the imagined states:

    if p(top-1) >= GATE:  play the top-1 move
    else:                 play argmax_k  -v(g(h, c_k)) + BETA * ln p(c_k)

where h is the latent state of the current position, computed once. Nothing is
played on a board to choose the move. (The only exception is `FinishPlies`,
shared with RDTChess2: with the opponent down to a bare king, a mate in one is
taken and a stalemate in one is refused.)

This file holds g (inference only) and the player. It reproduces, move for move,
the player measured in the RDTChess3 report.
"""

from typing import Optional, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import ChessEngine
from player import ModelPlayer


def flip_ranks(x: torch.Tensor) -> torch.Tensor:
    """(N, C, 8, 8): square s -> s ^ 56. The child position is encoded from the
    opponent's side, so the board is mirrored along the ranks."""
    return torch.flip(x, dims=[2])


def move_planes(idx: torch.Tensor) -> torch.Tensor:
    """Moves as two one-hot planes (from-square, to-square) in the child's frame.

    idx: (N,) = from * 64 + to, in the frame of the side to move.
    """
    n = idx.shape[0]
    out = torch.zeros(n, 2, 64, device=idx.device)
    rows = torch.arange(n, device=idx.device)
    out[rows, 0, (idx // 64) ^ 56] = 1.0
    out[rows, 1, (idx % 64) ^ 56] = 1.0
    return out.view(n, 2, 8, 8)


class _Res(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(x + self.bn2(self.conv2(y)))


class LatentLookAhead(nn.Module):
    """g(h, move) -> imagined latent state of the child position.

    The output is the mirrored h plus a learned correction: "what the move
    changes". 12 plain residual blocks of 128 channels, 3.84M parameters.
    """

    def __init__(self, channels: int = 128, blocks: int = 12):
        super().__init__()
        self.conv_in = nn.Conv2d(channels + 2, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.body = nn.Sequential(*[_Res(channels) for _ in range(blocks)])
        self.conv_out = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        hf = flip_ranks(h)
        x = F.relu(self.bn_in(self.conv_in(torch.cat([hf, move_planes(idx)], dim=1))))
        return hf + self.conv_out(self.body(x))


class ImaginationPlayer(ModelPlayer):
    """RDTChess3: the RDTChess2-family network plus latent look-ahead (rule R1)."""

    def __init__(self, checkpoint: str, device: str = "auto", finish_plies: int = 1):
        super().__init__(checkpoint, device=device, finish_plies=finish_plies)
        ck = torch.load(checkpoint, map_location="cpu")
        cfg = ck["imagination_config"]
        self.g = LatentLookAhead(cfg["channels"], cfg["blocks"])
        self.g.load_state_dict(ck["imagination"])
        self.g.to(self.device).eval()
        rule = ck["rule"]
        self.k, self.gate, self.beta = int(rule["k"]), float(rule["gate"]), float(rule["beta"])
        self.last_imagined = 0          # candidates imagined for the last move (0 = played by instinct)

    @torch.no_grad()
    def _policy_and_state(self, engine: ChessEngine, planes: np.ndarray):
        """One trunk pass: legal moves, their probabilities, the value, and h."""
        moves, indices = engine.legal_move_indices()
        if not moves:
            return [], indices, np.empty(0, dtype=np.float64), 0.0, None
        h = self.network.trunk(torch.from_numpy(planes).unsqueeze(0).to(self.device))
        logits, value = self.network.heads(h)
        sel = logits[0, torch.as_tensor(indices.astype(np.int64), device=self.device)]
        probs = torch.softmax(sel.float(), dim=0).cpu().numpy().astype(np.float64)
        total = probs.sum()
        probs = np.full(len(moves), 1.0 / len(moves)) if (not np.isfinite(total) or total <= 0.0) else probs / total
        return moves, indices, probs, float(value.item()), h

    @torch.no_grad()
    def evaluate_position(self, engine: ChessEngine, planes: Optional[np.ndarray] = None
                          ) -> Tuple[list, np.ndarray, float]:
        if planes is None:
            planes = engine.get_state_planes()
        moves, _, probs, value, _ = self._policy_and_state(engine, planes)
        return moves, probs, value

    @torch.no_grad()
    def get_move(self, engine: ChessEngine) -> Optional[chess.Move]:
        planes = engine.get_state_planes()
        moves, indices, probs, _, h = self._policy_and_state(engine, planes)
        self.last_imagined = 0
        if not moves:
            return None
        if self.finish_plies:
            probs = engine.steer_to_finish(moves, probs, self.finish_plies)
        order = np.argsort(-probs, kind="stable")[: self.k]
        if probs[order[0]] >= self.gate or len(order) < 2:
            return moves[int(order[0])]
        n = len(order)
        idx = torch.as_tensor(indices[order].astype(np.int64), device=self.device)
        imagined = self.g(h.expand(n, -1, -1, -1), idx)
        score = -self.network.heads(imagined)[1].squeeze(1).float().cpu().numpy()
        with np.errstate(divide="ignore"):
            score = score + self.beta * np.log(probs[order])
        self.last_imagined = n
        return moves[int(order[int(np.argmax(score))])]
