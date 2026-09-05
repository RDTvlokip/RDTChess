"""Policy + value network: a residual tower with two heads.

Inference-only definition, matching the released checkpoint exactly. Nothing
here trains; see the model card for how the weights were produced.
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import ACTION_SIZE, N_PLANES


class SqueezeExcitation(nn.Module):
    """Per-channel gate computed from the whole board at once.

    The 8x8 map is averaged to one number per channel (squeeze), passed
    through a bottleneck, and turned back into a per-channel scale and bias
    (excitation) -- the Leela Chess Zero form, emitting `2 * channels` split
    into a sigmoid gate and an additive bias.
    """

    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        hidden = max(8, channels // ratio)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 2 * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc2(F.relu(self.fc1(x.mean(dim=(2, 3)))))
        scale, bias = weights.chunk(2, dim=1)
        scale = torch.sigmoid(scale).unsqueeze(-1).unsqueeze(-1)
        return x * scale + bias.unsqueeze(-1).unsqueeze(-1)


class ResidualBlock(nn.Module):
    """One residual block, optionally with a private BatchNorm per loop pass.

    When the tower is applied several times (see `ChessNetwork.recurrence`)
    the same convolutions see very different activation distributions at
    pass 1 and at pass N, so each pass beyond the first gets its own
    BatchNorm. Pass 0 keeps the plain `bn1`/`bn2` attribute names so a
    non-looped checkpoint's state-dict keys are unchanged.
    """

    def __init__(self, channels: int, recurrence: int = 1, se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.se = SqueezeExcitation(channels) if se else None

        if recurrence > 1:
            extra = recurrence - 1
            self.loop_bn1 = nn.ModuleList(nn.BatchNorm2d(channels) for _ in range(extra))
            self.loop_bn2 = nn.ModuleList(nn.BatchNorm2d(channels) for _ in range(extra))

    def forward(self, x: torch.Tensor, step: int = 0) -> torch.Tensor:
        bn1 = self.bn1 if step == 0 else self.loop_bn1[step - 1]
        bn2 = self.bn2 if step == 0 else self.loop_bn2[step - 1]
        out = F.relu(bn1(self.conv1(x)))
        out = bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return F.relu(out + x)


class ChessNetwork(nn.Module):
    """Outputs raw policy **logits** and a tanh-bounded value.

    Returning logits rather than a softmax is what lets the caller mask
    illegal moves before normalising: softmaxing over all 4096 actions and
    then zeroing ~99% of them spends most of the model's capacity learning
    that illegal moves are illegal.
    """

    def __init__(
        self,
        input_planes: int = N_PLANES,
        channels: int = 128,
        blocks: int = 8,
        action_size: int = ACTION_SIZE,
        recurrence: int = 1,
        se: bool = False,
        gated: bool = False,
    ):
        super().__init__()
        self.input_planes = input_planes
        self.channels = channels
        self.blocks = blocks
        self.action_size = action_size
        self.recurrence = max(1, int(recurrence))
        self.se = bool(se)
        self.gated = bool(gated)

        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # ModuleList rather than Sequential because a looped pass has to hand
        # each block its step index. The key names are identical either way.
        self.tower = nn.ModuleList(
            ResidualBlock(channels, self.recurrence, self.se) for _ in range(blocks)
        )

        # Learned per-pass bias, broadcast over the 8x8 board and added
        # before each traversal so the shared blocks can tell one pass from
        # the next.
        if self.recurrence > 1:
            self.depth = nn.Parameter(torch.empty(self.recurrence, channels))

        # Gated recurrence (Chen, NCU 2026):
        #     z = sigmoid(W [candidate ; previous] + b)
        #     h = z * candidate + (1 - z) * previous
        if self.gated:
            self.gate = nn.Conv2d(2 * channels, channels, 1)

        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 64, action_size)

        self.value_conv = nn.Conv2d(channels, 16, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 64, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Stem plus tower: everything the heads share."""
        x = self.stem(x)
        for step in range(self.recurrence):
            candidate = x + self.depth[step].view(1, -1, 1, 1) if self.recurrence > 1 else x
            for block in self.tower:
                candidate = block(candidate, step)
            if self.gated:
                keep = torch.sigmoid(self.gate(torch.cat([candidate, x], dim=1)))
                x = keep * candidate + (1.0 - keep) * x
            else:
                x = candidate
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = self.policy_fc(p.flatten(1))

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ChessNetwork":
        """Rebuild the network with the architecture stored in the checkpoint.

        Every entry point in this repository goes through here, which is why
        the missing-weights message lives here rather than in each script:
        cloning the code does not bring the weights, they are hosted on the
        Hub, and torch's own FileNotFoundError does not say so.
        """
        if not os.path.isfile(path):
            raise SystemExit(
                f"No weights at {path}\n\n"
                "The weights are not in this repository -- they do not belong in a\n"
                "code repo's history. Download the one you want, e.g.:\n\n"
                "  curl -L -O https://huggingface.co/RDTvlokip/RDTChess/resolve/main/RDTChess.pt\n"
                "  curl -L -O https://huggingface.co/RDTvlokip/RDTChess/resolve/main/RDTChess2.pt\n"
            )
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(**checkpoint["network_config"])
        model.load_state_dict(checkpoint["network"])
        return model.to(device).eval()
