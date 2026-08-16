"""Policy + value network: a residual tower with two heads.

Inference-only definition, matching the released checkpoint exactly. Nothing
here trains; see the model card for how the weights were produced.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import ACTION_SIZE, N_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ChessNetwork(nn.Module):
    """Outputs raw policy **logits** and a tanh-bounded value.

    Returning logits rather than a softmax is what lets the caller mask illegal
    moves before normalising: softmaxing over all 4096 actions and then zeroing
    ~99% of them spends most of the model's capacity learning that illegal moves
    are illegal.
    """

    def __init__(
        self,
        input_planes: int = N_PLANES,
        channels: int = 128,
        blocks: int = 8,
        action_size: int = ACTION_SIZE,
    ):
        super().__init__()
        self.input_planes = input_planes
        self.channels = channels
        self.blocks = blocks
        self.action_size = action_size

        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))

        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 64, action_size)

        self.value_conv = nn.Conv2d(channels, 16, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 64, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tower(self.stem(x))

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = self.policy_fc(p.flatten(1))

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ChessNetwork":
        """Rebuild the network with the architecture stored in the checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(**checkpoint["network_config"])
        model.load_state_dict(checkpoint["network"])
        return model.to(device).eval()
