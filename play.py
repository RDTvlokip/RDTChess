"""Play the model in your browser.

    python play.py                  # you are White, opens http://localhost:8000
    python play.py --black          # you are Black
    python play.py --port 8080
    python play.py --model RDTChess2.pt   # or RDTChess.pt (default: RDTChess3.pt)

The page has three modes -- play the model, play the minimax, or watch the two
of them -- and a slider for the minimax depth. No build step, no CDN: the board
is served by the standard library and the model stays in Python.
"""

import argparse
import os

import torch

from algo_player import AlgoPlayer
from gui import serve
from player import ModelPlayer

CHECKPOINT = "RDTChess3.pt"


def load_player(path: str, device: str, finish_plies: int):
    """RDTChess3 checkpoints carry the look-ahead module g, so they load the
    imagination player; RDTChess2 and RDTChess load the plain policy player."""
    checkpoint = torch.load(path, map_location="cpu") if os.path.isfile(path) else {}
    if "imagination" in checkpoint:
        from imagination import ImaginationPlayer
        return ImaginationPlayer(path, device=device, finish_plies=finish_plies), "RDTChess3 (imagines when unsure)"
    return ModelPlayer(path, device=device, finish_plies=finish_plies), os.path.splitext(os.path.basename(path))[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=CHECKPOINT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--black", action="store_true", help="you play Black")
    parser.add_argument("--depth", type=int, default=2, help="starting minimax depth")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--raw-policy", action="store_true",
                        help="disable the one-ply terminal check (weaker, purer)")
    args = parser.parse_args()

    model, label = load_player(args.model, args.device, 0 if args.raw_policy else 1)
    print(f"loaded {args.model} on {model.device}")

    # Both modes and the depth slider live in the page, so hand over the model
    # plus a way to build a minimax rather than one fixed opponent.
    return serve(
        model,
        lambda depth: AlgoPlayer(depth=depth),
        model_label=label,
        human_is_white=not args.black,
        port=args.port,
        depth=args.depth,
    )


if __name__ == "__main__":
    raise SystemExit(main())
