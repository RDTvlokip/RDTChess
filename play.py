"""Play the model in your browser.

    python play.py                  # you are White, opens http://localhost:8000
    python play.py --black          # you are Black
    python play.py --port 8080

The page has three modes -- play the model, play the minimax, or watch the two
of them -- and a slider for the minimax depth. No build step, no CDN: the board
is served by the standard library and the model stays in Python.
"""

import argparse

from algo_player import AlgoPlayer
from gui import serve
from player import ModelPlayer

CHECKPOINT = "RDTChess.pt"


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

    model = ModelPlayer(args.model, device=args.device,
                        finish_plies=0 if args.raw_policy else 1)
    print(f"loaded {args.model} on {model.device}")

    # Both modes and the depth slider live in the page, so hand over the model
    # plus a way to build a minimax rather than one fixed opponent.
    return serve(
        model,
        lambda depth: AlgoPlayer(depth=depth),
        model_label="5.5M-game policy",
        human_is_white=not args.black,
        port=args.port,
        depth=args.depth,
    )


if __name__ == "__main__":
    raise SystemExit(main())
