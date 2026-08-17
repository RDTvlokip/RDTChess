"""Score the network against the alpha-beta minimax.

    python versus.py RDTChess.pt --games 300 --depth 2

A fixed opponent measures *how well you exploit that opponent*, not general
strength. In this project the same model scored 88.7 % against the depth-1
minimax and 59.5 % against the depth-2 one -- 29 points apart. Judge on depth 2:
depth 1 saturates for a trained model, while depth 2 puts it in the range where
differences are still visible.

Two things this deliberately does NOT do.

It does not shuffle openings. Both sides are deterministic, so every game from
the initial position would be the same game. Instead the model plays each colour
in turn and the *minimax* provides the variety: its own tie-breaking differs by
seed. For model-versus-model comparisons that is not enough, and
`round_robin.py` draws paired random openings instead.

And it does not report a rating. This is a score against one hand-written
opponent, not an Elo.
"""

import argparse
import random

import chess

from algo_player import AlgoPlayer
from engine import ChessEngine
from player import ModelPlayer

MAX_MOVES = 300


def one_game(model, depth: int, model_is_white: bool, seed: int) -> str:
    """Play one game. Returns 'model', 'algo' or 'draw'."""
    engine = ChessEngine()
    algo = AlgoPlayer(depth=depth)
    # The minimax picks among equal-scoring moves with this generator, which is
    # what keeps the games from being identical.
    random.seed(seed)

    while not engine.is_game_over() and engine.get_move_count() < MAX_MOVES:
        model_turn = (engine.get_turn() == chess.WHITE) == model_is_white
        move = model.get_move(engine) if model_turn else algo.get_move(engine)
        if move is None:
            break
        engine.make_move(move)

    outcome = engine.get_result()
    if outcome not in ("white", "black"):
        return "draw"
    winner_is_white = outcome == "white"
    return "model" if winner_is_white == model_is_white else "algo"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--depth", type=int, default=2, help="minimax depth (judge on 2)")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--raw-policy", action="store_true",
                        help="disable the one-ply terminal check")
    args = parser.parse_args()

    model = ModelPlayer(args.model, device=args.device,
                        finish_plies=0 if args.raw_policy else 1)
    print(f"{args.model} on {model.device}, vs minimax depth {args.depth}")

    wins = draws = losses = 0
    for game in range(args.games):
        # Colours alternate so the first-move advantage cancels instead of
        # being handed to one side for the whole run. Measured at 52.8 % over
        # 3600 games in this project, which is larger than most of the gaps
        # being looked for.
        result = one_game(model, args.depth, game % 2 == 0, args.seed + game)
        if result == "model":
            wins += 1
        elif result == "algo":
            losses += 1
        else:
            draws += 1
        if (game + 1) % 25 == 0:
            done = game + 1
            print(f"  {done:4d}/{args.games}  {(wins + draws / 2) / done * 100:5.1f} %",
                  flush=True)

    score = (wins + draws / 2) / args.games
    # Per-game variance of a score in {0, 0.5, 1}, so the standard error is the
    # honest way to read the number: a 2-point gap on 300 games is invisible.
    pw, pd = wins / args.games, draws / args.games
    variance = pw + pd * 0.25 - score ** 2
    error = (variance / args.games) ** 0.5
    print(f"\n{score * 100:5.1f} %  (+{wins} ={draws} -{losses})  on {args.games} games"
          f"  depth={args.depth}")
    print(f"standard error {error * 100:.1f} points, so a gap under "
          f"{2 * error * 100 * 1.4:.1f} points is not measurable here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
