"""Score a network against the alpha-beta minimax, with the protocol of the published figures.

    python versus.py RDTChess.pt --games 300 --depth 2
    python versus.py RDTChess3.pt --games 300 --depth 2      # RDTChess3 plays with its imagination

A fixed opponent measures *how well you exploit that opponent*, not general
strength. In this project the same model scored 88.7 % against the depth-1
minimax and 59.5 % against the depth-2 one -- 29 points apart. Judge on depth 2:
depth 1 saturates for a trained model, while depth 2 puts it in the range where
differences are still visible.

The protocol is the one of the training-time evaluator that produced those
figures (`chess_rl/evaluator.py` in the research repository):

* on the model's turns, the opening book first (`opening_book.py`, its first
  8 plies), then, still inside the first 8 plies, a move **drawn** from the
  policy (temperature 1) -- without this, both sides are deterministic and the
  same few games repeat, which makes the score meaningless;
* after that the model plays its normal move (RDTChess3: with its imagination);
* games stop at 200 plies (a draw), colours alternate, game i uses seed + i.

One difference from the original evaluator, on purpose: the minimax's
tie-breaking is seeded per game, so a run is reproducible. The original drew
those ties from an unseeded generator, so its figures can only be matched
statistically (the standard error is printed).

It does not report a rating. This is a score against one hand-written opponent,
not an Elo.
"""

import argparse
import random

import chess
import numpy as np

from algo_player import AlgoPlayer
from engine import ChessEngine
from opening_book import OpeningBook
from player import ModelPlayer

MAX_MOVES = 200          # plies, as in the original evaluator (scored as a draw)
OPENING_RANDOM_PLIES = 8


def load_player(checkpoint: str, device: str, finish_plies: int):
    """RDTChess3 files carry the look-ahead module: they play with it."""
    import torch
    ck = torch.load(checkpoint, map_location="cpu")
    if "imagination" in ck:
        from imagination import ImaginationPlayer
        return ImaginationPlayer(checkpoint, device=device, finish_plies=finish_plies)
    return ModelPlayer(checkpoint, device=device, finish_plies=finish_plies)


def model_move(model, engine: ChessEngine, book: OpeningBook, book_rng: random.Random,
               rng: np.random.Generator):
    """Book, then a policy draw inside the first plies, then the model's own move."""
    move = book.get_move(engine.board, book_rng)
    if move is not None:
        return move
    if engine.get_move_count() < OPENING_RANDOM_PLIES:
        moves, probabilities, _ = model.evaluate_position(engine)
        if not moves:
            return None
        if model.finish_plies:
            probabilities = engine.steer_to_finish(moves, probabilities, model.finish_plies)
        return moves[int(rng.choice(len(moves), p=probabilities))]
    return model.get_move(engine)


def one_game(model, depth: int, model_is_white: bool, seed: int) -> str:
    """Play one game. Returns 'model', 'algo' or 'draw'."""
    engine = ChessEngine()
    algo = AlgoPlayer(depth=depth)
    book = OpeningBook()
    rng = np.random.default_rng(seed)
    book_rng = random.Random(seed)
    random.seed(seed)                     # the minimax's tie-breaking
    while not engine.is_game_over() and engine.get_move_count() < MAX_MOVES:
        model_turn = (engine.get_turn() == chess.WHITE) == model_is_white
        move = model_move(model, engine, book, book_rng, rng) if model_turn else algo.get_move(engine)
        if move is None:
            break
        engine.make_move(move)
    outcome = engine.get_result()
    if outcome not in ("white", "black"):
        return "draw"
    winner_is_white = outcome == "white"
    return "model" if winner_is_white == model_is_white else "algo"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--depth", type=int, default=2, help="minimax depth (judge on 2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--raw-policy", action="store_true",
                        help="disable the one-ply terminal check")
    args = parser.parse_args()

    model = load_player(args.model, args.device, 0 if args.raw_policy else 1)
    print(f"{args.model} on {model.device}, vs minimax depth {args.depth}")

    wins = draws = losses = 0
    for game in range(args.games):
        # Colours alternate so the first-move advantage cancels instead of
        # being handed to one side for the whole run.
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
    # honest way to read the number: a 2-point gap on 300 games is not measurable.
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
