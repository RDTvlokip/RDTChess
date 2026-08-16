"""RDTChess as a UCI engine, to plug into `lichess-bot`.

    python uci.py --model RDTChess.pt

UCI assumes an engine that *searches*: it gets a time budget, thinks, and
answers when it is done. There is no search here — one forward pass and the
move is out, in about 7 ms on two CPU threads. Every `go` parameter (wtime,
btime, depth, movetime) is therefore parsed and ignored: the move does not
depend on the time available, and the clock cannot run out.

Two consequences:

* `ponder` is meaningless and declared absent — there is nothing to precompute
  while the opponent thinks;
* the reported depth is **1**, which is the truth rather than modesty: the
  network unrolls no variation.

The only half-ply of search is `FinishPlies`, which looks one move ahead solely
to avoid missing a mate or offering a stalemate. It is a policy improvement
operator, not the beginning of a search engine.
"""

import argparse
import math
import sys

import chess

DEFAULT_MODEL = "RDTChess.pt"


def say(line: str) -> None:
    """Write one UCI response.

    The explicit flush is not cosmetic: when stdout is a pipe to `lichess-bot`
    rather than a terminal, Python switches to block buffering and `bestmove`
    can sit in the buffer until 8 KB accumulate. The bot then waits forever for
    a move that was computed instantly.
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def to_centipawns(value: float) -> int:
    """Map the network's value in [-1, 1] to centipawns.

    The value head estimates an outcome probability, not a material imbalance;
    the two scales are not linearly related. This is the Leela Chess Zero
    transform, which stretches the ends so a position that is "95% winning"
    does not display as a modest +2.
    """
    clamped = max(-0.99, min(0.99, value))
    return int(111.714640912 * math.tan(1.5620688421 * clamped))


class Session:
    """One UCI session: a current position and a loaded model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.board = chess.Board()
        self.player = None
        self.finish_plies = 1
        self.threads = 2
        self.device = "auto"

    def load(self) -> None:
        """Load the network, at the latest on `isready`.

        Importing torch and reading the weights takes a second or two. Doing it
        at startup would delay the answer to `uci`, which some interfaces time;
        UCI guarantees that `isready` may take its time, so that is the place.
        """
        if self.player is not None:
            return

        import torch

        torch.set_num_threads(max(1, self.threads))
        from player import ModelPlayer

        self.player = ModelPlayer(
            self.model_path, device=self.device, finish_plies=self.finish_plies
        )

    def set_option(self, name: str, value: str) -> None:
        key = name.lower()
        if key == "finishplies":
            self.finish_plies = max(0, min(2, int(value)))
            if self.player is not None:
                self.player.finish_plies = self.finish_plies
        elif key == "threads":
            self.threads = max(1, int(value))
        elif key == "device":
            self.device = value.strip().lower()

    def set_position(self, tokens: list) -> None:
        """Apply `position [startpos | fen <6 fields>] [moves ...]`."""
        if not tokens:
            return
        if tokens[0] == "startpos":
            self.board = chess.Board()
            rest = tokens[1:]
        elif tokens[0] == "fen":
            # A FEN is six fields, and `moves` starts where it ends.
            end = tokens.index("moves") if "moves" in tokens else len(tokens)
            self.board = chess.Board(" ".join(tokens[1:end]))
            rest = tokens[end:]
        else:
            return

        if rest and rest[0] == "moves":
            for uci in rest[1:]:
                move = chess.Move.from_uci(uci)
                if move in self.board.legal_moves:
                    self.board.push(move)

    def best_move(self) -> None:
        """Pick a move and announce it, with an honest `info` line."""
        self.load()

        from engine import ChessEngine

        engine = ChessEngine(self.board.copy())
        moves, probabilities, value = self.player.evaluate_position(engine)
        if not moves:
            say("bestmove 0000")  # null move: the game is over
            return

        move = self.player.get_move(engine)
        if move is None or move == chess.Move.null():
            move = moves[0]

        # An immediate mate is reported as `mate 1`, not in centipawns: that is
        # what the interface displays, and the terminal half-ply always finds it.
        probe = self.board.copy()
        probe.push(move)
        score = "mate 1" if probe.is_checkmate() else f"cp {to_centipawns(value)}"

        confidence = float(max(probabilities)) if len(probabilities) else 0.0
        say(
            f"info depth 1 nodes 1 score {score} "
            f"string policy {confidence * 100:.1f}% over {len(moves)} legal moves "
            f"pv {move.uci()}"
        )
        say(f"bestmove {move.uci()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args, _ = parser.parse_known_args()

    session = Session(args.model)

    for raw in sys.stdin:
        tokens = raw.split()
        if not tokens:
            continue
        command = tokens[0]

        if command == "uci":
            say("id name RDTChess 128x8 5.5M")
            say("id author Theo CHARLET")
            say(f"option name FinishPlies type spin default {session.finish_plies} min 0 max 2")
            say(f"option name Threads type spin default {session.threads} min 1 max 32")
            say("option name Device type combo default auto var auto var cpu var cuda")
            say("uciok")

        elif command == "isready":
            session.load()
            say("readyok")

        elif command == "setoption":
            # setoption name <possibly multi-word name> value <value>
            if "name" in tokens:
                start = tokens.index("name") + 1
                if "value" in tokens:
                    split = tokens.index("value")
                    session.set_option(" ".join(tokens[start:split]), " ".join(tokens[split + 1:]))
                else:
                    session.set_option(" ".join(tokens[start:]), "")

        elif command == "ucinewgame":
            session.board = chess.Board()

        elif command == "position":
            session.set_position(tokens[1:])

        elif command == "go":
            # wtime, btime, depth, movetime: read by the protocol, no effect here.
            session.best_move()

        elif command == "stop":
            session.best_move()  # nothing is running, but the protocol expects a move

        elif command in ("quit", "exit"):
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
