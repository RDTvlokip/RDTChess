"""Pick a move with the released network. One forward pass, no search.

    python predict.py                      # from the initial position
    python predict.py "FEN ..."            # from any FEN

The three steps that matter, and the order they go in:

1. encode the board in the **canonical frame** -- the side to move is always
   "white, playing up the board", so a position and its colour-reversed twin
   share one representation;
2. read the policy logits **only at the legal move indices**, because the head
   emits all 4096 from/to pairs and roughly 99% of them are illegal here;
3. softmax over that short list, not over the 4096.

Skipping step 2 is the usual way to get nonsense out of this model.

With RDTChess3.pt (the default), the script then shows what the full player
does: when the top move is under 60 %, it imagines its three best moves in
latent space and judges them with its own value head (`imagination.py`).

    python predict.py "FEN ..." RDTChess2.pt   # any checkpoint, as 2nd argument
    python predict.py "" RDTChess.pt           # "" = initial position
"""

import sys

import chess
import numpy as np
import torch

from encoding import board_to_planes, legal_move_indices
from network import ChessNetwork

CHECKPOINT = "RDTChess3.pt"


@torch.no_grad()
def evaluate(model: ChessNetwork, board: chess.Board, device: str = "cpu"):
    """Return the legal moves, their probabilities, and the value of `board`.

    `value` is in [-1, 1] and always answers "how good is this for the side to
    move", which is what the canonical frame buys.
    """
    moves, indices = legal_move_indices(board)
    if not moves:
        return [], np.empty(0), 0.0

    planes = torch.from_numpy(board_to_planes(board)).unsqueeze(0).to(device)
    logits, value = model(planes)

    legal = logits[0, torch.as_tensor(indices.astype(np.int64), device=device)]
    probabilities = torch.softmax(legal.float(), dim=0).cpu().numpy()
    return moves, probabilities, float(value.item())


def main() -> int:
    board = chess.Board(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else chess.Board()
    path = sys.argv[2] if len(sys.argv) > 2 else CHECKPOINT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessNetwork.from_checkpoint(path, device=device)

    moves, probabilities, value = evaluate(model, board, device)
    if not moves:
        print("no legal move -- the game is over")
        return 0

    order = np.argsort(-probabilities)
    print(board)
    print(f"\n{'white' if board.turn else 'black'} to move")
    print(f"value {value:+.3f}   (positive favours the side to move)\n")
    print("top 5 moves")
    for rank in order[:5]:
        print(f"  {board.san(moves[rank]):8s} {probabilities[rank] * 100:5.1f} %")

    print(f"\npolicy alone plays: {board.san(moves[order[0]])}")

    if "imagination" in torch.load(path, map_location="cpu"):
        from engine import ChessEngine
        from imagination import ImaginationPlayer

        player = ImaginationPlayer(path, device=device, finish_plies=1)
        move = player.get_move(ChessEngine(board.copy()))
        how = (f"imagined its {player.last_imagined} best moves" if player.last_imagined
               else "policy sure enough, no imagination")
        print(f"RDTChess3 plays:    {board.san(move)}   ({how})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
