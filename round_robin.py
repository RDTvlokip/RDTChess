"""Round-robin entre modeles: qui bat qui, et de combien.

Le score contre le minimax sature et compresse les ecarts -- 66.3 % et 80.7 %
contre le meme adversaire ne disent pas combien de points separent les deux
modeles. Les faire jouer directement le dit.

Deux precautions:

**Ouvertures aleatoires.** Les deux camps jouent en greedy, donc deterministe:
sans variete, les 300 parties d'un appariement seraient 300 copies de la meme.
On tire 8 demi-coups legaux au hasard avant de rendre la main aux modeles.

**Ouvertures appariees.** La meme ouverture sert dans les deux sens (A blanc,
puis B blanc), sinon on compare deux echantillons differents et l'ecart entre
couleurs melange l'avantage du trait et le hasard du tirage.

    python round_robin.py [parties_par_couleur] [modele ...]

    python round_robin.py 300 RDTChess.pt autre.pt [encore.pt ...]

Deux modèles suffisent. Au-delà, tous les appariements sont joués, ce qui est
utile pour départager des réseaux que les adversaires fixes classent
différemment -- ce qui arrive dès qu'ils sont proches.
"""

import itertools
import os
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

import chess
import numpy as np

from engine import ChessEngine
from player import ModelPlayer

OPENING_PLIES = 8
MAX_MOVES = 300


def random_opening(seed: int) -> list:
    """Huit demi-coups legaux au hasard, rejoues a l'identique dans les deux sens.

    Renvoie la liste des coups plutot qu'une position: un ChessEngine par partie
    est necessaire de toute facon, et rejouer la liste evite de partager un objet
    entre fils.
    """
    for attempt in range(20):
        rng = np.random.default_rng(seed + attempt * 1_000_003)
        board = chess.Board()
        moves = []
        for _ in range(OPENING_PLIES):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = legal[int(rng.integers(len(legal)))]
            board.push(move)
            moves.append(move)
        if len(moves) == OPENING_PLIES and not board.is_game_over():
            return moves
    raise RuntimeError(f"aucune ouverture jouable pour la graine {seed}")


def play(white, black, opening: list) -> str:
    """Une partie. Renvoie 'white', 'black' ou 'draw'."""
    engine = ChessEngine()
    for move in opening:
        engine.make_move(move)

    while not engine.is_game_over() and engine.get_move_count() < MAX_MOVES:
        player = white if engine.get_turn() == chess.WHITE else black
        move = player.get_move(engine)
        if move is None:
            break
        engine.make_move(move)

    outcome = engine.get_result()
    return outcome if outcome in ("white", "black") else "draw"


def duel(white_name, black_name, players, games, base_seed):
    """`games` parties, un camp fixe de chaque cote. Compte du point de vue des blancs."""
    wins = draws = losses = 0
    openings = [random_opening(base_seed + i) for i in range(games)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = pool.map(
            lambda o: play(players[white_name], players[black_name], o), openings
        )
        for result in results:
            if result == "white":
                wins += 1
            elif result == "black":
                losses += 1
            else:
                draws += 1
    return wins, draws, losses


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: python round_robin.py GAMES_PER_COLOUR MODEL.pt MODEL.pt [MODEL.pt ...]\n"
            "  example: python round_robin.py 300 RDTChess.pt other.pt"
        )
    games = int(sys.argv[1])
    paths = sys.argv[2:]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"checkpoints introuvables: {missing}")

    # Le nom court sert d'etiquette dans les tableaux; deux fichiers de meme
    # basename seraient indistinguables, donc on le refuse plutot que de sortir
    # un tableau ambigu.
    choisis = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(set(choisis)) != len(choisis):
        raise SystemExit(f"noms de fichiers ambigus: {choisis}")

    players = {}
    for name, path in zip(choisis, paths):
        players[name] = ModelPlayer(path, device="auto")
        print(f"charge {name}", flush=True)

    rows = []
    totals = {name: [0, 0, 0] for name in choisis}

    for a, b in itertools.combinations(choisis, 2):
        # Meme graine de base pour les deux sens: ouvertures appariees.
        # crc32 et non hash(): le hachage des chaines est randomise a chaque
        # processus Python, donc hash() rendrait le tirage irreproductible.
        seed = zlib.crc32(f"{a}|{b}".encode()) % 1_000_000
        for white, black in ((a, b), (b, a)):
            w, d, l = duel(white, black, players, games, seed)
            rows.append((f"{a} vs {b}", f"{white} blanc", w, d, l, (w + d / 2) / games))
            totals[white][0] += w
            totals[white][1] += d
            totals[white][2] += l
            totals[black][0] += l
            totals[black][1] += d
            totals[black][2] += w
            print(f"  {a} vs {b}, {white} blanc : +{w} ={d} -{l}", flush=True)

    print("\n| matchup | couleur | blancs V | N | blancs D | score blancs% |")
    print("|---|---|---|---|---|---|")
    for matchup, colour, w, d, l, score in rows:
        print(f"| {matchup} | {colour} | {w} | {d} | {l} | {score*100:.1f} % |")

    print("\n| modele | V total | N total | D total | score global% |")
    print("|---|---|---|---|---|")
    for name in choisis:
        w, d, l = totals[name]
        total = w + d + l
        print(f"| {name} | {w} | {d} | {l} | {(w + d/2)/total*100:.1f} % |")


if __name__ == "__main__":
    main()
