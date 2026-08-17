"""Top-1 du réseau sur les puzzles Lichess, groupés par longueur de solution.

Le top-1 sur coups humains mesure l'accord avec un joueur moyen dans des
positions dont la plupart sont calmes. Il ne dit rien sur la **composition**:
enchaîner deux ou trois coups dont le premier n'a de sens qu'à cause du
troisième. Les puzzles donnent exactement cet axe, parce que la longueur de la
solution est connue -- c'est l'analogue échiquéen du «nombre de hops» des
papiers sur la profondeur récurrente.

Deux précautions, sans lesquelles le résultat ne veut rien dire:

**Le rating est confondu avec la longueur.** Un mat en 3 est mieux coté qu'un
mat en 1, donc «la précision baisse avec la longueur» pourrait n'être que «les
puzzles durs sont durs». On échantillonne donc par cellule (longueur × bande de
rating) et on lit la grille, pas la marge.

**Le même échantillon pour tous les modèles.** Le tirage est écrit dans
`logs/puzzle_sample.csv` et relu tel quel aux exécutions suivantes: deux
modèles voient les mêmes puzzles, donc la comparaison est appariée.

    python puzzles.py MODELE [--per-cell 1000] [--theme mateIn2] [--resample]

La base vient de https://database.lichess.org/lichess_db_puzzle.csv.zst.
"""

import argparse
import csv
import io
import os
import random
import sys
from collections import defaultdict

import chess
import numpy as np
import torch

from encoding import board_to_planes, legal_move_indices
from player import ModelPlayer

PUZZLES = "lichess_db_puzzle.csv.zst"
SAMPLE = os.path.join("logs", "puzzle_sample.csv")

# Longueurs en coups *du solveur*: la liste `Moves` alterne adversaire/solveur
# et commence par le coup de l'adversaire, donc len(Moves) // 2. Au-delà de 5
# les effectifs s'effondrent et les cellules deviennent illisibles.
LENGTHS = (1, 2, 3, 4, 5)
BANDS = ((0, 1200), (1200, 1600), (1600, 2000), (2000, 2400), (2400, 10000))


def band_of(rating: int) -> int:
    for index, (low, high) in enumerate(BANDS):
        if low <= rating < high:
            return index
    return len(BANDS) - 1


def band_label(index: int) -> str:
    low, high = BANDS[index]
    if index == 0:
        return f"<{high}"
    if index == len(BANDS) - 1:
        return f"{low}+"
    return f"{low}-{high - 1}"


def draw_sample(path: str, per_cell: int, seed: int, theme: str = ""):
    """Échantillonne `per_cell` puzzles par cellule (longueur × bande).

    Réservoir en un seul passage: la base fait 5.6 M lignes et 1.1 Go décompressé,
    la charger entièrement pour en garder 25 000 serait absurde. Le réservoir
    donne un tirage uniforme dans chaque cellule sans connaître son effectif à
    l'avance.
    """
    import zstandard

    rng = random.Random(seed)
    reservoir = defaultdict(list)
    seen = defaultdict(int)
    natural = defaultdict(int)
    total = 0

    with open(path, "rb") as raw:
        stream = zstandard.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
        reader = csv.reader(text)
        next(reader)  # en-tête
        for row in reader:
            if len(row) < 8:
                continue
            total += 1
            if total % 500_000 == 0:
                print(f"  {total // 1000} k lignes lues", file=sys.stderr)
            moves = row[2].split()
            length = len(moves) // 2
            natural[length] += 1
            if length not in LENGTHS:
                continue
            if theme and theme not in row[7].split():
                continue
            try:
                rating = int(row[3])
            except ValueError:
                continue
            key = (length, band_of(rating))
            seen[key] += 1
            if len(reservoir[key]) < per_cell:
                reservoir[key].append(row)
            else:
                slot = rng.randrange(seen[key])
                if slot < per_cell:
                    reservoir[key][slot] = row

    picked = [row for rows in reservoir.values() for row in rows]
    picked.sort(key=lambda r: r[0])  # ordre stable, indépendant du hasard
    return picked, natural, total


def load_or_draw(per_cell: int, seed: int, theme: str, resample: bool):
    if os.path.exists(SAMPLE) and not resample:
        with open(SAMPLE, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        print(f"Échantillon relu depuis {SAMPLE} ({len(rows)} puzzles)")
        return rows

    if not os.path.exists(PUZZLES):
        print(f"Base absente: {PUZZLES}", file=sys.stderr)
        print("  curl -L -O https://database.lichess.org/lichess_db_puzzle.csv.zst", file=sys.stderr)
        raise SystemExit(1)

    print(f"Tirage depuis {PUZZLES} (un passage complet, ~1 min)...")
    rows, natural, total = draw_sample(PUZZLES, per_cell, seed, theme)

    print(f"\n{total} puzzles dans la base. Distribution naturelle des longueurs:")
    for length in sorted(natural):
        share = 100.0 * natural[length] / total
        flag = "" if length in LENGTHS else "   (hors échantillon)"
        print(f"  {length} coup(s): {natural[length]:8d}  {share:5.1f} %{flag}")

    os.makedirs(os.path.dirname(SAMPLE) or ".", exist_ok=True)
    with open(SAMPLE, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    print(f"\n{len(rows)} puzzles tirés, écrits dans {SAMPLE}")
    return rows


def entry_kind(board: chess.Board, uci: str) -> str:
    """Nature du coup d'entrée: échec, prise, ou calme.

    Un réseau de politique entraîné sur des parties humaines voit des milliers
    de prises et d'échecs; ce sont des coups qu'on trouve en regardant la
    position. Un coup calme qui prépare une menace ne se justifie que par ce
    qui suit -- c'est celui-là qui exige de voir plus loin. La découpe sépare
    donc ce qui se reconnaît de ce qui se calcule.
    """
    move = chess.Move.from_uci(uci)
    if board.gives_check(move):
        return "échec"  # le plus contraignant, y compris quand il prend
    if board.is_capture(move):
        return "prise"
    return "calme"


def predict(player, boards, batch: int):
    """Coup de plus fort logit *parmi les coups légaux*, pour chaque position."""
    network = player.network
    network.eval()
    best = []
    for start in range(0, len(boards), batch):
        chunk = boards[start : start + batch]
        planes = np.stack([board_to_planes(board) for board in chunk])
        tensor = torch.from_numpy(planes).to(player.device)
        with torch.no_grad():
            logits, _ = network(tensor)
        logits = logits.float().cpu().numpy()
        for board, row in zip(chunk, logits):
            moves, indices = legal_move_indices(board)
            if not moves:
                best.append(None)
                continue
            best.append(moves[int(np.argmax(row[indices.astype(np.int64)]))])
    return best


def solve(player, rows, batch: int):
    """Interroge le réseau à **chaque** coup de la ligne et note tout.

    Le point crucial: on ne s'arrête pas au premier coup raté. La ligne avance
    toujours avec les coups du fichier, y compris quand le réseau s'est trompé.
    Sans ça, la précision au coup k ne serait mesurée que sur les puzzles dont
    les k-1 premiers coups sont déjà trouvés -- un échantillon trié par
    construction, où il devient impossible de séparer «le réseau compose» de
    «les puzzles survivants étaient les faciles».

    Avec le forçage, chaque coup k a une précision marginale q_k mesurée sur
    *tous* les puzzles de la cellule, et on peut comparer la ligne entière
    observée au produit des q_k. Le produit, c'est ce que donnerait un réseau
    qui joue coup par coup sans idée d'ensemble.

    `first` = steps[0], `solved` = tous les steps.

    Deux nuances honnêtes. Au **dernier** coup, un mat différent de celui du
    fichier compte comme juste: Lichess n'enregistre qu'une ligne mais accepte
    n'importe quel mat de même longueur. Et une sous-promotion est *impossible*
    à produire: l'encodage collapse from/to et décode toujours en dame
    (encoding.py). Ces puzzles sont comptés comme échecs et dénombrés à part --
    c'est une limite du codage des coups, pas du raisonnement.
    """
    states = []
    skipped = 0
    for row in rows:
        moves = row[2].split()
        try:
            board = chess.Board(row[1])
            board.push_uci(moves[0])
        except ValueError:
            skipped += 1  # une ligne illisible ne doit pas tuer un run de 25 000
            continue
        states.append(
            {
                "id": row[0],
                "rating": int(row[3]),
                "themes": row[7],
                "length": len(moves) // 2,
                "board": board,
                "solver": moves[1::2],
                "reply": moves[2::2],
                "step": 0,
                "legal": len(list(board.legal_moves)),
                "entry": entry_kind(board, moves[1]),
                "steps": [],
                "underpromotion": any(len(m) == 5 and m[4] != "q" for m in moves[1::2]),
            }
        )

    if skipped:
        print(f"  {skipped} puzzles illisibles, écartés")

    active = list(range(len(states)))
    rounds = 0
    while active:
        rounds += 1
        guesses = predict(player, [states[k]["board"] for k in active], batch)
        following = []
        for k, guess in zip(active, guesses):
            state = states[k]
            wanted = chess.Move.from_uci(state["solver"][state["step"]])
            last = state["step"] == len(state["solver"]) - 1
            correct = guess == wanted

            # Exception mat: au dernier coup, tout mat vaut le mat enregistré.
            if not correct and last and guess is not None:
                probe = state["board"].copy()
                probe.push(wanted)
                if probe.is_checkmate():
                    other = state["board"].copy()
                    other.push(guess)
                    correct = other.is_checkmate()

            state["steps"].append(correct)
            if last:
                continue

            # On avance toujours avec la ligne du fichier, juste ou faux.
            state["board"].push(wanted)
            state["board"].push_uci(state["reply"][state["step"]])
            state["step"] += 1
            following.append(k)
        active = following

    for state in states:
        state["first"] = state["steps"][0]
        state["solved"] = all(state["steps"])

    print(f"  {rounds} tours de propagation")
    return states


def grid(states, field: str):
    """Tableau longueur × bande d'une proportion, avec effectif et IC 95 %."""
    cells = defaultdict(lambda: [0, 0])
    for state in states:
        cell = cells[(state["length"], band_of(state["rating"]))]
        cell[0] += int(state[field])
        cell[1] += 1
    return cells


def show(title: str, cells, note: str = "") -> None:
    print(f"\n{title}")
    if note:
        print(note)
    header = "  long. | " + " | ".join(f"{band_label(b):>12}" for b in range(len(BANDS)))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for length in LENGTHS:
        pieces = []
        for band in range(len(BANDS)):
            hits, total = cells.get((length, band), (0, 0))
            if total == 0:
                pieces.append(f"{'--':>12}")
                continue
            rate = 100.0 * hits / total
            margin = 196.0 * ((rate / 100) * (1 - rate / 100) / total) ** 0.5
            pieces.append(f"{rate:5.1f}±{margin:4.1f}")
        print(f"  {length:5d} | " + " | ".join(f"{p:>12}" for p in pieces))

    counts = []
    for band in range(len(BANDS)):
        total = sum(cells.get((length, band), (0, 0))[1] for length in LENGTHS)
        counts.append(f"{total:>12d}")
    print("      n | " + " | ".join(counts))


def composition(states) -> None:
    """La ligne entière vaut-elle plus que le produit de ses coups ?

    q_k est la précision au k-ième coup mesurée sur *tous* les puzzles de la
    cellule (la ligne est forcée, donc pas de sélection). Un réseau qui joue
    coup par coup, sans idée d'ensemble, réussit la ligne entière avec la
    probabilité produit des q_k. Un réseau qui *compose* -- qui trouve le
    premier coup parce qu'il voit déjà le troisième -- fait mieux que le
    produit: ses coups sont corrélés.

    Le rapport observé/produit est donc le signal recherché. 1.0 = coup par
    coup. Nettement au-dessus = composition.
    """
    groups = defaultdict(list)
    for state in states:
        groups[(state["length"], band_of(state["rating"]))].append(state["steps"])

    print("\nComposition: ligne entière observée / produit des coups indépendants")
    print("  (1.00 = le réseau joue coup par coup; >1 = ses coups sont liés)")
    header = "  long. | " + " | ".join(f"{band_label(b):>12}" for b in range(len(BANDS)))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for length in LENGTHS:
        if length == 1:
            continue  # une ligne d'un coup est son propre produit
        pieces = []
        for band in range(len(BANDS)):
            vectors = groups.get((length, band), [])
            if len(vectors) < 50:
                pieces.append(f"{'--':>12}")
                continue
            total = len(vectors)
            product = 1.0
            for k in range(length):
                product *= sum(v[k] for v in vectors) / total
            observed = sum(all(v) for v in vectors) / total
            if product <= 0:
                pieces.append(f"{'--':>12}")
                continue
            pieces.append(f"{100 * observed:4.1f}/{100 * product:4.1f} {observed / product:4.2f}")
        print(f"  {length:5d} | " + " | ".join(f"{p:>12}" for p in pieces))

    print("\nCoup d'entrée trouvé, selon sa nature (toutes longueurs):")
    kinds = defaultdict(lambda: [0, 0])
    for state in states:
        cell = kinds[(state["entry"], band_of(state["rating"]))]
        cell[0] += int(state["steps"][0])
        cell[1] += 1
    header = "  entrée | " + " | ".join(f"{band_label(b):>12}" for b in range(len(BANDS)))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for kind in ("échec", "prise", "calme"):
        pieces = []
        for band in range(len(BANDS)):
            hits, total = kinds.get((kind, band), (0, 0))
            pieces.append(f"{100.0 * hits / total:5.1f} ({total:4d})" if total else f"{'--':>12}")
        print(f"  {kind:>6} | " + " | ".join(f"{p:>12}" for p in pieces))

    print("\nPrécision par coup de la ligne (marginale, ligne forcée):")
    for length in LENGTHS:
        if length == 1:
            continue
        vectors = [v for (lg, _), rows in groups.items() if lg == length for v in rows]
        marginals = [
            f"{100 * sum(v[k] for v in vectors) / len(vectors):4.1f}" for k in range(length)
        ]
        print(f"  {length} coups (n={len(vectors):5d}): " + " -> ".join(marginals))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--per-cell", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--theme", default="", help="ne garder qu'un thème, ex. mateIn2")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--resample", action="store_true", help="refaire le tirage")
    args = parser.parse_args()

    rows = load_or_draw(args.per_cell, args.seed, args.theme, args.resample)

    player = RLPlayer.from_checkpoint(args.model, device="auto")
    name = os.path.splitext(os.path.basename(args.model))[0]
    print(f"\n{name}: {len(rows)} puzzles sur {player.device}")
    states = solve(player, rows, args.batch)

    first = sum(s["first"] for s in states)
    solved = sum(s["solved"] for s in states)
    underpromotions = sum(s["underpromotion"] for s in states)
    legal = sum(s["legal"] for s in states) / len(states)

    show(
        f"Top-1 sur le PREMIER coup de la solution -- {name}",
        grid(states, "first"),
        "  (le réseau propose-t-il le bon coup, une seule passe avant, sans recherche)",
    )
    show(
        f"Ligne ENTIÈRE trouvée -- {name}",
        grid(states, "solved"),
        "  (tous les coups du solveur, l'adversaire jouant la réponse du fichier)",
    )

    composition(states)

    print(f"\n  global: premier coup {100.0 * first / len(states):.1f} %, "
          f"ligne entière {100.0 * solved / len(states):.1f} %")
    print(f"  {legal:.1f} coups légaux en moyenne, soit {100.0 / legal:.1f} % au hasard")
    if underpromotions:
        print(f"  {underpromotions} puzzles ({100.0 * underpromotions / len(states):.2f} %) "
              f"exigent une sous-promotion, que l'encodage ne peut pas produire")

    out = os.path.join("logs", f"puzzles_{name}.csv")
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "length", "rating", "legal", "first", "solved", "themes"])
        for state in states:
            writer.writerow([
                state["id"], state["length"], state["rating"], state["legal"],
                int(state["first"]), int(state["solved"]), state["themes"],
            ])
    print(f"  détail par puzzle: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
