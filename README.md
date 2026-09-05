# ♟️ RDTChess

![RDTChess](thumbnail.png)

A **11.05M-parameter** residual network — recurrent depth ×2, gated memory, Squeeze-and-Excitation — that plays chess in **one forward pass** — no search, no MCTS, no opening book. Trained from scratch on a single GTX 1080 Ti by supervised imitation of **≈10.99M** human games from Lichess. This is `RDTChess2.pt`, currently the model live on the [Lichess bot](https://lichess.org/@/RDTChessBot).

[![Play on Lichess](https://img.shields.io/badge/play%20it%20on-Lichess-0CA678?style=flat-square&logo=lichess&logoColor=white)](https://lichess.org/@/RDTChessBot)
[![Weights on Hugging Face](https://img.shields.io/badge/weights-Hugging%20Face-ffcc4d?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/RDTvlokip/RDTChess)
![license](https://img.shields.io/badge/license-Apache--2.0-4C6EF5?style=flat-square)
![params](https://img.shields.io/badge/params-11.05M-4C6EF5?style=flat-square)
![games](https://img.shields.io/badge/games-~11M-4C6EF5?style=flat-square)
![positions](https://img.shields.io/badge/positions-801.3M-4C6EF5?style=flat-square)
![search](https://img.shields.io/badge/search-none-495057?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)

> ⚠️ **One seed, one run.** Every number in the Identity and Evaluation sections just below comes from a single training run on a single seed (123). Directions are consistent across three of the four instruments used there; magnitudes are not multi-seed validated. Read this as a lab notebook, not a benchmark.

This repository holds the **inference code**. The weights live on [Hugging Face](https://huggingface.co/RDTvlokip/RDTChess) — **two** checkpoints: `RDTChess2.pt` (this page's main subject, live on Lichess) is recurrent depth ×2 + gated + SE on a 128×8 tower, trained on ≈11M games — compared against the plain architecture just below. `RDTChess.pt` is that plain architecture on its own: the original, single-seed, 5.5M-game run, described with its own numbers in [📎 RDTChess (original)](#rdtchess) near the end of this page. Smaller and differently-shaped siblings were trained alongside both, but none are released, and none of the numbers on this page come from them.

---

## 🧩 Identity

| | |
|---|---|
| **Architecture** | Residual tower, **128 channels × 8 blocks**, recurrence ×2, gated, SE, two heads |
| **Parameters** | 11.05 M |
| **Input** | 19 planes of 8×8, canonical frame |
| **Policy output** | 4096 raw logits (from-square × 64 + to-square) |
| **Value output** | one scalar in [−1, 1], `tanh` |
| **Search** | **none** — one forward pass per move |
| **Training data** | ≈10.99 M Lichess games, both players ≥ 1800 Elo (8 sequential slices, one seed carried through with `--resume`) |
| **Positions seen** | 801,295,726, summed across the 8 slices |
| **Training time** | 68.6 h total on a GTX 1080 Ti, across the 8 sequential runs |

This is `RDTChess2.pt`. `RDTChess.pt` — the plain architecture on its own, no recurrence, no gate, no SE — has its own Identity/Evaluation/Data-scaling numbers in [📎 RDTChess (original)](#rdtchess) near the end of this page.

### The encoding, and why it is shaped this way

Everything lives in a **canonical frame**: the side to move is always "White, playing up the board". When Black is to move, every square is mirrored (`square ^ 56`) and the piece colours swap.

Two consequences, both deliberate:

- the value head always answers **"how good is this for the player to move"**, which is a well-defined question — unlike an absolute encoding, which does not even tell the network whose turn it is;
- a position and its colour-reversed twin share one representation, so every game trains both colours at once.

The 19 planes: 12 for pieces (6 ours, 6 theirs), 4 castling rights, 1 en passant, 1 fifty-move counter, 1 repetition flag.

The policy head emits **logits, not a softmax**. That is what lets the caller mask illegal moves *before* normalising. Softmaxing over all 4096 actions and then zeroing ~99 % of them spends most of the model's capacity learning that illegal moves are illegal — capacity this size cannot spare.

---

## How it was made

**Architecture.** Same 128×8 residual tower as `RDTChess.pt`, plus three additions applied together:

- **Recurrent depth ×2** — the tower is applied twice per move, sharing the same convolution weights both times (effective depth ×2 at no extra parameter cost for the convolutions themselves).
- **Gated shortcut** — a learned per-channel, per-square gate decides how much of the previous pass to keep versus overwrite (`z = sigmoid(W[candidate; previous])`, initialised at `sigmoid(-2) ≈ 0.12` so ~88 % of the previous state survives each pass by default).
- **Squeeze-and-Excitation** in every residual block — a per-channel scale and bias computed from the whole board at once, buying long-range interaction (e.g. a1–h8) without extra convolutional depth.

**Data.** The Lichess July 2026 standard-game dump, split into 8 sequential slices of ≈1.37M games each (both players rated ≥ 1800). One seed (123) trained on the first slice, then carried through the other 7 with `--resume`, so the final checkpoint has seen all ≈10.99M games and 801,295,726 positions, one epoch each.

**Objective.** Plain behaviour cloning. The policy is cross-entropy against the move the human actually played, computed over the legal moves only. The value head regresses the final game result, expressed from the side to move.

**Recipe.** AdamW, learning rate 1e-3, weight decay 1e-4, betas (0.9, 0.999), batch size 512, gradient clipping 1.0, one epoch per slice, no learning-rate schedule.

That is the whole method. No self-play, no MCTS, no distillation from an engine, no opening book, no endgame tablebase.

---

## Evaluation

**Compared against the plain (non-recurrent, non-gated, non-SE) 128×8 architecture, same data, same seed, one run each time — puzzles, Maia and minimax across all 8 slices; the round-robin across the first 7, run before the 8th finished training:**

| yardstick | result |
|---|---|
| Puzzles, first move (McNemar, 23,845 puzzles) — single forward pass, no search | **ahead 8 / 8** — never inverted, always significant |
| Maia-1100 / 1500 / 1900, no search either side — 300 games per slice per level | **ahead 23 / 24** — one isolated reversal, recovered next slice |
| Full round-robin, 14 checkpoints, 91 pairs, 27,300 games, no search either side | **ahead 7 / 7** — never inverted, the most stable instrument of all four |
| Minimax depth 2 (n=300) — an adversary that searches, unlike the three above | ahead 4 / 8 — no stable direction, each swing confirmed on a second independent draw |

> 🚫 **No absolute Elo is claimed.** This model has never been played against a rated pool. Every figure here is a *difference* measured inside one family of models, on one training seed.

**Read this precisely, not optimistically.** Puzzles, Maia, and the round-robin all judge in **one pass, no search** — the same regime this model is actually deployed in — and all three are stable and one-sided across every slice tested. Minimax is a stress test against an opponent that searches for a refutation; the mechanism never collapses against it, but the signal is too noisy at n=300 to claim it wins there either. On the 5 hand-picked positions where the production model actually lost a real game, both architectures bounce between fixing and re-breaking the same handful of positions slice to slice — until, on the very last slice, both converged on choosing the exact identical move on all 6 tested, the first time that happened.

This was confirmed on **one seed (123) only**, extended sequentially across 8 slices of the same data with `--resume`. The 1.38M-game starting point was separately confirmed on three independent seeds (24, 123, 454) before this single-seed extension began; the extension itself has not been repeated on another seed.

### What it is good at, and where it breaks

Measured on the same **23,845 Lichess puzzles** as the comparison above, on the final checkpoint (all 8 slices). One move per position, no search.

| | |
|---|---|
| first move of the solution | **61.5 %** |
| entire line | **42.5 %** |
| random baseline | 3.6 % (28.0 legal moves on average) |

#### The entry move is the hard one

The network is asked at **every** move of the line — even after it has already erred, so that no step is measured on a pre-filtered sample.

| length | accuracy, first move → last |
|---|---|
| 2 moves | 54.5 → **78.3** |
| 3 moves | 56.9 → 71.1 → **87.0** |
| 4 moves | 64.1 → 72.7 → 77.7 → **89.6** |
| 5 moves | 68.7 → 75.5 → 77.6 → 83.0 → **91.7** |

**The entry is the hardest move of every line here too, by 10 to 22 points — a smaller gap than `RDTChess.pt`'s 20 to 35, but not closed.** The last move of the line sits around 90 %, same as the original architecture.

#### And it is the *quiet* entry that breaks, same as the original

First move found, split by what kind of move it is.

| entry move | <1200 | 1200-1599 | 1600-1999 | 2000-2399 | 2400+ |
|---|---|---|---|---|---|
| check | 79.7 | 65.2 | 52.6 | 42.0 | 55.3 |
| capture | 92.2 | 77.4 | 63.8 | 60.7 | **59.9** |
| quiet | **95.1** | 80.3 | 59.1 | 42.0 | **33.3** |

The quiet move is still the hardest category at high rating — 33.3 % at 2400+, essentially unchanged from `RDTChess.pt`'s 32.8 %. Every band improved a little (quiet at 1600-1999: 59.1 vs 54.7; at 2000-2399: 42.0 vs 39.3), but the *shape* — quiet collapsing from easiest to hardest as rating rises — is identical. Recurrent depth, gating and SE bought accuracy; they did not change what kind of move the network struggles to initiate.

---

## Play it right now

The model runs live as a Lichess bot: **[lichess.org/@/RDTChessBot](https://lichess.org/@/RDTChessBot)**

Challenge it in bullet, blitz, rapid or classical, casual or rated. Correspondence is declined on purpose — the bot answers in about 7 ms, so a game played over days would tie up a slot for weeks and tell you nothing.

**The bot currently runs `RDTChess2.pt`** — check `uci.py`'s `DEFAULT_MODEL` if you're reproducing a game and a move doesn't match what you'd expect from `RDTChess.pt`.

## Run it yourself

```bash
git clone https://github.com/RDTvlokip/RDTChess
cd RDTChess
pip install -r requirements.txt

# the weights are not in this repo (44 MB each); pull the one you want from Hugging Face
curl -L -o RDTChess2.pt https://huggingface.co/RDTvlokip/RDTChess/resolve/main/RDTChess2.pt
curl -L -o RDTChess.pt https://huggingface.co/RDTvlokip/RDTChess/resolve/main/RDTChess.pt
```

### Pick a move

```bash
python predict.py                                        # initial position
python predict.py "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1" # any FEN
```

```
top 5 moves
  Ra8#      64.3 %
  Ra7       11.4 %
  Re1        5.6 %
```

### Play it in your browser

```bash
python play.py              # you are White, opens http://localhost:8000
python play.py --black
python play.py --port 8080 --depth 3
```

Three modes in the page — play the model, play the minimax, or watch the two of them — plus a depth slider. Standard-library HTTP server, no build step, no CDN.

### As a UCI engine

```bash
python uci.py --model RDTChess2.pt
```

Speaks UCI, so it plugs into any interface or into [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot). It reports depth 1 and ignores every time-control parameter, which is the truth rather than modesty: there is no search to budget.

### In your own code

```python
import chess, numpy as np, torch
from encoding import board_to_planes, legal_move_indices
from network import ChessNetwork

model = ChessNetwork.from_checkpoint("RDTChess2.pt", device="cpu")  # or "RDTChess.pt"
board = chess.Board()

moves, indices = legal_move_indices(board)          # legal moves + their indices
planes = torch.from_numpy(board_to_planes(board)).unsqueeze(0)

with torch.no_grad():
    logits, value = model(planes)

legal = logits[0, torch.as_tensor(indices.astype(np.int64))]
probs = torch.softmax(legal.float(), dim=0).numpy()  # over legal moves ONLY
print(board.san(moves[int(probs.argmax())]), float(value))
```

⚠️ **The one mistake to avoid**: reading `argmax` over the full 4096 logits. About 99 % of them are illegal in any given position. Always gather at `legal_move_indices` first.

`ChessNetwork.from_checkpoint` returns the model already in `eval()` mode. Keep it there — the tower is BatchNorm, and train-mode batch statistics will corrupt single-position inference.

---

## Reproduce the measurements

The tools that produced every number on this page are here. They are the
methodological part of the project, and the part worth reusing. Pass either
checkpoint's filename — the examples below use `RDTChess.pt` because that is
where these scripts were first built, but every one of them works identically
on `RDTChess2.pt`.

### The puzzle grid

```bash
curl -L -O https://database.lichess.org/lichess_db_puzzle.csv.zst   # 290 MB
pip install zstandard
python puzzles.py RDTChess2.pt
```

23,845 puzzles sampled **per cell of (solution length × rating band)**, because
the two are confounded: a mate-in-3 is rated higher than a mate-in-1, so
"accuracy falls with length" would otherwise just restate "hard puzzles are
hard".

Lines are **forced**: the network is queried at every move of the solution even
after it has already erred. Without that, accuracy at move *k* would only be
measured on puzzles whose first *k−1* moves were already found — a sample sorted
by construction, where "the network composes" cannot be told apart from "the
surviving puzzles were the easy ones".

The first run writes its sample to `logs/puzzle_sample.csv` and every later run
reuses it, so two models are compared on the same positions and the test is
paired.

### Against the minimax

```bash
python versus.py RDTChess2.pt --games 300 --depth 2
```

Judge on **depth 2**. Depth 1 saturates for a trained model: `RDTChess.pt`
scored 88.7 % against depth 1 and 59.5 % against depth 2, twenty-nine points
apart for the same model. It prints the standard error, because a two-point gap
on 300 games is not measurable.

### Model against model

```bash
python round_robin.py 300 RDTChess2.pt RDTChess.pt
```

Two greedy networks are two deterministic functions: from the initial position
they replay **the same game** forever. So 8 random legal plies are drawn before
handing control back, and each opening is played in **both colour directions** —
otherwise you compare two different samples and the first-move advantage
(measured at 52.8 % over 3600 games) mixes with the luck of the draw.

Sanity check that the pairing is symmetric: run a model against a copy of
itself and the score is exactly 50.0 %.

---

## Limits — read these before quoting any number

- **It imitates, it does not solve.** The objective is agreement with 1800+ humans. Where those humans are systematically wrong, so is the model, by construction. It is not trying to find the best move; it is trying to find the *likely* move.
- **No search, and it shows.** One forward pass. The puzzle breakdown in the RDTChess (original) section locates the cost precisely: 90 % on forced continuations, 32.8 % on quiet entries in hard positions — the same weakness this architecture inherits, since neither addition changes what the model is asked to do.
- **One seed, one run.** No variance control on the extension past 1.38M games (the 1.38M starting point itself was 3-seed confirmed).
- **Under-promotions cannot be represented.** The action space collapses from/to pairs and always decodes promotions to a queen. This costs well under 0.1 % of moves in normal play, but it makes a handful of knight-promotion puzzles unsolvable *by construction*.
- **No absolute rating.** Never played against a rated pool. Do not read a win-rate against minimax as an Elo.
- **The value head is weak supervision.** It regresses the final result of the game, which is a very noisy label for a position at move 12.
- **Distribution.** Trained on rated blitz/rapid from one month of one server, at one rating floor. Behaviour outside that distribution — correspondence, odds games, composed positions — is untested.
- **The released weights are the raw supervised checkpoint.** No self-play fine-tuning was applied to it.
- **The minimax comparison is noisy at n=300.** Read the round-robin (91 pairs, 27,300 games) as the more direct instrument; minimax swings by double digits between two independent draws.

---

## What is in this repository

| file | role |
|---|---|
| `network.py` | the architecture — loads either checkpoint, config comes from the file |
| `encoding.py` | board ↔ tensor, move ↔ index |
| `player.py` | the network as a player, `get_move(engine)` |
| `engine.py` | board wrapper, canonical encoding, hand-written evaluation |
| `algo_player.py` | alpha-beta minimax, the opponent used in evaluation |
| `gui.py`, `play.py` | the browser board |
| `uci.py` | UCI engine, for lichess-bot and chess GUIs |
| `predict.py` | one-position example |
| `puzzles.py` | the puzzle grid, sampled by cell and paired across models |
| `versus.py` | score against the minimax, with its standard error |
| `round_robin.py` | model against model, paired random openings |

**Inference only.** The training code is not part of this repository; the recipe above is complete enough to reproduce it.

---

<a id="rdtchess"></a>
## 📎 RDTChess (original)

Everything in this section is about `RDTChess.pt` specifically: the plain 128×8 tower, no recurrence, no gate, no SE — the original single-seed, 5.5M-game run. None of the numbers below apply to `RDTChess2.pt` above; its own comparison against this architecture is in the Evaluation section near the top of this page.

### Identity

| | |
|---|---|
| **Architecture** | residual tower, **128 channels × 8 blocks**, two heads |
| **Parameters** | 10.92 M |
| **Input** | 19 planes of 8×8, canonical frame |
| **Policy output** | 4096 raw logits (from-square × 64 + to-square) |
| **Value output** | one scalar in [−1, 1], `tanh` |
| **Search** | **none** — one forward pass per move |
| **Training data** | 5,497,103 Lichess games, both players ≥ 1800 Elo |
| **Positions seen** | 400,581,531, one epoch |
| **Training time** | 12 h 54 on a GTX 1080 Ti |

### How it was made

**Data.** The Lichess July 2026 standard-game dump. A game is kept when **both** players are rated ≥ 1800 (the filter takes `min(WhiteElo, BlackElo)`), which leaves 5,497,103 games in the 5 GB archive — all of them used. Every position of every kept game becomes one training sample, from the first ply: 400,581,531 positions, seen once.

**Objective.** Plain behaviour cloning. The policy is cross-entropy against the move the human actually played, computed over the legal moves only. The value head regresses the final game result, expressed from the side to move.

**Recipe.** AdamW, learning rate 1e-3, weight decay 1e-4, betas (0.9, 0.999), batch size 512, gradient clipping 1.0, one epoch, no learning-rate schedule.

That is the whole method. No self-play, no MCTS, no distillation from an engine, no opening book, no endgame tablebase.

### Evaluation

Three independent yardsticks, because they disagree — and the disagreement is itself a finding.

| yardstick | result | detail |
|---|---|---|
| **Top-1 on held-out human moves** | **51.7 %** | agreement with what the human actually played |
| **vs minimax depth 1** | **88.7 %** | 300 games · +249 =34 −17 |
| **vs minimax depth 2** | **59.5 %** | 300 games · +144 =69 −87 |
| **vs the same net trained on 2M games** | **55.1 %** | 600 games · +35 Elo · p = 0.0034 |

A fixed opponent measures *how well you exploit that opponent*, not general strength: the depth-1 and depth-2 figures move by 29 points for the same model. The head-to-head duel is the most informative of the three.

> 🚫 **No absolute Elo is claimed.** Every Elo figure here is a *difference* measured inside one family of models. The [Lichess bot](https://lichess.org/@/RDTChessBot) is where an absolute rating is being built, game by game — currently on `RDTChess2.pt`, not this architecture.

#### Data scaling

| games | net | decade | gain vs d1 | rate |
|---|---|---|---|---|
| 150 k → 1 M | 96×4 | 0.82 | +21.2 pts | 25.7 / decade |
| 1 M → 2 M | 96×4 | 0.30 | +6.3 pts | 20.9 / decade |
| **2 M → 5.5 M** | **128×8** | 0.44 | **+8.0 pts** | **18.2 / decade** |

The rate bends steadily — 25.7, 20.9, 18.2 — without collapsing. The +8.0 was **predicted at +9 before the run** from the two earlier points.

The last row is a wider net than the first two, so the three rates are **not** a single controlled curve: read the bend as a trend, not as a measurement.

### What it is good at, and where it breaks

Measured on **23,845 Lichess puzzles**, sampled by cell (solution length × rating band) so that puzzle difficulty is not confounded with solution length. One move per position, no search.

| | |
|---|---|
| first move of the solution | **57.8 %** |
| entire line | **38.3 %** |
| random baseline | 3.6 % (28.0 legal moves on average) |

#### The entry move is the hard one

The network is asked at **every** move of the line — even after it has already erred, so that no step is measured on a pre-filtered sample.

| length | accuracy, first move → last |
|---|---|
| 2 moves | 51.9 → **76.1** |
| 3 moves | 53.6 → 67.5 → **85.1** |
| 4 moves | 60.4 → 69.6 → 75.2 → **88.0** |
| 5 moves | 65.5 → 72.5 → 75.5 → 80.3 → **90.6** |

**The entry is the hardest move of every line, by 20 to 35 points. The last is around 90 %, whatever the depth.** Once inside a forcing sequence — recaptures, checks, the mate — the network barely errs. Starting one is the problem.

#### And it is the *quiet* entry that breaks

First move found, split by what kind of move it is.

| entry move | <1200 | 1200-1599 | 1600-1999 | 2000-2399 | 2400+ |
|---|---|---|---|---|---|
| check | 74.4 | 59.5 | 47.5 | 39.5 | 51.4 |
| capture | 91.6 | 73.8 | 60.7 | 57.7 | **56.1** |
| quiet | **94.6** | 75.6 | 54.7 | 39.3 | **32.8** |

The quiet move goes from the **easiest** category to the **hardest** — 23 points below captures in the same band — as difficulty rises. Captures and checks are visible in the position; a quiet preparatory move is justified only by what comes three plies later.

> 💡 **In one sentence: this network executes tactics, it does not initiate them.**

---

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{charlet2026rdtchess,
  author = {Théo CHARLET},
  title  = {RDTChess: 128x8 chess policy and value networks trained on human Lichess games},
  year   = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/RDTvlokip/RDTChess}}
}
```

---

**Théo CHARLET**

<p>
TSSR Graduate (IT Systems &amp; Networks Technician) — AI/ML Specialization<br />
Creator of AG-BPE (Attention-Guided Byte-Pair Encoding)
</p>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0a66c2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/théo-charlet)
[![Website](https://img.shields.io/badge/Website-1f2937?style=flat-square&logo=googlechrome&logoColor=white)](https://rdtvlokip.fr)
[![Search](https://img.shields.io/badge/RDTvlokip%20Search-4C6EF5?style=flat-square&logo=duckduckgo&logoColor=white)](https://search.rdtvlokip.fr)

🚀 Seeking internship opportunities
