"""A board you can click, served to your browser.

The model stays in Python -- it is a PyTorch network and belongs where torch
is -- and the page talks to it over a tiny JSON API on localhost. That keeps
the whole thing dependency-free: `http.server` is standard library, and the
page needs no framework, no CDN and no build step.
"""

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import chess

from engine import ChessEngine

# The filled glyphs for both sides, with the colour done in CSS. Using the
# outline set for White and the filled set for Black is the obvious idea and it
# does not work: the two render nearly identically once the page picks a single
# text colour, so on a dark board every piece looked white.
PIECES = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}


class Session:
    """One game, guarded by a lock so a double click cannot race an engine.

    Each colour is either an engine -- anything with ``get_move(engine)``, so
    the network and the minimax are interchangeable -- or None for the human.
    Two engines and nobody human is the spectator mode: the game plays itself
    one ply per `/api/step`, which lets the browser pace and pause it.
    """

    def __init__(self, white, black, max_moves: int, labels=("white", "black")):
        self.players = {chess.WHITE: white, chess.BLACK: black}
        self.labels = {chess.WHITE: labels[0], chess.BLACK: labels[1]}
        self.max_moves = max_moves
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.engine = ChessEngine()
        self.history = []
        self._auto_play()

    @property
    def watching(self) -> bool:
        return all(player is not None for player in self.players.values())

    @property
    def human_color(self) -> Optional[bool]:
        for color, player in self.players.items():
            if player is None:
                return color
        return None

    @property
    def human_is_white(self) -> bool:
        return self.human_color != chess.BLACK

    def _finished(self) -> bool:
        return self.engine.is_game_over() or self.engine.get_move_count() >= self.max_moves

    def step(self) -> bool:
        """Play one engine ply. False when it is a human's turn or the game ended."""
        if self._finished():
            return False
        player = self.players[self.engine.get_turn()]
        if player is None:
            return False
        move = player.get_move(self.engine)
        if move is None:
            return False
        self.history.append(self.engine.board.san(move))
        self.engine.make_move(move)
        return True

    def _auto_play(self) -> None:
        """Let the engines move until a human is on turn.

        Bounded to one ply in spectator mode so the browser sees every move
        rather than a finished game.
        """
        if self.watching:
            return
        while self.step():
            pass

    def play(self, uci: str) -> Optional[str]:
        """Apply the human move, then the engine's reply. Returns an error, if any."""
        if self._finished() or self.watching:
            return "the game is over"
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return "unreadable move"
        # Promotion is not asked for in the UI; anything else than a queen is
        # rare enough that offering it would cost more clicks than it saves.
        if move not in self.engine.board.legal_moves:
            promoted = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if promoted in self.engine.board.legal_moves:
                move = promoted
            else:
                return "illegal move"

        self.history.append(self.engine.board.san(move))
        self.engine.make_move(move)
        self._auto_play()
        return None

    def state(self) -> dict:
        board = self.engine.board
        squares = []
        for rank in range(7, -1, -1):
            for file in range(8):
                piece = board.piece_at(chess.square(file, rank))
                if piece is None:
                    squares.append(None)
                else:
                    squares.append(
                        {
                            "glyph": PIECES[piece.symbol().lower()],
                            "side": "w" if piece.color == chess.WHITE else "b",
                        }
                    )

        legal = {}
        if not self._finished() and board.turn == self.human_color:
            for move in board.legal_moves:
                legal.setdefault(chess.square_name(move.from_square), []).append(
                    chess.square_name(move.to_square)
                )

        if board.is_checkmate():
            winner = self.labels[not board.turn]
            status = (
                f"Checkmate — {winner} wins"
                if self.watching
                else "Checkmate — " + ("you win" if board.turn != self.human_color else "you lose")
            )
        elif board.is_stalemate():
            status = "Stalemate — draw"
        elif board.is_insufficient_material():
            status = "Insufficient material — draw"
        elif board.is_repetition(3):
            status = "Threefold repetition — draw"
        elif board.halfmove_clock >= 100:
            status = "Fifty-move rule — draw"
        elif self.engine.get_move_count() >= self.max_moves:
            status = "Move limit reached"
        elif board.is_check():
            status = "Check!"
        elif self.watching:
            status = f"{self.labels[board.turn]} to move"
        else:
            status = "Your move" if board.turn == self.human_color else "Thinking…"

        # A piece with no legal move is the commonest "the board is broken"
        # moment: the click does nothing and nothing says why. Ship the reason.
        check_square = None
        if board.is_check():
            king = board.king(board.turn)
            check_square = chess.square_name(king) if king is not None else None
        pinned = [
            chess.square_name(square)
            for square in chess.SQUARES
            if (piece := board.piece_at(square)) is not None
            and piece.color == self.human_color
            and board.is_pinned(self.human_color, square)
        ]

        return {
            "squares": squares,
            "legal": legal,
            "check": check_square,
            "pinned": pinned,
            "yourTurn": board.turn == self.human_color and not self._finished(),
            "watching": self.watching,
            "whiteToMove": board.turn == chess.WHITE,
            "white": self.labels[chess.WHITE],
            "black": self.labels[chess.BLACK],
            "last": [
                chess.square_name(board.peek().from_square),
                chess.square_name(board.peek().to_square),
            ] if board.move_stack else None,
            "status": status,
            "finished": self._finished(),
            "flipped": not self.human_is_white,
            # The whole game, not a tail: truncating it while numbering from 1
            # made every excerpt look like a game that opened with a king move.
            "history": self.history,
            "plies": self.engine.get_move_count(),
            "evaluation": round(self.engine.evaluate() / 100.0, 2),
        }


PAGE = """<!doctype html>
<meta charset="utf-8"><title>chess-rl</title>
<style>
  :root { color-scheme: light dark; --light:#ebecd0; --dark:#779556; --pick:#f6f669; --hint:#00000030; }
  body { margin:0; padding:1.5rem; box-sizing:border-box; min-height:100vh; display:flex;
         align-items:center; justify-content:center; gap:2rem; flex-wrap:wrap;
         font:16px/1.5 system-ui,sans-serif; background:#312e2b; color:#eee; }
  #board { display:grid; grid-template-columns:repeat(8,min(9vw,72px)); border-radius:6px;
           overflow:hidden; box-shadow:0 12px 40px #0008; }
  .sq { aspect-ratio:1; display:flex; align-items:center; justify-content:center;
        font-size:min(7vw,56px); cursor:pointer; user-select:none; position:relative; line-height:1; }
  .sq.light { background:var(--light); } .sq.dark { background:var(--dark); }
  /* Both sides use the same filled glyph; only the colour tells them apart.
     The halo is drawn with text-shadow rather than -webkit-text-stroke: a
     stroke sits astride the outline and eats into the fill, and at this glyph
     size a 2px light stroke covered more of a black piece than its own body
     did, so the two sides looked swapped. A shadow paints behind the glyph. */
  .w { color:#fff; text-shadow:0 0 3px #000c, 0 1px 2px #0008; }
  .b { color:#141414; text-shadow:0 0 3px #fffc, 0 1px 2px #0006; }
  .sq.sel { background:var(--pick) !important; }
  .sq.last { box-shadow:inset 0 0 0 4px #f6f66955; }
  .sq.check { background:radial-gradient(circle, #e74c3c 30%, #e74c3c60 70%, transparent 75%) !important; }
  .sq.tgt::after { content:""; position:absolute; width:30%; height:30%; border-radius:50%;
                   background:var(--hint); }
  #why { min-height:1.4em; font-size:.9rem; color:#e8c07d; margin-bottom:.5rem; }
  aside { width:min(90vw,260px); }
  h1 { font-size:1.1rem; margin:0 0 .5rem; letter-spacing:.04em; text-transform:uppercase; opacity:.7; }
  #status { font-size:1.3rem; font-weight:600; margin-bottom:.75rem; min-height:1.6em; }
  #meta { opacity:.6; font-size:.85rem; margin-bottom:1rem; }
  #moves { max-height:40vh; overflow-y:auto; font-family:ui-monospace,monospace; font-size:.85rem;
           opacity:.8; background:#0003; padding:.6rem; border-radius:6px; }
  button { font:inherit; padding:.5rem 1rem; border:0; border-radius:6px; cursor:pointer;
           background:#779556; color:#fff; margin-bottom:1rem; }
  button:hover { filter:brightness(1.1); }
  #modes { display:flex; flex-direction:column; gap:.4rem; margin-bottom:1rem; }
  #modes button { margin:0; background:#4a4744; text-align:left; }
  #modes button.on { background:#779556; font-weight:600; }
  #depthRow { display:block; margin-bottom:1rem; font-size:.9rem; opacity:.85; }
  #depth { width:100%; accent-color:#779556; margin-top:.3rem; }
  #depthHint { display:block; font-size:.8rem; opacity:.6; }
  #players { margin-bottom:1rem; font-size:.9rem; }
  #players div { display:flex; align-items:center; gap:.5rem; padding:.15rem 0; }
  #players i { width:.9rem; height:.9rem; border-radius:50%; border:1px solid #0006; flex:none; }
  #players i.w { background:#fff; } #players i.b { background:#141414; border-color:#fff6; }
  #players div.turn { font-weight:600; }
  #players div.turn::after { content:"← to move"; opacity:.55; font-weight:400; font-size:.8rem; }
</style>
<div id="board"></div>
<aside>
  <h1 id="title">chess-rl</h1>
  <div id="status">…</div>
  <div id="why"></div>
  <div id="meta"></div>
  <div id="modes">
    <button data-mode="model">You vs model</button>
    <button data-mode="algo">You vs minimax</button>
    <button data-mode="watch">▶ Model vs minimax</button>
  </div>
  <label id="depthRow">minimax depth <b id="depthValue">2</b>
    <input id="depth" type="range" min="1" max="5" value="2">
    <span id="depthHint"></span>
  </label>
  <div id="players"></div>
  <button onclick="newGame()">New game</button>
  <button id="pause" onclick="togglePause()" hidden>Pause</button>
  <div id="moves"></div>
</aside>
<script>
const board = document.getElementById('board');
let picked = null, state = null, paused = false;

function squareName(i, flipped) {
  const file = flipped ? 7 - (i % 8) : i % 8;
  const rank = flipped ? i >> 3 : 7 - (i >> 3);
  return 'abcdefgh'[file] + (rank + 1);
}

function render() {
  board.innerHTML = '';
  const order = state.flipped ? [...state.squares].reverse() : state.squares;
  order.forEach((piece, i) => {
    const name = squareName(i, state.flipped);
    const div = document.createElement('div');
    const dark = ((i >> 3) + (i % 8)) % 2 === 1;
    div.className = 'sq ' + (dark ? 'dark' : 'light');
    if (piece) {
      div.textContent = piece.glyph;
      div.classList.add(piece.side);
    }
    if (name === picked) div.classList.add('sel');
    if (state.last && state.last.includes(name)) div.classList.add('last');
    if (state.check === name) div.classList.add('check');
    if (picked && (state.legal[picked] || []).includes(name)) div.classList.add('tgt');
    div.onclick = () => click(name);
    board.appendChild(div);
  });
  document.getElementById('status').textContent = state.status;
  document.getElementById('title').textContent = state.watching
    ? state.white + ' — ' + state.black
    : 'chess-rl — vs ' + (state.flipped ? state.white : state.black);
  // Who is which colour: the one thing a spectator cannot guess.
  document.getElementById('players').innerHTML =
    [['w', 'White', state.white], ['b', 'Black', state.black]]
      .map(([side, name, who]) => {
        const turn = state.whiteToMove === (side === 'w') && !state.finished;
        return `<div class="${turn ? 'turn' : ''}"><i class="${side}"></i>` +
               `<span>${name} — ${who}</span></div>`;
      }).join('');

  const pause = document.getElementById('pause');
  pause.hidden = !state.watching || state.finished;
  pause.textContent = !paused ? 'Pause' : (state.plies ? 'Resume' : '▶ Start');
  document.querySelectorAll('#modes button').forEach(b => {
    b.classList.toggle('on', b.dataset.mode === state.mode);
  });
  document.getElementById('depthRow').hidden = state.mode === 'model';
  const ev = state.evaluation > 0 ? '+' + state.evaluation : state.evaluation;
  document.getElementById('meta').textContent = `ply ${state.plies} · material ${ev}`;
  const moves = document.getElementById('moves');
  moves.textContent =
    state.history.map((m, i) => (i % 2 ? m + '\\n' : (i / 2 + 1) + '. ' + m + '  ')).join('');
  moves.scrollTop = moves.scrollHeight;
}

async function click(name) {
  if (state.finished) return;
  if (picked && (state.legal[picked] || []).includes(name)) {
    const from = picked; picked = null;
    state.status = 'Thinking…'; render();
    await send('/api/move', {move: from + name});
  } else {
    picked = state.legal[name] ? name : null;
    explain(name);
    render();
  }
}

// Silence is the worst answer to "why won't this piece move?".
function explain(name) {
  const why = document.getElementById('why');
  if (state.legal[name]) { why.textContent = ''; return; }

  // state.squares is always a8-first, whichever way the board is drawn.
  const file = name.charCodeAt(0) - 97, rank = +name[1] - 1;
  const piece = state.squares[(7 - rank) * 8 + file];
  const mine = piece && piece.side === (state.flipped ? 'b' : 'w');

  if (!mine) why.textContent = '';
  else if (!state.yourTurn) why.textContent = 'Not your turn yet.';
  else if (state.check) why.textContent = 'You are in check — only moves that answer it are legal.';
  else if (state.pinned.includes(name)) why.textContent = 'That piece is pinned to your king.';
  else why.textContent = 'That piece has nowhere legal to go.';
}

async function send(url, body) {
  const r = await fetch(url, {method: 'POST', body: JSON.stringify(body || {})});
  state = await r.json(); picked = null; render();
}
async function newGame() {
  await send('/api/new');
  paused = state.watching;   // a fresh spectator game waits for Start too
  render();
  watch();
}
function togglePause() { paused = !paused; render(); if (!paused) watch(); }

const depth = document.getElementById('depth');
// Each extra ply multiplies the tree; past 4 a move takes visible seconds.
const HINTS = {1: 'instant, no lookahead past captures', 2: 'fast', 3: 'a moment per move',
               4: 'slow — a few seconds per move', 5: 'very slow'};
function showDepth() {
  document.getElementById('depthValue').textContent = depth.value;
  document.getElementById('depthHint').textContent = HINTS[depth.value] || '';
}
depth.oninput = showDepth;
depth.onchange = () => setMode(state.mode);

async function setMode(mode) {
  // Spectator mode waits for Start: switching to it should give you time to
  // set the depth, not launch a game the instant you click.
  paused = mode === 'watch';
  await send('/api/mode', {mode: mode, depth: +depth.value});
  watch();
}
document.querySelectorAll('#modes button').forEach(b => {
  b.onclick = () => setMode(b.dataset.mode);
});

// Spectator mode: ask for one ply at a time so the board animates instead of
// jumping to a finished game, and so Pause can actually stop it.
let watching = false;
async function watch() {
  if (watching) return;
  watching = true;
  try {
    while (state.watching && !state.finished && !paused) {
      const before = state.plies;
      await send('/api/step');
      if (state.plies === before) break;   // nothing moved: don't spin
      await new Promise(r => setTimeout(r, 450));
    }
  } finally { watching = false; }
}

(async () => {
  state = await (await fetch('/api/state')).json();
  depth.value = state.depth;
  showDepth();
  paused = state.watching;
  render();
  watch();
})();
</script>
"""


def serve(model, make_algo, model_label: str, human_is_white: bool = True,
          max_moves: int = 300, port: int = 8000, depth: int = 2) -> int:
    """Serve the board, with mode and minimax depth switchable from the page.

    `make_algo(depth)` builds a fresh minimax, so the slider can change the
    search depth without restarting the server.
    """

    def build(mode: str, depth: int) -> Session:
        algo_label = f"minimax d{depth}"
        if mode == "watch":
            # The model takes White, so the game opens on its own choice.
            return Session(model, make_algo(depth), max_moves, (model_label, algo_label))
        rival, label = (
            (make_algo(depth), algo_label) if mode == "algo" else (model, model_label)
        )
        if human_is_white:
            return Session(None, rival, max_moves, ("you", label))
        return Session(rival, None, max_moves, (label, "you"))

    state = {"session": build("model", depth), "mode": "model", "depth": depth}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # the board is the interface; request logs only get in the way

        def _send(self, payload: bytes, kind: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _payload(self) -> bytes:
            session = state["session"]
            with session.lock:
                snapshot = session.state()
            snapshot["mode"] = state["mode"]
            snapshot["depth"] = state["depth"]
            return json.dumps(snapshot).encode()

        def do_GET(self):
            if self.path.startswith("/api/state"):
                self._send(self._payload(), "application/json")
            else:
                self._send(PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")

            if self.path.startswith("/api/mode"):
                mode = body.get("mode", "model")
                requested = int(body.get("depth", state["depth"]))
                if mode in ("model", "algo", "watch"):
                    state["mode"] = mode
                    state["depth"] = max(1, min(5, requested))
                    state["session"] = build(mode, state["depth"])
            else:
                session = state["session"]
                with session.lock:
                    if self.path.startswith("/api/new"):
                        session.reset()
                    elif self.path.startswith("/api/move"):
                        session.play(body.get("move", ""))
                    elif self.path.startswith("/api/step"):
                        session.step()

            self._send(self._payload(), "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Board at {url}  (Ctrl-C to stop)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        server.server_close()
    return 0
