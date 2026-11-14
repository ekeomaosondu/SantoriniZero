"""Santorini agent: the trained network + MCTS + exact solver, a game referee, and
the API behind the play widget (play.html).

    python santorini_agent.py --ckpt latest.pkl [--port 8000]     # local web app
    # in Colab / Jupyter:
    from santorini_agent import Net, Session, show_colab
    show_colab(Session(Net("latest.pkl")))

Cells are 0..24 = row*5 + col, written A1..E5 (column letter, row number, row 1 on top).
Player 0 ("Blue") places first and moves first; player 1 is "Gold".
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

import santorini as S

NAMES = ["Blue", "Gold"]
DR = [-1, -1, -1, 0, 0, 1, 1, 1]
DC = [-1, 0, 1, -1, 1, -1, 0, 1]


def cell_name(c):
    return "ABCDE"[c % 5] + str(c // 5 + 1)


def move_text(f, t, b):
    return f"{cell_name(f)}–{cell_name(t)} wins" if b < 0 else f"{cell_name(f)}–{cell_name(t)} build {cell_name(b)}"


# --------------------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------------------
class Net:
    """The trained policy/value network. `ckpt` is the training notebook's latest.pkl,
    a weights_XXXX.msgpack / phase0_state.msgpack file, or None for a uniform network
    (zero logits, zero value: plain MCTS + solver, useful for testing)."""

    def __init__(self, ckpt=None):
        self.info = {"source": "uniform (no network)", "iteration": None, "params": 0}
        self.variables = None
        if ckpt is None:
            return
        import jax
        import jax.numpy as jnp
        import flax.linen as nn
        from flax import serialization

        raw_bytes, meta = None, {}
        if ckpt.endswith(".pkl"):
            ck = pickle.load(open(ckpt, "rb"))
            raw_bytes = ck["state"]
            meta = {"iteration": ck.get("iter"), "trained_on": ck.get("backend")}
            hist = [h for h in ck.get("history", []) if h.get("phase") == 1]
            if hist:
                meta["test_won_top1_winning"] = round(hist[-1].get("won_top1_winning", float("nan")), 3)
        else:
            raw_bytes = open(ckpt, "rb").read()
        raw = serialization.msgpack_restore(raw_bytes)
        params, stats = raw["params"], raw["batch_stats"]
        blocks = sum(1 for k in params if k.startswith("ResBlock_"))
        ch = int(np.asarray(params["Conv_0"]["kernel"]).shape[-1])

        class ResBlock(nn.Module):
            ch: int
            @nn.compact
            def __call__(self, x):
                y = nn.Conv(self.ch, (3, 3), use_bias=False)(x)
                y = nn.relu(nn.BatchNorm(use_running_average=True, momentum=0.9)(y))
                y = nn.Conv(self.ch, (3, 3), use_bias=False)(y)
                y = nn.BatchNorm(use_running_average=True, momentum=0.9)(y)
                return nn.relu(x + y)

        class SantoriniNet(nn.Module):
            blocks: int
            ch: int
            @nn.compact
            def __call__(self, x):
                bn = lambda: nn.BatchNorm(use_running_average=True, momentum=0.9)
                x = nn.relu(bn()(nn.Conv(self.ch, (3, 3), use_bias=False)(x)))
                for _ in range(self.blocks):
                    x = ResBlock(self.ch)(x)
                p = nn.relu(bn()(nn.Conv(32, (1, 1), use_bias=False)(x)))
                p = nn.Dense(S.NACT)(p.reshape(p.shape[0], -1))
                v = nn.relu(bn()(nn.Conv(16, (1, 1), use_bias=False)(x)))
                v = nn.relu(nn.Dense(128)(v.reshape(v.shape[0], -1)))
                return p, jnp.tanh(nn.Dense(1)(v))[:, 0]

        model = SantoriniNet(blocks, ch)
        self.variables = jax.device_put({"params": params, "batch_stats": stats})
        self._fwd = jax.jit(lambda var, x: model.apply(var, x))
        n_params = sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params))
        self.info = {"source": os.path.basename(ckpt), "blocks": blocks, "channels": ch,
                     "params": int(n_params), "device": jax.devices()[0].device_kind, **meta}

    def __call__(self, planes, n, pad_to=None):
        """Evaluate the first n rows of planes [*, 5, 5, 15] -> (logits [n, 153], values [n])."""
        if self.variables is None:
            return np.zeros((n, S.NACT), np.float32), np.zeros(n, np.float32)
        size = pad_to or n
        x = planes[:size] if len(planes) >= size else np.concatenate(
            [planes, np.zeros((size - len(planes),) + planes.shape[1:], planes.dtype)])
        if n < size:
            x = x.copy(); x[n:] = 0
        lg, v = self._fwd(self.variables, x)
        return np.asarray(lg, np.float32)[:n], np.asarray(v, np.float32)[:n]


# --------------------------------------------------------------------------------------
# Agent: search for one position at a time
# --------------------------------------------------------------------------------------
class Agent:
    def __init__(self, net, batch=16, cpuct=1.5, solver=True, root_budget=200_000, child_budget=2_000):
        self.net, self.batch = net, batch
        self.params = S.SearchParams()
        self.params.cpuct = cpuct
        self.set_solver(solver, root_budget, child_budget)
        self.buf = np.zeros((batch, 5, 5, S.NPLANES), np.float32)
        self.search, self.key, self.started, self.solve_seconds, self.think_seconds = None, None, 0, 0.0, 0.0

    def set_solver(self, on, root_budget=200_000, child_budget=2_000):
        self.params.root_budget = root_budget if on else 0
        self.params.child_budget = child_budget

    def begin(self, state, key):
        """Start (or keep) the search tree for this position."""
        if self.search is not None and self.key == key:
            return
        t = time.time()
        self.search = S.Search(np.asarray(state, np.int8), self.params)
        self.solve_seconds = time.time() - t
        self.key, self.think_seconds = key, 0.0

    def think(self, sims):
        """Run up to `sims` more simulations (stops early if the root is decided)."""
        s, t = self.search, time.time()
        target = s.simulations + sims
        while s.simulations < target and not s.done:
            n = s.prepare(self.buf, self.batch)
            if n == 0:
                if s.done:
                    break
                continue
            lg, v = self.net(self.buf, n, pad_to=self.batch)
            s.apply(np.ascontiguousarray(lg), np.ascontiguousarray(v))
        self.think_seconds += time.time() - t

    def snapshot(self, top=10, pv_depth=8):
        s = self.search
        moves = s.moves(pv_depth)
        total = max(1, sum(m["visits"] for m in moves))
        out = []
        for m in moves[:top]:
            out.append({"from": m["from"], "to": m["to"], "build": m["build"], "text": move_text(m["from"], m["to"], m["build"]),
                        "visits": m["visits"], "share": m["visits"] / total, "q": m["q"], "prior": m["prior"],
                        "proof": m["proof"], "step_up": m["step_up"],
                        "pv": [list(x) for x in m["pv"]], "pv_ends_in_win": m["pv_ends_in_win"]})
        best = s.best()
        return {"simulations": s.simulations, "status": s.status, "solver": s.solver_result,
                "net_value": s.net_value, "search_value": s.search_value if s.simulations else s.net_value,
                "done": s.done, "tree_size": s.tree_size, "n_moves": len(moves),
                "solve_seconds": round(self.solve_seconds, 2), "think_seconds": round(self.think_seconds, 2),
                "moves": out,
                "best": {"from": best["from"], "to": best["to"], "build": best["build"],
                         "text": move_text(best["from"], best["to"], best["build"])} if best["action"] >= 0 else None}


# --------------------------------------------------------------------------------------
# Opening placement (the network was trained on random placements, so the agent picks
# placements by evaluating the resulting positions with the value head).
# --------------------------------------------------------------------------------------
def _sym_cells():
    # D4 symmetries of the 5x5 board (same convention as the C++ tables).
    out = []
    for s in range(8):
        m = []
        for c in range(25):
            x, y = c % 5, c // 5
            for _ in range(s & 3):
                x, y = 4 - y, x
            if s >= 4:
                x = 4 - x
            m.append(y * 5 + x)
        out.append(m)
    return out


SYM = _sym_cells()


class Placement:
    """Incremental placement search so the widget can show progress.
    first=True : the agent places first (maximise over its pair, minimise over the reply).
    first=False: the agent places second against fixed opponent workers."""

    def __init__(self, net, first, opp=None, rng=None):
        self.net, self.first, self.opp = net, first, opp or []
        self.rng = rng or np.random.default_rng()
        free = [c for c in range(25) if c not in self.opp]
        pairs = [(a, b) for i, a in enumerate(free) for b in free[i + 1:]]
        if first:  # one representative per symmetry class
            seen, reps = set(), []
            for p in pairs:
                key = min(tuple(sorted((SYM[s][p[0]], SYM[s][p[1]]))) for s in range(8))
                if key not in seen:
                    seen.add(key); reps.append(p)
            self.cands = reps
        else:
            self.cands = pairs
        self.scores = {}       # pair -> score for the agent (-1..1)
        self.i = 0

    def _eval(self, me_pairs, op_pairs):
        """Values for the side to move (me) in the positions right after placement."""
        st = np.zeros((len(me_pairs), 29), np.int8)
        st[:, 25:27] = np.sort(np.array(me_pairs), 1)
        st[:, 27:29] = np.sort(np.array(op_pairs), 1)
        x, _, _ = S.encode(st, np.zeros(len(st), np.int32))
        _, v = self.net(x, len(st), pad_to=256 * ((len(st) + 255) // 256))
        return v

    def step(self, budget=2048):
        """Evaluate about `budget` positions; returns True when finished."""
        done = 0
        while self.i < len(self.cands) and done < budget:
            if self.first:
                A = self.cands[self.i]
                free = [c for c in range(25) if c not in A]
                B = [(a, b) for i, a in enumerate(free) for b in free[i + 1:]]
                v = self._eval([A] * len(B), B)      # we (first player) move first
                self.scores[A] = float(v.min())      # the opponent picks the reply worst for us
                self.i += 1
                done += len(B)
            else:
                chunk = self.cands[self.i:self.i + 512]
                v = self._eval([tuple(self.opp)] * len(chunk), chunk)   # the opponent moves first
                for p, val in zip(chunk, v):
                    self.scores[p] = -float(val)
                self.i += len(chunk)
                done += len(chunk)
        return self.done

    @property
    def done(self):
        return self.i >= len(self.cands)

    def all_scores(self):
        """Scores for every concrete pair (symmetry classes expanded)."""
        if not self.first:
            return dict(self.scores)
        out = {}
        for p, v in self.scores.items():
            for s in range(8):
                out[tuple(sorted((SYM[s][p[0]], SYM[s][p[1]])))] = v
        return out

    def view(self, top=5):
        sc = self.all_scores()
        heat = [None] * 25
        for (a, b), v in sc.items():
            for c in (a, b):
                heat[c] = v if heat[c] is None else max(heat[c], v)
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])[:top]
        return {"progress": self.i / max(1, len(self.cands)), "candidates": len(self.cands),
                "heat": heat, "top": [{"cells": list(p), "score": v} for p, v in ranked]}

    def choose(self, tolerance=0.02):
        sc = self.all_scores()
        best = max(sc.values())
        options = [p for p, v in sc.items() if v >= best - tolerance]
        return list(options[self.rng.integers(len(options))]), best


# --------------------------------------------------------------------------------------
# Game referee
# --------------------------------------------------------------------------------------
class Game:
    """Board, turn order, legality and history. frames[k] is the position after the
    first k log entries, so the timeline can show any earlier position."""

    def __init__(self, human=0):
        self.human, self.agent = human, 1 - human
        self.heights = [0] * 25
        self.workers = [[], []]
        self.phase, self.turn = "place", 0
        self.winner, self.reason = None, ""
        self.log, self.evals = [], []
        self.frames = [self._frame()]

    def _frame(self):
        return {"heights": list(self.heights), "workers": [list(self.workers[0]), list(self.workers[1])],
                "turn": self.turn, "phase": self.phase}

    def state(self, player=None):
        """29-byte state from `player`'s perspective (default: side to move)."""
        p = self.turn if player is None else player
        s = np.zeros(29, np.int8)
        s[:25] = self.heights
        s[25:27] = sorted(self.workers[p]); s[27:29] = sorted(self.workers[1 - p])
        return s

    def key(self):
        return (tuple(self.heights), tuple(sorted(self.workers[0])), tuple(sorted(self.workers[1])), self.turn)

    def legal(self):
        if self.phase != "play":
            return []
        return [tuple(int(v) for v in m) for m in S.all_moves(self.state())]

    def place(self, cell):
        assert self.phase == "place", "not in the placement phase"
        assert 0 <= cell < 25 and cell not in self.workers[0] + self.workers[1], "that cell is taken"
        self.workers[self.turn].append(int(cell))
        self.log.append({"player": self.turn, "kind": "place", "cell": int(cell),
                         "text": f"places on {cell_name(cell)}"})
        if len(self.workers[self.turn]) == 2:
            if self.turn == 0:
                self.turn = 1
            else:
                self.phase, self.turn = "play", 0
                self._check_stuck()
        self.frames.append(self._frame())

    def move(self, f, t, b):
        assert self.phase == "play", "the game is not in the move phase"
        if (f, t, b) not in set(self.legal()):
            raise ValueError(f"illegal move {move_text(f, t, b)}")
        p = self.turn
        self.workers[p][self.workers[p].index(f)] = t
        self.log.append({"player": p, "kind": "move", "from": f, "to": t, "build": b, "text": move_text(f, t, b)})
        if b < 0:
            self.phase, self.winner = "over", p
            self.reason = f"{NAMES[p]} stepped up to level 3 on {cell_name(t)}"
        else:
            self.heights[b] += 1
            self.turn = 1 - p
            self._check_stuck()
        self.frames.append(self._frame())

    def _check_stuck(self):
        if self.phase == "play" and not self.legal():
            self.phase, self.winner = "over", 1 - self.turn
            self.reason = f"{NAMES[self.turn]} has no legal move"

    def rewind(self, k):
        """Go back to the position after the first k log entries."""
        assert 0 <= k < len(self.frames), "no such position"
        f = self.frames[k]
        self.heights, self.workers = list(f["heights"]), [list(f["workers"][0]), list(f["workers"][1])]
        self.turn, self.phase = f["turn"], f["phase"]
        self.winner, self.reason = None, ""
        self.log, self.frames = self.log[:k], self.frames[:k + 1]
        self.evals = [e for e in self.evals if e["entry"] < k]

    def to_json(self):
        return {"phase": self.phase, "turn": self.turn, "human": self.human, "agent": self.agent,
                "winner": self.winner, "reason": self.reason, "heights": self.heights,
                "workers": self.workers, "log": self.log, "evals": self.evals, "frames": self.frames,
                "legal": [list(m) for m in self.legal()] if self.turn == self.human else [],
                "names": NAMES}


# --------------------------------------------------------------------------------------
# Session: the API used by the widget
# --------------------------------------------------------------------------------------
def _win(q):
    return None if q is None else (float(q) + 1) / 2


class Session:
    DEFAULTS = {"sims": 400, "solver": True}
    TREE_DEFAULTS = {"root": 0, "max_nodes": 120, "prune": 100, "depth": 6, "unvisited": False}

    def __init__(self, net, batch=None, seed=None, keep_thoughts=200):
        self.net = net
        self.settings = dict(self.DEFAULTS)
        if batch is None:
            batch = 16 if net.variables is not None else 1
        self.agent = Agent(net, batch=batch)
        self.rng = np.random.default_rng(seed)
        self.keep_thoughts = keep_thoughts
        self._reset(Game(human=0))

    def _reset(self, game):
        self.game, self.placement, self.last_thought = game, None, None
        self.thoughts = {}       # id -> {"search", "snapshot", "player", "entry", "kind"}
        self.next_id = 0
        self.live = None         # id of the search on the current position
        self.agent.search = None
        self.step_planes = np.zeros((1, 5, 5, S.NPLANES), np.float32)
        self.step_info = None

    # -- the live search -------------------------------------------------------------------
    def _live_search(self):
        g = self.game
        self.agent.begin(g.state(), g.key())
        if self.live is None or self.thoughts.get(self.live, {}).get("search") is not self.agent.search:
            self.live = self.next_id
            self.next_id += 1
            self.thoughts[self.live] = {"search": self.agent.search, "snapshot": None, "player": g.turn,
                                        "entry": None, "kind": "live"}
            # forget old searches
            old = sorted(k for k, t in self.thoughts.items() if t["entry"] is not None)
            for k in old[:-self.keep_thoughts]:
                del self.thoughts[k]
        return self.agent.search

    def _thought_json(self):
        g = self.game
        if g.phase != "play" or self.agent.search is None:
            return None
        snap = self.agent.snapshot()
        snap["player"] = g.turn
        snap["target"] = self.settings["sims"]
        snap["id"] = self.live
        return snap

    def _attach_thought(self, entry, kind):
        """Keep the live search as the thinking behind log entry `entry`."""
        t = self.thoughts.get(self.live)
        if t is None:
            return None
        snap = self._thought_json()
        if snap is None or snap["simulations"] == 0 and snap["status"] == 0:
            return None
        t.update({"snapshot": snap, "entry": entry, "kind": kind})
        self._record_eval(entry, snap, kind)
        return snap

    def _record_eval(self, entry, snap, kind):
        q = float(snap["status"]) if snap["status"] != 0 else snap["search_value"]
        blue = (q + 1) / 2 if snap["player"] == 0 else (1 - q) / 2
        self.game.evals = [e for e in self.game.evals if e["entry"] != entry] + [
            {"entry": entry, "blue": blue, "proven": snap["status"] != 0, "by": kind}]

    def _finish_step(self):
        s = self.agent.search
        if s is None:
            return
        if s.step_phase == 1:
            lg, v = self.net(self.step_planes, 1, pad_to=self.agent.batch)
            s.step_expand(np.ascontiguousarray(lg[0]), float(v[0]))
        elif s.step_phase == 2:
            s.step_expand_known()
        if s.step_phase == 3:
            s.step_backup()

    # -- tree ------------------------------------------------------------------------------
    def _tree_json(self, tid, root=0, max_nodes=120, prune=100, depth=6, unvisited=False):
        t = self.thoughts.get(tid)
        if t is None:
            return None
        search, p0 = t["search"], t["player"]
        items = search.tree(int(root), int(max_nodes), float(prune), int(depth), bool(unvisited))
        nodes = []
        for it in items:
            mover = p0 if it["depth"] % 2 == 0 else 1 - p0
            mv = it["move"]
            moved = None if mv is None else 1 - mover
            workers = [None, None]
            workers[mover], workers[1 - mover] = list(it["me"]), list(it["op"])
            unvisited_node = it["id"] < 0 or it["n"] == 0
            if mv is None:
                win, net_win = _win(it["q"]), _win(it["net"])
            else:
                win = None if unvisited_node else _win(it["q"])
                net_win = None if it["net"] is None else (1 - it["net"]) / 2
            proof = None
            if it["status"] == 1:
                proof = mover
            elif it["status"] == -1:
                proof = 1 - mover
            terminal = None
            if it["moves"] == 0 and it["status"] == 1:
                terminal = f"{NAMES[mover]} can step up and win"
            elif it["moves"] == 0 and it["status"] == -1:
                terminal = f"{NAMES[mover]} has no safe move"
            build = None if mv is None else mv[2]   # tree branches are never step-up wins (those are terminal nodes)
            nodes.append({
                "id": it["id"], "parent": it["parent"], "depth": it["depth"], "n": it["n"],
                "to_move": mover, "moved_by": moved, "win": win, "net": net_win,
                "prior": it["p"] if mv is not None else None, "q": it["q"], "u": it["u"],
                "puct": (it["q"] if not unvisited_node else 0.0) + it["u"] if mv is not None else None,
                "proof": proof, "terminal": terminal, "expanded": it["expanded"], "unvisited": unvisited_node,
                "move": None if mv is None else [mv[0], mv[1], build],
                "move_text": None if mv is None else move_text(mv[0], mv[1], build),
                "heights": it["heights"], "workers": workers, "pv": it["pv"], "step": it["step"],
                "moves": it["moves"]})
        return {"thought": tid, "root_player": p0, "nodes": nodes, "total_nodes": search.node_count,
                "simulations": search.simulations, "cpuct": float(self.agent.params.cpuct)}

    def _with_tree(self, out, tree):
        if tree:
            params = dict(self.TREE_DEFAULTS, **{k: v for k, v in tree.items() if k in self.TREE_DEFAULTS})
            target = tree.get("target", "live")
            tid = self.live if target == "live" else self._thought_for_entry(int(target))
            out["tree"] = self._tree_json(tid, **params) if tid is not None else None
        return out

    def _thought_for_entry(self, entry):
        for k, t in self.thoughts.items():
            if t["entry"] == entry:
                return k
        return None

    def _response(self, **extra):
        return {"game": self.game.to_json(), "settings": self.settings, "net": self.net.info,
                "step": self.step_info, **extra}

    # -- API ---------------------------------------------------------------------------------
    def handle(self, name, payload=None):
        payload = payload or {}
        try:
            fn = getattr(self, "api_" + name)
        except AttributeError:
            return {"error": f"unknown request '{name}'"}
        try:
            return fn(**payload)
        except (AssertionError, ValueError) as e:
            return {"error": str(e), **self._response()}

    def api_state(self, tree=None):
        return self._with_tree(self._response(thought=self.last_thought), tree)

    def api_new(self, human=0, sims=None, solver=None):
        if sims is not None:
            self.settings["sims"] = int(sims)
        if solver is not None:
            self.settings["solver"] = bool(solver)
        self.agent.set_solver(self.settings["solver"])
        self._reset(Game(human=int(human)))
        return self._response()

    def api_settings(self, sims=None, solver=None):
        if sims is not None:
            self.settings["sims"] = int(sims)
        if solver is not None and bool(solver) != self.settings["solver"]:
            self.settings["solver"] = bool(solver)
            self.agent.set_solver(self.settings["solver"])
            self.agent.search, self.live = None, None
        return self._response()

    def api_place(self, cell):
        g = self.game
        assert g.turn == g.human and g.phase == "place", "it is not your turn to place"
        g.place(int(cell))
        return self._response()

    def api_move(self, frm, to, build):
        g = self.game
        assert g.turn == g.human and g.phase == "play", "it is not your turn"
        entry = len(g.log)
        hint = self.agent.search is not None and self.agent.key == g.key() and self.agent.search.simulations > 0
        if (int(frm), int(to), int(build)) not in set(g.legal()):
            raise ValueError(f"illegal move {move_text(int(frm), int(to), int(build))}")
        if hint:
            self._finish_step()
            self._attach_thought(entry, "hint")
        g.move(int(frm), int(to), int(build))
        if hint:
            g.log[-1]["thought"] = self.live
        self.agent.search, self.live, self.last_thought, self.step_info = None, None, None, None
        return self._response()

    def api_think(self, sims=64, tree=None):
        """Think about the current position (the agent's turn, or a hint on yours)."""
        g = self.game
        if g.phase == "place" and g.turn == g.agent:
            if self.placement is None:
                self.placement = Placement(self.net, first=(g.agent == 0), opp=list(g.workers[g.human]), rng=self.rng)
            self.placement.step(budget=max(256, int(sims) * 16))
            return self._response(placement=self.placement.view())
        if g.phase != "play":
            return self._response()
        self._live_search()
        self._finish_step()
        self.agent.think(int(sims))
        self.last_thought = self._thought_json()
        return self._with_tree(self._response(thought=self.last_thought), tree)

    def api_step(self, action="one", n=1, tree=None):
        """Run the search one stage at a time: select, expand (network), backup."""
        g = self.game
        assert g.phase == "play", "stepping through the search works once the workers are placed"
        s = self._live_search()
        info = {}
        if action in ("select", "one"):
            self._finish_step()
            r = s.step_select(self.step_planes)
            if r == 0:
                info = {"phase": "done", "explain": "The search is finished: the result of this position is already proven."}
        if action in ("expand", "one") and s.step_phase in (1, 2):
            if s.step_phase == 1:
                lg, v = self.net(self.step_planes, 1, pad_to=self.agent.batch)
                s.step_expand(np.ascontiguousarray(lg[0]), float(v[0]))
            else:
                s.step_expand_known()
        if action in ("backup", "one") and s.step_phase == 3:
            s.step_backup()
        if action == "many":
            self._finish_step()
            self.agent.think(int(n))
        if not info:
            info = self._step_info(s)
        self.step_info = info
        self.last_thought = self._thought_json()
        return self._with_tree(self._response(thought=self.last_thought), tree)

    def _step_info(self, s):
        phase = {0: "idle", 1: "selected", 2: "selected", 3: "evaluated", 4: "backed_up"}[s.step_phase]
        if phase == "idle":
            return {"phase": "idle", "explain": ""}
        p0 = self.game.turn
        path_ids = list(s.step_nodes)
        leaf = s.step_leaf
        depth = len(path_ids) - 1
        leaf_mover = p0 if depth % 2 == 0 else 1 - p0
        items = {it["id"]: it for it in s.tree(0, 100000, 100.0, 64, False)} if depth else {}
        path = []
        for nid in path_ids[1:]:
            it = items.get(nid)
            if it and it["move"]:
                mv = it["move"]
                path.append({"id": nid, "text": move_text(mv[0], mv[1], mv[2]), "player": p0 if it["depth"] % 2 == 1 else 1 - p0})
        out = {"phase": phase, "path": path, "leaf": leaf, "leaf_player": leaf_mover, "known": s.step_phase == 2}
        who = NAMES[leaf_mover]
        moves = ", then ".join(f"{NAMES[x['player']]} {x['text']}" for x in path) or "no moves (the current position)"
        if phase == "selected":
            if s.step_phase == 2:
                res = "wins" if s.step_value > 0 else "loses"
                out["explain"] = (f"Select: starting from the current position, it took the move with the highest Q + U at each "
                                  f"step: {moves}. That position is already decided ({who} {res}), so no network call is needed.")
            elif not path:
                out["explain"] = ("Select: this is the first simulation, so there is nothing to choose between yet: the "
                                  "search starts at the current position itself, which the network has not evaluated.")
            else:
                out["explain"] = (f"Select: starting from the current position, it took the move with the highest Q + U "
                                  f"(average value so far plus an exploration bonus) at each step: {moves}. "
                                  f"The network has not looked at the resulting position yet.")
        elif phase == "evaluated":
            v = s.step_value
            sub = s.tree(leaf, 400, 100.0, 1, True)
            kids = sorted([x for x in sub if x["move"] is not None], key=lambda x: -x["p"])[:3]
            fav = ", ".join(f"{move_text(x['move'][0], x['move'][1], x['move'][2])} "
                            f"({100 * x['p']:.0f}%)" for x in kids)
            if out["known"]:
                out["explain"] = f"Evaluate: the result is known exactly, {who} {'wins' if v > 0 else 'loses'}, so the value is {v:+.0f}."
            else:
                out["explain"] = (f"Expand and evaluate: the network says {who}, to move there, has a {100 * (v + 1) / 2:.0f}% chance "
                                  f"to win (value {v:+.2f}). It adds that position's {sub[0]['moves'] if sub else 0} moves as new "
                                  f"branches with its priors; favourites: {fav}.")
            out["value"] = v
        else:
            out["explain"] = (f"Back up: every move on the path got one more visit, and its average value moved toward the "
                              f"evaluation ({s.step_value:+.2f} for {who}, flipped in sign at each step for the other player). "
                              f"The search has now run {s.simulations} simulations.")
        return out

    def api_tree(self, target="live", **params):
        tid = self.live if target == "live" else self._thought_for_entry(int(target))
        return {"tree": self._tree_json(tid, **{k: v for k, v in params.items() if k in self.TREE_DEFAULTS})
                if tid is not None else None}

    def api_thought(self, entry, tree=None):
        """The stored thinking behind log entry `entry` (for the timeline)."""
        tid = self._thought_for_entry(int(entry))
        snap = self.thoughts[tid]["snapshot"] if tid is not None else None
        e = self.game.log[int(entry)] if 0 <= int(entry) < len(self.game.log) else {}
        out = {"thought": snap, "placement": e.get("placement")}
        if tree:
            tree = dict(tree, target=int(entry))
        return self._with_tree(out, tree)

    def api_agent_play(self, tree=None):
        """Play the agent's move (finishing the thinking first if needed)."""
        g = self.game
        assert g.turn == g.agent and g.phase in ("place", "play"), "it is not the agent's turn"
        entry = len(g.log)
        if g.phase == "place":
            if self.placement is None:
                self.placement = Placement(self.net, first=(g.agent == 0), opp=list(g.workers[g.human]), rng=self.rng)
            while not self.placement.done:
                self.placement.step()
            view = self.placement.view()
            cells, score = self.placement.choose()
            for c in cells:
                g.place(c)
            g.log[-1]["note"] = f"expects {100 * (score + 1) / 2:.0f}% to win"
            g.log[-2]["placement"] = g.log[-1]["placement"] = view
            self.placement = None
            return self._response(placement=view, played=cells)
        self._live_search()
        self._finish_step()
        remaining = self.settings["sims"] - self.agent.search.simulations
        if remaining > 0:
            self.agent.think(remaining)
        snap = self._attach_thought(entry, "agent") or self._thought_json()
        best = snap["best"]
        if best is None:  # every legal move lets the opponent win at once: play any of them
            f, t, b = g.legal()[self.rng.integers(len(g.legal()))]
            best = {"from": f, "to": t, "build": b, "text": move_text(f, t, b)}
            snap["best"], snap["status"] = best, -1
            self._record_eval(entry, snap, "agent")
        tid = self.live
        g.move(best["from"], best["to"], best["build"])
        g.log[-1]["thought"] = tid
        g.log[-1]["note"] = (("proven win" if snap["status"] == 1 else "proven loss" if snap["status"] == -1 else
                              f"{100 * (snap['search_value'] + 1) / 2:.0f}% to win") + f", {snap['simulations']} sims")
        self.agent.search, self.live, self.step_info = None, None, None
        self.last_thought = None
        out = self._response(thought=snap, played=best)
        if tree:
            out = self._with_tree(out, dict(tree, target=entry))
        return out

    def api_rewind(self, entry):
        """Go back to the position before log entry `entry` (it must be your turn there)."""
        g = self.game
        k = int(entry)
        assert 0 <= k < len(g.frames), "no such position"
        f = g.frames[k]
        assert f["turn"] == g.human and f["phase"] != "over", "you can only continue from a position where it is your turn"
        g.rewind(k)
        for t in self.thoughts.values():
            if t["entry"] is not None and t["entry"] >= k:
                t["entry"] = None
        self.thoughts = {i: t for i, t in self.thoughts.items() if t["entry"] is not None}
        self.agent.search, self.live, self.placement, self.last_thought, self.step_info = None, None, None, None, None
        return self._response()

    def api_undo(self):
        """Take back your last move (and the agent's reply)."""
        g = self.game
        mine = [i for i, e in enumerate(g.log) if e["player"] == g.human]
        assert mine, "nothing to take back"
        return self.api_rewind(mine[-1])


# --------------------------------------------------------------------------------------
# Front ends
# --------------------------------------------------------------------------------------
def widget_html(transport):
    html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "play.html")).read()
    return html.replace("/*TRANSPORT*/", f"window.SANTORINI_TRANSPORT = {json.dumps(transport)};")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def show_colab(session):
    """Display the widget in a Colab output cell, talking to `session` in this kernel."""
    from IPython.display import HTML, JSON, display
    from google.colab import output

    def api(name, payload_json):
        return JSON(_clean(session.handle(name, json.loads(payload_json or "{}"))))

    output.register_callback("santorini_api", api)
    display(HTML(widget_html("colab")))


def show(session, port=8000):
    """Show the widget: in Colab through kernel callbacks, elsewhere (Jupyter) by starting
    the local web server in a background thread and embedding it."""
    try:
        import google.colab  # noqa: F401
        return show_colab(session)
    except ImportError:
        pass
    import threading
    from IPython.display import IFrame, display
    threading.Thread(target=serve, args=(session, port), daemon=True).start()
    time.sleep(0.5)
    print(f"Widget served at http://127.0.0.1:{port}/ (open it in a browser if the frame stays empty)")
    display(IFrame(f"http://127.0.0.1:{port}/", width="100%", height=1150))


def serve(session, port=8000, host="127.0.0.1"):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, widget_html("http"), "text/html; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if not self.path.startswith("/api/"):
                return self._send(404, "not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            out = _clean(session.handle(self.path[5:], payload))
            self._send(200, json.dumps(out), "application/json")

    print(f"Santorini agent ({session.net.info.get('source')}) on http://{host}:{port}", flush=True)
    HTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="latest.pkl or weights_XXXX.msgpack (default: no network)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    serve(Session(Net(a.ckpt)), a.port, a.host)