"""Santorini AlphaZero training: solver-labelled warm start (phase 0) + self-play (phase 1).

    python train.py                                  # full run, outputs in runs/default
    python train.py --smoke                          # tiny end-to-end check (~minutes on a laptop)
    python train.py --run-dir runs/exp1 --set sims=400 --set iterations=100

Everything is resumable: re-running with the same --run-dir picks up the cached datasets,
the phase-0 weights and the latest self-play checkpoint (latest.pkl).

Conventions: cells A1..E5, row 1 on top. States are always from the side to move's perspective.
"""
import argparse
import ast
import functools
import json
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass, fields

import numpy as np

T_START = time.time()


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')} | {(time.time() - T_START) / 60:6.1f} min] {msg}", flush=True)


def fmt_time(sec):
    sec = max(0.0, float(sec))
    return f"{sec:.0f}s" if sec < 90 else f"{sec / 60:.1f} min" if sec < 5400 else f"{sec / 3600:.1f} h"


class Every:
    """True at most once every `seconds` (rate-limits progress lines)."""

    def __init__(self, seconds):
        self.s, self.t = seconds, time.time()

    def __call__(self):
        if time.time() - self.t >= self.s:
            self.t = time.time()
            return True
        return False


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
CPUS = os.cpu_count() or 2


@dataclass
class Cfg:
    # network (checkpoints only load into a network of the same size)
    blocks: int = 10
    channels: int = 128
    # optimisation
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_exact: float = 3.0        # value-loss weight of solver-proven targets
    alpha_q: float = 0.5             # y = (1-a) z + a q when not proven
    # phase 0: supervised warm start on solver-labelled late positions
    p0_positions: int = 60_000
    p0_min_blocks: int = 24
    p0_max_blocks: int = 55
    p0_root_budget: int = 50_000     # df-pn expansions per position
    p0_child_budget: int = 1_000     # per candidate move, to find the winning moves
    p0_epochs: int = 4
    p0_mix: float = 0.1              # fraction of each self-play batch drawn from phase-0 data
    # held-out solver test set
    test_positions: int = 2_000
    test_min_blocks: int = 20
    test_max_blocks: int = 45
    test_root_budget: int = 200_000
    test_child_budget: int = 5_000
    # self-play
    n_games: int = 512               # games in lock-step = inference batch
    sims: int = 200
    temp_plies: int = 8
    label_min_blocks: int = 22       # exact root solve from this many blocks
    sp_root_budget: int = 20_000
    sp_child_budget: int = 2_000
    iterations: int = 300
    games_per_iter: int = 512
    sample_reuse: float = 4.0        # training samples drawn per new self-play sample
    replay_capacity: int = 500_000
    lr_drops: tuple = (0.6, 0.85)    # lr x0.1 at these fractions of `iterations`
    # evaluation
    arena_games: int = 128
    arena_every: int = 5             # set very large to disable arenas
    # system
    label_threads: int = max(1, CPUS - 1)
    sp_threads: int = max(1, min(16, CPUS - 1))
    tt_gb: float = 0.25              # per solver thread (data generation)
    ckpt_dir: str = "runs/default"
    seed: int = 0
    log_every: float = 20.0          # seconds between progress lines


SMOKE_CFG = dict(blocks=2, channels=16, batch_size=64, p0_positions=256, p0_epochs=2, p0_root_budget=20_000,
                 p0_min_blocks=28, test_positions=64, test_min_blocks=28, test_root_budget=50_000,
                 n_games=16, sims=16, iterations=3, games_per_iter=16, arena_games=8, arena_every=2,
                 tt_gb=0.0625, label_threads=2, sp_threads=2, ckpt_dir="runs/smoke", log_every=2.0)


def parse_overrides(pairs):
    types = {f.name: f.type for f in fields(Cfg)}
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got '{p}'")
        k, v = p.split("=", 1)
        if k not in types:
            raise SystemExit(f"unknown config key '{k}'. Keys: {', '.join(types)}")
        t = types[k]
        if t in (int, "int"):
            out[k] = int(float(v.replace("_", "")))
        elif t in (float, "float"):
            out[k] = float(v)
        elif t in (tuple, "tuple"):
            out[k] = tuple(ast.literal_eval(v))
        else:
            out[k] = v
    return out


def get_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", help="where datasets, weights and checkpoints go (default runs/default)")
    ap.add_argument("--smoke", action="store_true", help="tiny network and budgets for a quick end-to-end test")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config field, e.g. --set sims=400 (repeatable)")
    ap.add_argument("--show-config", action="store_true", help="print the resolved config and exit")
    return ap.parse_args()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    args = get_args()
    cfg = Cfg(**SMOKE_CFG) if args.smoke else Cfg()
    if not args.smoke and CPUS <= 4:
        cfg.label_min_blocks, cfg.sp_root_budget, cfg.sp_child_budget = 26, 10_000, 1_000
        log(f"only {CPUS} CPU cores: lighter in-game solving (from {cfg.label_min_blocks} blocks, "
            f"root budget {cfg.sp_root_budget:,}, child budget {cfg.sp_child_budget:,})")
    for k, v in parse_overrides(args.set).items():
        setattr(cfg, k, v)
    if args.run_dir:
        cfg.ckpt_dir = args.run_dir
    for k, val in asdict(cfg).items():
        print(f"    {k:18s} {val}")
    if args.show_config:
        return

    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import optax
    from flax import serialization
    from flax.training import train_state
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    import santorini as S

    backend = jax.default_backend()
    log(f"JAX {jax.__version__} | backend {backend} | {jax.device_count()} device(s): "
        f"{jax.devices()[0].device_kind} | host CPUs {CPUS} | smoke {args.smoke}")
    log(f"module santorini: {S.NPLANES} input planes, {S.NACT} policy outputs")
    log(f"solver memory: data generation {cfg.label_threads} x {cfg.tt_gb} GB, "
        f"self-play {cfg.sp_threads} x {min(cfg.tt_gb, 0.125)} GB")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    existing = sorted(os.listdir(cfg.ckpt_dir))
    log(f"run folder: {os.path.abspath(cfg.ckpt_dir)}")
    log("  existing files: " + (", ".join(existing) if existing else "none (fresh run)"))
    json.dump(asdict(cfg), open(os.path.join(cfg.ckpt_dir, "config.json"), "w"), indent=2)

    # ---------------------------------------------------------------------------------
    # Network and training step
    # ---------------------------------------------------------------------------------
    NDEV = jax.device_count()
    mesh = Mesh(np.array(jax.devices()), ("data",))
    DATA = NamedSharding(mesh, P("data"))
    REPL = NamedSharding(mesh, P())

    def pick_dtype():
        if backend == "tpu":
            return jnp.bfloat16
        if backend == "gpu":
            try:
                cc = float(jax.devices()[0].compute_capability)
            except Exception:
                cc = 0.0
            return jnp.bfloat16 if cc >= 8.0 else jnp.float32   # A100/L4/H100: bf16; T4/V100: fp32
        return jnp.float32

    DTYPE = pick_dtype()
    assert cfg.batch_size % NDEV == 0, "batch_size must be divisible by the device count"
    INF_BATCH = int(math.ceil(cfg.n_games / NDEV) * NDEV)   # fixed inference shape

    class ResBlock(nn.Module):
        ch: int
        dtype: object

        @nn.compact
        def __call__(self, x, train):
            y = nn.Conv(self.ch, (3, 3), use_bias=False, dtype=self.dtype)(x)
            y = nn.relu(nn.BatchNorm(use_running_average=not train, momentum=0.9, dtype=self.dtype)(y))
            y = nn.Conv(self.ch, (3, 3), use_bias=False, dtype=self.dtype)(y)
            y = nn.BatchNorm(use_running_average=not train, momentum=0.9, dtype=self.dtype)(y)
            return nn.relu(x + y)

    class SantoriniNet(nn.Module):
        blocks: int
        ch: int
        dtype: object = jnp.float32

        @nn.compact
        def __call__(self, x, train=False):                       # x: [B, 5, 5, 15]
            bn = lambda: nn.BatchNorm(use_running_average=not train, momentum=0.9, dtype=self.dtype)
            x = x.astype(self.dtype)
            x = nn.relu(bn()(nn.Conv(self.ch, (3, 3), use_bias=False, dtype=self.dtype)(x)))
            for _ in range(self.blocks):
                x = ResBlock(self.ch, self.dtype)(x, train)
            p = nn.relu(bn()(nn.Conv(32, (1, 1), use_bias=False, dtype=self.dtype)(x)))
            p = nn.Dense(S.NACT, dtype=jnp.float32)(p.reshape(p.shape[0], -1))
            v = nn.relu(bn()(nn.Conv(16, (1, 1), use_bias=False, dtype=self.dtype)(x)))
            v = nn.relu(nn.Dense(128, dtype=jnp.float32)(v.reshape(v.shape[0], -1)))
            v = jnp.tanh(nn.Dense(1, dtype=jnp.float32)(v))[:, 0]
            return p.astype(jnp.float32), v.astype(jnp.float32)

    model = SantoriniNet(cfg.blocks, cfg.channels, DTYPE)

    class TrainState(train_state.TrainState):
        batch_stats: dict

    def make_tx(learning_rate):
        return optax.chain(optax.clip_by_global_norm(1.0),
                           optax.adamw(learning_rate, weight_decay=cfg.weight_decay))

    def create_state(seed):
        variables = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 5, 5, S.NPLANES)), train=False)
        tx = optax.inject_hyperparams(make_tx)(learning_rate=cfg.lr)
        st = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx,
                               batch_stats=variables["batch_stats"])
        return jax.device_put(st, REPL)

    def set_lr(state, lr):
        state.opt_state.hyperparams["learning_rate"] = jax.device_put(jnp.asarray(lr, jnp.float32), REPL)
        return state

    @functools.partial(jax.jit, donate_argnums=0)
    def train_step(state, batch):
        def loss_fn(params):
            (logits, v), upd = state.apply_fn({"params": params, "batch_stats": state.batch_stats},
                                              batch["x"], train=True, mutable=["batch_stats"])
            legal = batch["legal"]
            legal = jnp.where(legal.any(1, keepdims=True), legal, True)   # guard all-illegal rows
            logp = jax.nn.log_softmax(jnp.where(legal, logits[:, :S.NMOVE], -1e9))
            has_pi = (batch["pi"].sum(1) > 0).astype(jnp.float32)
            pol = -(batch["pi"] * logp).sum(1)
            pol_loss = (pol * has_pi).sum() / jnp.maximum(has_pi.sum(), 1.0)
            val_loss = (batch["w"] * (batch["y"] - v) ** 2).sum() / batch["w"].sum()
            top1 = jnp.argmax(logp, 1)
            hit = jnp.take_along_axis(batch["pi"], top1[:, None], 1)[:, 0] > 0
            m = {"loss": pol_loss + val_loss, "policy": pol_loss, "value": val_loss,
                 "top1_in_target": (hit * has_pi).sum() / jnp.maximum(has_pi.sum(), 1.0)}
            return pol_loss + val_loss, (upd, m)
        (_, (upd, m)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads, batch_stats=upd["batch_stats"]), m

    @jax.jit
    def _forward(variables, x):
        return model.apply(variables, x, train=False)

    def variables_of(state):
        return {"params": state.params, "batch_stats": state.batch_stats}

    def predict(variables, x):
        # Network on any number of rows, in fixed INF_BATCH chunks (one compiled shape).
        # variables=None is the uniform baseline (zero logits, zero value).
        n = len(x)
        if variables is None:
            return np.zeros((n, S.NACT), np.float32), np.zeros(n, np.float32)
        L, V = [], []
        for i in range(0, n, INF_BATCH):
            chunk = x[i:i + INF_BATCH]
            pad = INF_BATCH - len(chunk)
            if pad:
                chunk = np.concatenate([chunk, np.zeros((pad,) + chunk.shape[1:], chunk.dtype)])
            lg, v = _forward(variables, jax.device_put(chunk, DATA))
            L.append(np.asarray(lg)[:INF_BATCH - pad]); V.append(np.asarray(v)[:INF_BATCH - pad])
        return np.concatenate(L), np.concatenate(V)

    state = create_state(cfg.seed)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    log(f"SantoriniNet {cfg.blocks}x{cfg.channels}: {n_params / 1e6:.2f}M params, compute {DTYPE.__name__}, "
        f"{NDEV} device(s), inference batch {INF_BATCH}, train batch {cfg.batch_size}")
    t = time.time()
    predict(variables_of(state), np.zeros((INF_BATCH, 5, 5, S.NPLANES), np.float32))
    log(f"inference compiled in {fmt_time(time.time() - t)}")

    # ---------------------------------------------------------------------------------
    # Data, batches and the training loop helper
    # ---------------------------------------------------------------------------------
    class Replay:
        def __init__(self, cap):
            self.state = np.zeros((cap, S.STATE_LEN), np.int8)
            self.pi = np.zeros((cap, S.NMOVE), np.float16)
            self.y = np.zeros(cap, np.float32)
            self.w = np.zeros(cap, np.float32)
            self.cap, self.n, self.i = cap, 0, 0

        def add(self, state, pi, y, w):
            n = len(y)
            idx = (self.i + np.arange(n)) % self.cap
            self.state[idx], self.pi[idx], self.y[idx], self.w[idx] = state, pi, y, w
            self.i = (self.i + n) % self.cap
            self.n = min(self.cap, self.n + n)

        def sample(self, k, rng):
            j = rng.integers(0, self.n, k)
            return self.state[j], self.pi[j].astype(np.float32), self.y[j], self.w[j]

    def make_batch(states, pi, y, w, rng):
        # Random D4 symmetry per sample; policy targets are permuted with the board.
        syms = rng.integers(0, 8, len(y)).astype(np.int32)
        x, legal, perm = S.encode(states, syms)
        pi_t = np.zeros_like(pi, dtype=np.float32)
        np.put_along_axis(pi_t, perm, pi.astype(np.float32), axis=1)   # pi_t[perm[a]] = pi[a]
        return {"x": x, "legal": legal.astype(bool), "pi": pi_t, "y": y.astype(np.float32),
                "w": w.astype(np.float32)}

    def solver_targets(values, win):
        # Phase-0 / test targets: exact value; policy = uniform over proven winning moves.
        pi = win.astype(np.float32)
        s = pi.sum(1, keepdims=True)
        pi = np.where(s > 0, pi / np.maximum(s, 1), 0)
        return pi, values.astype(np.float32)

    compiled = [False]

    def run_training(state, steps, batch_fn, lr_fn, label):
        # `steps` train steps; logs averaged metrics about every cfg.log_every seconds.
        t0, tick = time.time(), Every(cfg.log_every)
        window, total, t_data = [], [], 0.0
        for s in range(steps):
            state = set_lr(state, lr_fn(s))
            a = time.time()
            batch = jax.device_put(batch_fn(), DATA)
            t_data += time.time() - a
            state, m = train_step(state, batch)
            window.append(m); total.append(m)
            if s == 0 and not compiled[0]:
                jax.block_until_ready(m["loss"])
                compiled[0] = True
                log(f"{label}: first step done in {fmt_time(time.time() - t0)} (includes XLA compilation)")
            if tick() or s == steps - 1:
                avg = {k: float(np.mean([float(x[k]) for x in window])) for k in window[0]}
                window = []
                el = time.time() - t0
                rate = (s + 1) / el
                log(f"{label}: step {s + 1}/{steps} ({100 * (s + 1) / steps:.0f}%) | loss {avg['loss']:.3f} "
                    f"(policy {avg['policy']:.3f}, value {avg['value']:.3f}) | top1 in target "
                    f"{avg['top1_in_target']:.2f} | lr {lr_fn(s):.2e} | {rate:.1f} steps/s, "
                    f"{rate * cfg.batch_size:,.0f} samples/s | batch prep {100 * t_data / el:.0f}% of time "
                    f"| ETA {fmt_time((steps - s - 1) / rate)}")
        avg = {k: float(np.mean([float(x[k]) for x in total])) for k in total[0]} if total else {}
        return state, avg

    # ---------------------------------------------------------------------------------
    # Solver-labelled data (cached chunk by chunk, resumable)
    # ---------------------------------------------------------------------------------
    def generate_cached(name, n, lo, hi, root_budget, child_budget, seed, chunk=None):
        path = os.path.join(cfg.ckpt_dir, f"{name}.npz")
        if os.path.exists(path):
            d = np.load(path)
            if len(d["v"]) >= n:
                log(f"{name}: loaded {len(d['v'])} positions from {path}")
                return d["states"][:n], d["v"][:n], d["win"][:n]
        chunk = chunk or max(64, 8 * cfg.label_threads)        # ~8 positions per solver thread per chunk
        parts_dir = os.path.join(cfg.ckpt_dir, f"{name}_parts_c{chunk}")
        os.makedirs(parts_dir, exist_ok=True)
        n_chunks = math.ceil(n / chunk)
        part = lambda k: os.path.join(parts_dir, f"part_{k:05d}.npz")
        done = sum(min(chunk, n - k * chunk) for k in range(n_chunks) if os.path.exists(part(k)))
        log(f"{name}: {n} positions, blocks {lo}-{hi}, root budget {root_budget:,}, child budget "
            f"{child_budget:,}, {cfg.label_threads} threads, {n_chunks} chunks of {chunk}"
            + (f" | resuming with {done} already saved" if done else ""))
        t0, made = time.time(), 0
        for k in range(n_chunks):
            if os.path.exists(part(k)):
                continue
            m, tc = min(chunk, n - k * chunk), time.time()
            st, v, w = S.generate(m, lo, hi, root_budget=root_budget, child_budget=child_budget,
                                  threads=cfg.label_threads, tt_gb=cfg.tt_gb, seed=seed * 100_003 + k)
            tmp = os.path.join(parts_dir, f"tmp_{k:05d}.npz")
            np.savez(tmp, states=st, v=v, win=w)
            os.replace(tmp, part(k))
            made, done = made + m, done + m
            rate = made / (time.time() - t0)
            b = st[:, :25].astype(np.int32).sum(1)
            log(f"  {name}: chunk {k + 1}/{n_chunks} | {done}/{n} ({100 * done / n:.0f}%) | "
                f"{m / (time.time() - tc):.1f} pos/s (avg {rate:.1f}) | ETA {fmt_time((n - done) / rate)} | "
                f"mover wins {np.mean(v == 1):.2f} | blocks {b.min()}-{b.max()} (mean {b.mean():.0f}) | "
                f"winning moves known {w[v == 1].sum(1).mean() if (v == 1).any() else 0:.1f}")
        parts = [np.load(part(k)) for k in range(n_chunks)]
        states, v, win = (np.concatenate([p[key] for p in parts]) for key in ("states", "v", "win"))
        np.savez_compressed(path, states=states, v=v, win=win)
        log(f"{name}: saved {len(v)} positions to {path} ({fmt_time(time.time() - t0)} this run)")
        return states, v, win

    test_states, test_v, test_win = generate_cached("testset", cfg.test_positions, cfg.test_min_blocks,
                                                    cfg.test_max_blocks, cfg.test_root_budget,
                                                    cfg.test_child_budget, seed=777)
    test_blocks = test_states[:, :25].astype(np.int32).sum(1)
    log(f"test set: {len(test_v)} positions, mover wins {np.mean(test_v == 1):.2f}, "
        f"blocks {test_blocks.min()}-{test_blocks.max()}")

    def eval_solver_set(variables):
        t = time.time()
        x, legal, _ = S.encode(test_states, np.zeros(len(test_v), np.int32))
        logits, v = predict(variables, x)
        lg = np.where(legal, logits[:, :S.NMOVE], -1e9)
        top = lg.argmax(1)
        won = test_v == 1
        out = {"value_sign_acc": float(np.mean(np.sign(v) == test_v)),
               "won_top1_winning": float(np.mean(test_win[won, top[won]] > 0)),
               "value_mse": float(np.mean((v - test_v) ** 2))}
        for lo, hi in [(15, 24), (25, 34), (35, 68)]:
            sel = (test_blocks >= lo) & (test_blocks <= hi)
            if sel.any():
                out[f"sign_acc_{lo}-{hi}"] = float(np.mean(np.sign(v[sel]) == test_v[sel]))
        out["eval_seconds"] = time.time() - t
        return out

    def fmt_eval(ev):
        return " | ".join(f"{k} {v:.3f}" for k, v in ev.items() if k != "eval_seconds")

    log("untrained network on the test set: " + fmt_eval(eval_solver_set(variables_of(state))))

    p0_states, p0_v, p0_win = generate_cached("phase0", cfg.p0_positions, cfg.p0_min_blocks, cfg.p0_max_blocks,
                                              cfg.p0_root_budget, cfg.p0_child_budget, seed=1)
    p0_pi, p0_y = solver_targets(p0_v, p0_win)
    p0_w = np.ones(len(p0_y), np.float32)
    p0_blocks = p0_states[:, :25].astype(np.int32).sum(1)
    log(f"phase-0 set: {len(p0_y)} positions, mover wins {np.mean(p0_v == 1):.2f}, "
        f"blocks {p0_blocks.min()}-{p0_blocks.max()} (mean {p0_blocks.mean():.0f}), "
        f"policy targets on {np.mean(p0_pi.sum(1) > 0):.0%} of positions")

    # ---------------------------------------------------------------------------------
    # Phase 0: supervised warm start (skipped when a self-play checkpoint already exists)
    # ---------------------------------------------------------------------------------
    rng = np.random.default_rng(cfg.seed)
    history = []
    latest = os.path.join(cfg.ckpt_dir, "latest.pkl")
    p0_done = os.path.join(cfg.ckpt_dir, "phase0_state.msgpack")
    if os.path.exists(latest):
        log("self-play checkpoint found: skipping phase 0")
    elif os.path.exists(p0_done):
        state = jax.device_put(serialization.from_bytes(jax.device_get(state), open(p0_done, "rb").read()), REPL)
        log(f"loaded phase-0 weights from {p0_done}")
    else:
        steps = int(cfg.p0_epochs * len(p0_y) / cfg.batch_size)
        sched = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, max(1, steps // 20), max(2, steps), cfg.lr * 0.05)
        log(f"phase 0: {steps} steps = {cfg.p0_epochs} epochs over {len(p0_y)} positions, batch {cfg.batch_size}")

        def p0_batch():
            j = rng.integers(0, len(p0_y), cfg.batch_size)
            return make_batch(p0_states[j], p0_pi[j], p0_y[j], p0_w[j], rng)

        state, _ = run_training(state, steps, p0_batch, lambda s: float(sched(s)), "phase 0")
        open(p0_done, "wb").write(serialization.to_bytes(jax.device_get(state)))
        log(f"saved phase-0 weights to {p0_done}")
    if not os.path.exists(latest):
        ev = eval_solver_set(variables_of(state))
        history.append({"iter": 0, "phase": 0, **ev})
        log("after phase 0, test set: " + fmt_eval(ev))

    # ---------------------------------------------------------------------------------
    # Self-play and arena
    # ---------------------------------------------------------------------------------
    plane_buf = np.zeros((INF_BATCH, 5, 5, S.NPLANES), np.float32)

    def sp_config(n_games, arena=False, max_games=1 << 30, seed=0):
        c = S.Config()
        c.n_games, c.sims, c.seed = n_games, cfg.sims, seed
        c.label_min_blocks, c.root_budget, c.child_budget = cfg.label_min_blocks, cfg.sp_root_budget, cfg.sp_child_budget
        c.threads, c.tt_gb = cfg.sp_threads, min(cfg.tt_gb, 0.125)
        c.temp_plies = 4 if arena else cfg.temp_plies
        c.noise, c.arena, c.max_games = not arena, arena, max_games
        return c

    def play_rounds(sp, nets, stop, label, games_target):
        # nets: one variables dict (self-play) or two (arena: net_id 0 / 1); None = uniform baseline.
        t0, tick = time.time(), Every(cfg.log_every)
        s0, g0 = sp.stats, sp.games_finished
        rounds, t_cpp, t_net = 0, 0.0, 0.0

        def report(final=False):
            s, el = sp.stats, max(time.time() - t0, 1e-9)
            games = sp.games_finished - g0
            moves = s["moves_played"] - s0["moves_played"]
            solves = s["root_solves"] - s0["root_solves"]
            proven = s["root_proven"] - s0["root_proven"]
            rate = games / el
            eta = f"ETA {fmt_time((games_target - games) / rate)}" if rate > 0 and not final else ""
            log(f"{label}: {'done ' if final else ''}{games}/{games_target} games | {sp.active_games} active | "
                f"{moves / el:.1f} moves/s | {(s['leaf_evals'] - s0['leaf_evals']) / el:,.0f} net evals/s "
                f"({rounds} rounds) | root solves {solves} ({proven} proven) | "
                f"proof moves {s['solver_moves'] - s0['solver_moves']} | "
                f"time C++ {100 * t_cpp / el:.0f}% / net {100 * t_net / el:.0f}% | {fmt_time(el)} {eta}")

        while not stop():
            a = time.time()
            n, gi, ni = sp.prepare(plane_buf)
            b = time.time(); t_cpp += b - a
            if n == 0:
                break
            outs = [predict(net, plane_buf) for net in nets]
            c = time.time(); t_net += c - b
            if len(outs) == 1:
                logits, values = outs[0][0][:n], outs[0][1][:n]
            else:
                pick = (ni == 1)
                logits = np.where(pick[:, None], outs[1][0][:n], outs[0][0][:n])
                values = np.where(pick, outs[1][1][:n], outs[0][1][:n])
            sp.apply(gi, logits, values)
            t_cpp += time.time() - c
            rounds += 1
            if tick():
                report()
        report(final=True)
        return rounds

    def arena(net_a, net_b, games, seed, label):
        # Win rate of net_a against net_b over `games` games (colours alternate).
        sp = S.SelfPlay(sp_config(min(games, INF_BATCH), arena=True, max_games=games, seed=seed))
        play_rounds(sp, [net_a, net_b], stop=lambda: sp.active_games == 0, label=label, games_target=games)
        a, b = sp.net_wins
        log(f"{label}: {a} wins, {b} losses -> win rate {a / max(1, a + b):.2f}")
        return a / max(1, a + b)

    # ---------------------------------------------------------------------------------
    # Phase 1: AlphaZero self-play loop
    # ---------------------------------------------------------------------------------
    replay = Replay(cfg.replay_capacity)
    start_iter = 1
    if os.path.exists(latest):
        t = time.time()
        ck = pickle.load(open(latest, "rb"))
        state = jax.device_put(serialization.from_bytes(jax.device_get(state), ck["state"]), REPL)
        history, start_iter = ck["history"], ck["iter"] + 1
        if "replay" in ck:
            r = ck["replay"]; replay.add(r["state"], r["pi"], r["y"], r["w"])
        log(f"resumed from {latest}: iteration {start_iter}, {replay.n} replay samples, "
            f"saved on {ck.get('backend', '?')} ({fmt_time(time.time() - t)} to load)")
    else:
        log("no self-play checkpoint yet: starting phase 1 from the phase-0 weights")

    def save_checkpoint(it):
        t = time.time()
        ck = {"state": serialization.to_bytes(jax.device_get(state)), "history": history, "iter": it,
              "cfg": asdict(cfg), "backend": backend}
        n_keep = min(replay.n, 200_000)        # keep the newest samples with the checkpoint
        idx = (replay.i - 1 - np.arange(n_keep)) % replay.cap
        ck["replay"] = {"state": replay.state[idx], "pi": replay.pi[idx], "y": replay.y[idx], "w": replay.w[idx]}
        tmp = latest + ".tmp"
        pickle.dump(ck, open(tmp, "wb")); os.replace(tmp, latest)
        if it % 25 == 0:
            open(os.path.join(cfg.ckpt_dir, f"weights_{it:04d}.msgpack"), "wb").write(ck["state"])
        log(f"checkpoint saved: iteration {it}, {os.path.getsize(latest) / 2**20:.0f} MB, {n_keep} replay "
            f"samples, {fmt_time(time.time() - t)}")

    def lr_at(it):
        lr = cfg.lr
        for f in cfg.lr_drops:
            if it >= f * cfg.iterations:
                lr *= 0.1
        return lr

    def frozen_copy(variables):                             # survives buffer donation in train_step
        return jax.device_put(jax.device_get(variables), REPL)

    sp = S.SelfPlay(sp_config(INF_BATCH, seed=cfg.seed * 7919 + start_iter))
    snapshot = frozen_copy(variables_of(state))              # opponent for the arena
    log(f"phase 1: iterations {start_iter}..{cfg.iterations}, {cfg.games_per_iter} games each, "
        f"{cfg.sims} sims/move, {INF_BATCH} games in parallel")
    iter_times = []
    for it in range(start_iter, cfg.iterations + 1):
        t0 = time.time()
        log(f"===== iteration {it}/{cfg.iterations} | lr {lr_at(it):.1e} | replay {replay.n} samples =====")
        target = sp.games_finished + cfg.games_per_iter
        g0, fpw0 = sp.games_finished, sp.first_player_wins
        play_rounds(sp, [variables_of(state)], stop=lambda: sp.games_finished >= target,
                    label=f"iter {it} self-play", games_target=cfg.games_per_iter)
        d = sp.harvest()
        exact = d["exact"] == 1
        y = np.where(exact, d["v"], (1 - cfg.alpha_q) * d["z"] + cfg.alpha_q * d["q"]).astype(np.float32)
        w = np.where(exact, cfg.lambda_exact, 1.0).astype(np.float32)
        replay.add(d["state"], d["pi"], y, w)
        t_play = time.time() - t0
        games = max(1, sp.games_finished - g0)
        log(f"iter {it}: {len(y)} new samples ({exact.mean():.0%} solver-exact) from {games} games, "
            f"first player won {(sp.first_player_wins - fpw0) / games:.0%} | replay now {replay.n}")

        steps = max(1, int(cfg.sample_reuse * len(y) / cfg.batch_size))
        n_p0 = int(cfg.p0_mix * cfg.batch_size)

        def sp_batch():
            a = replay.sample(cfg.batch_size - n_p0, rng)
            j = rng.integers(0, len(p0_y), n_p0)
            return make_batch(np.concatenate([a[0], p0_states[j]]), np.concatenate([a[1], p0_pi[j]]),
                              np.concatenate([a[2], p0_y[j]]), np.concatenate([a[3], p0_w[j]]), rng)

        state, m = run_training(state, steps, sp_batch, lambda s: lr_at(it), f"iter {it} training")

        ev = eval_solver_set(variables_of(state))
        log(f"iter {it} test set: " + fmt_eval(ev))
        rec = {"iter": it, "phase": 1, "samples": len(y), "exact_frac": float(exact.mean()),
               "first_player_win": (sp.first_player_wins - fpw0) / games, "train_steps": steps,
               "lr": lr_at(it), "t_play": t_play, **m, **ev}
        if it % cfg.arena_every == 0:
            cur = variables_of(state)
            rec["arena_vs_prev"] = arena(cur, snapshot, cfg.arena_games, seed=it, label=f"iter {it} arena vs previous")
            rec["arena_vs_uniform"] = arena(cur, None, cfg.arena_games, seed=it + 1,
                                            label=f"iter {it} arena vs uniform")
            snapshot = frozen_copy(cur)
        rec["t_total"] = time.time() - t0
        history.append(rec)
        save_checkpoint(it)
        plot_progress(history, os.path.join(cfg.ckpt_dir, "progress.png"))
        iter_times.append(rec["t_total"])
        left = cfg.iterations - it
        log(f"iter {it} finished in {fmt_time(rec['t_total'])} (play {fmt_time(t_play)}) | "
            f"{left} iterations left, ETA {fmt_time(left * np.mean(iter_times[-5:]))}")
    log("phase 1 complete")


def plot_progress(history, path):
    """Loss, solver test-set accuracy and arena win rate per iteration -> PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    h1 = [h for h in history if h["phase"] == 1]
    if not h1:
        return
    it = [h["iter"] for h in h1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(it, [h["policy"] for h in h1], label="policy loss")
    ax[0].plot(it, [h["value"] for h in h1], label="value loss")
    ax[0].legend(); ax[0].set_xlabel("iteration")
    for k in [k for k in h1[-1] if k.startswith("sign_acc_")] + ["won_top1_winning"]:
        ax[1].plot(it, [h.get(k, np.nan) for h in h1], label=k)
    ax[1].legend(); ax[1].set_title("solver test set"); ax[1].set_xlabel("iteration")
    ar = [h for h in h1 if "arena_vs_prev" in h]
    ax[2].plot([h["iter"] for h in ar], [h["arena_vs_prev"] for h in ar], "o-", label="vs previous snapshot")
    ax[2].plot([h["iter"] for h in ar], [h["arena_vs_uniform"] for h in ar], "s-", label="vs uniform-prior MCTS")
    ax[2].axhline(0.5, color="grey", lw=0.5); ax[2].set_ylim(0, 1); ax[2].legend(); ax[2].set_title("arena win rate")
    plt.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
