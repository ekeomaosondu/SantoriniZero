# SantoriniZero

An AlphaZero-style agent for the base game of **Santorini** (no god powers). It combines:

- a fast **C++ engine** (bitboards, move generation, forced-move pruning, D4 symmetry hashing);
- an exact **df-pn proof-number solver** that labels late-game positions as proven wins or losses;
- a batched **self-play MCTS** in C++ with MCTS-Solver proof propagation and an exact root solve late in the game;
- a **JAX/Flax ResNet** policy/value network trained first on solver labels, then by self-play;
- a **browser UI** for playing against the agent and watching its search.

A trained checkpoint (`checkpoints/latest.pkl`) is included, so you can play right after cloning.

---

## Contents

- [Quick start: play against the agent](#quick-start-play-against-the-agent)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Playing](#playing)
- [Training](#training)
  - [Running a training job](#running-a-training-job)
  - [How the training loop works](#how-the-training-loop-works)
  - [Configuration reference](#configuration-reference)
  - [Outputs, checkpoints and resuming](#outputs-checkpoints-and-resuming)
  - [Reading the logs](#reading-the-logs)
- [The included checkpoint](#the-included-checkpoint)
- [Tests](#tests)
- [Colab notebooks](#colab-notebooks)
- [Technical details](#technical-details)

---

## Quick start: play against the agent

```bash
git clone https://github.com/ekeomaosondu/SantoriniZero.git
cd SantoriniZero
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                       # compiles the C++ module (~20 s)
python play/santorini_agent.py --ckpt checkpoints/latest.pkl
```

Open <http://127.0.0.1:8000> in your browser. (`make install` and `make play` do the same steps.)

---

## Repository layout

```
SantoriniZero/
├── cpp/                       C++ core, compiled into the Python module `santorini`
│   ├── engine.h               board, bitboards, move generation, forced-move pruning, symmetries
│   ├── solver.h               transposition table + df-pn solver, training labels
│   ├── az.h                   AlphaZero layer: action encoding, input planes, labelling, self-play MCTS
│   ├── agent.h                single-position search used for play/analysis (with introspection)
│   ├── santorini_py.cpp       pybind11 bindings
│   └── test.cpp               correctness suite (independent brute-force reference)
├── train.py                   training: data generation, phase 0, self-play loop, evaluation
├── play/
│   ├── santorini_agent.py     agent, game referee and local HTTP server
│   └── play.html              the interactive board / search visualiser
├── tests/check_bindings.py    quick Python-side checks of the compiled module
├── checkpoints/latest.pkl     trained checkpoint (self-play iteration 13)
├── notebooks/                 the original Google Colab notebooks (training on GPU, playing)
├── setup.py, pyproject.toml   build of the C++ extension
├── requirements.txt
└── Makefile                   shortcuts: install, test, smoke, train, play
```

---

## Installation

**Requirements**
- Python 3.10–3.12 (3.12 recommended).
- A C++17 compiler: `clang` (macOS: `xcode-select --install`) or `g++` (Linux: `sudo apt install build-essential`).
- For training at full size, an NVIDIA GPU is strongly recommended. Playing works fine on CPU.

**Steps**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` compiles `cpp/santorini_py.cpp` into the `santorini` extension module. To rebuild after editing the C++, run `pip install -e .` again or `make build`.

The build uses `-march=native` for speed. If you need a binary that runs on other machines, or your compiler rejects the flag, build with:

```bash
SANTORINI_NATIVE=0 pip install -e .
```

**GPU (NVIDIA, Linux).** `requirements.txt` installs the CPU build of JAX. For CUDA, also run:

```bash
pip install -U "jax[cuda12]"
```

Check with `python -c "import jax; print(jax.default_backend(), jax.devices())"`, which should report `gpu`. On A100/L4/H100-class GPUs (compute capability ≥ 8.0) the network runs in bfloat16; on older ones (T4, V100) in float32. Parameters are always stored in float32, so checkpoints move freely between CPU, GPU and TPU.

---

## Playing

```bash
python play/santorini_agent.py --ckpt checkpoints/latest.pkl [--port 8000] [--host 127.0.0.1]
# or
make play                     # CKPT=... PORT=... to override
```

Then open `http://127.0.0.1:8000`.

- `--ckpt` accepts a `latest.pkl` from training, a `weights_XXXX.msgpack` snapshot or `phase0_state.msgpack`. The network size is read from the file. Without `--ckpt`, the agent plays with MCTS and the solver only, without a network.
- To make the server reachable from other machines on your network, use `--host 0.0.0.0`. The server has no authentication, so only do this on a trusted network.

**What the board shows**
- **The agent** uses the network plus the same MCTS used in self-play, with the exact df-pn solver at the root. You choose the number of simulations per move, from 64 (quick) to 4,000 (deep). On a laptop CPU, 400 simulations take about 2 seconds.
- **Arrows** show the moves it is weighing (thicker means more simulations). Squares mark where it would build.
- **"Which moves did it try?"** shows simulations per move as bars, the network prior as dashed outlines and the win chance after search. Hover a bar to draw its expected line on the board.
- **The search tree** shows every position the search reached, with visit count N, win chance, the network's value and prior, and optionally the PUCT score. Click a node to explore from there.
- **Step through the search** runs it one stage at a time: Select, Expand and evaluate (one network call), Back up. Tick "Pause the agent on its turn" to step through the agent's own search.
- **Game timeline**: review any earlier move with the thinking behind it, or "Play from here".
- **Placement**: the network was trained on random worker placements, so at play time the agent chooses placements by evaluating every resulting position with the value head. When it places first, the board shades each cell by its value. Turn on **Hints on my turn** for the same analysis of your own moves.

**From Python / Jupyter**

```python
import sys; sys.path.insert(0, "play")
from santorini_agent import Net, Session, show
session = Session(Net("checkpoints/latest.pkl"))
show(session)          # serves the widget locally and embeds it in the notebook
```

---

## Training

### Running a training job

```bash
python train.py --smoke                    # tiny network and budgets: an end-to-end check in ~1 minute
python train.py                            # full run, outputs in runs/default/
python train.py --run-dir runs/exp1 --set sims=400 --set iterations=100
python train.py --show-config [--set ...]  # print the resolved config and exit
```

- `--run-dir DIR` sets where datasets, weights and checkpoints go (default `runs/default`, or `runs/smoke` with `--smoke`).
- `--set KEY=VALUE` overrides any field of the config (repeatable). See the [configuration reference](#configuration-reference).
- `--smoke` uses a 2-block × 16-channel network, 256 phase-0 positions, 16 games per iteration and 3 iterations.

To keep a log file as well as the console output:

```bash
python train.py --run-dir runs/exp1 2>&1 | tee runs/exp1.log
```

Training is **fully resumable**: stop it at any time (Ctrl-C, crash, reboot) and run the same command again. It continues from the last completed step (see [resuming](#outputs-checkpoints-and-resuming)).

**Hardware and time.** Data generation and the in-game solver run on CPU threads, while network inference and training run on the accelerator. With the default settings, solver data generation takes tens of minutes to hours depending on core count, phase 0 takes minutes, and each self-play iteration (512 games × 200 simulations) took about 4 minutes on a Colab GPU with 2 CPU cores. On hosts with 4 or fewer CPU cores, in-game solving is made lighter automatically.

### How the training loop works

Training has two phases. A held-out, solver-labelled test set is used to measure progress throughout.

#### 0. Data generation (C++, all CPU cores)

`S.generate(n, lo, hi, root_budget, child_budget, …)` plays random *safe* moves from a random placement until the board has a random number of blocks in `[lo, hi]`. A move is safe if it doesn't hand the opponent an immediate win. The resulting position is then solved with df-pn:

- the **root solve** (up to `root_budget` node expansions) proves the position a win (+1) or loss (−1) for the side to move. Unresolved positions are dropped;
- for won positions, each candidate move gets its own solve (up to `child_budget`) to find **which moves win**. That set becomes the policy target.

Two datasets are generated and cached in the run dir:

| dataset | used for | default size | blocks | budgets |
|---|---|---|---|---|
| `testset.npz` | evaluation only, never trained on | 2,000 | 20–45 | 200k / 5k |
| `phase0.npz` | phase-0 training and the `p0_mix` share of every later batch | 60,000 | 24–55 | 50k / 1k |

Generation is saved chunk by chunk (`*_parts_c*/`), so an interruption loses at most one chunk.

#### Phase 0: supervised warm start

The network learns endgame values and winning moves from exact labels before any self-play:

- value target = the proven result (±1); policy target = uniform over the proven winning moves (no policy target for lost positions);
- `p0_epochs` (4) passes over the data, batch 1024, AdamW with warmup + cosine decay to 5% of `lr`;
- the weights are saved to `phase0_state.msgpack`, and a rerun loads them instead of retraining.

This gives self-play a network that already recognises tactical wins and losses near the end of the game, which is where random networks waste most self-play games.

#### Phase 1: AlphaZero self-play loop

Each iteration (`iterations` = 300 by default) has four steps.

**1. Self-play.** `games_per_iter` (512) games are played with the current network. `n_games` (512) games run in lock-step inside one C++ `SelfPlay` object:

- each round, every game descends its MCTS tree to one leaf that needs the network. All leaves are evaluated in **one batched forward pass** on the accelerator, and the results are fed back for expansion and backup;
- search: PUCT with `c_puct = 1.5`, `sims` (200) simulations per move, Dirichlet noise (α = 0.3, 25%) at the root;
- **forced-move pruning**: moves that let the opponent win on the next step are never searched, and positions with an immediate win are terminal;
- **MCTS-Solver**: proven wins and losses propagate up the tree, so solved subtrees cost nothing;
- **exact root solve**: from `label_min_blocks` (22) blocks on, each root is first solved with df-pn (`sp_root_budget` / `sp_child_budget`). If proven, the move is played straight from the proof and the sample gets an **exact value**;
- moves are sampled in proportion to visit counts for the first `temp_plies` (8) plies, then played greedily;
- worker placement is random (the 25 placement actions in the policy head are reserved and masked).

**2. Store samples.** Each position yields `(state, π, target y, weight w)` in a replay buffer of `replay_capacity` (500k):

- π = MCTS visit distribution over the 128 move+build actions;
- if the solver proved the position: `y = v_proven`, `w = lambda_exact` (3.0);
- otherwise: `y = (1 − alpha_q)·z + alpha_q·q`, `w = 1`, where `z` is the final game result and `q` the root search value (`alpha_q` = 0.5).

**3. Train.** The network takes `sample_reuse × new_samples / batch_size` steps (each new sample is seen about 4 times on average):

- each batch of 1024 draws `1 − p0_mix` (90%) from the replay buffer and `p0_mix` (10%) from phase-0 data, so endgame knowledge isn't forgotten;
- each sample gets a random D4 symmetry (8 rotations/reflections). The C++ `encode` returns the transformed planes, the legal mask and the action permutation, so the policy target moves with the board;
- loss = policy cross-entropy (over legal actions, on samples with a policy target) + weighted value MSE `Σ w (y − v)² / Σ w`;
- optimiser: AdamW (`lr` 1e-3, weight decay 1e-4), gradient clipping at 1.0; the learning rate drops ×0.1 at 60% and 85% of `iterations` (`lr_drops`).

**4. Evaluate and checkpoint.**

- solver test set: `value_sign_acc` (does the sign of v match the proven result?), per block-range accuracies, `value_mse`, and `won_top1_winning` (in won positions, is the network's top move a proven winning move? This is a lower bound, because the solver lists only the winning moves it proved within budget);
- every `arena_every` iterations: `arena_games` games against the previous snapshot and against uniform-prior MCTS (colours alternate);
- `latest.pkl` is saved (weights + optimiser state + history + newest 200k replay samples), along with `weights_XXXX.msgpack` every 25 iterations, and `progress.png` is updated.

#### The network

```
input  5×5×15 planes
stem   3×3 conv (channels) + BN + ReLU
body   `blocks` × residual block [3×3 conv, BN, ReLU, 3×3 conv, BN, skip, ReLU]
policy 1×1 conv (32) + BN + ReLU → dense → 153 logits   (128 move+build + 25 placement)
value  1×1 conv (16) + BN + ReLU → dense 128 + ReLU → dense 1 → tanh
```

Default 10 blocks × 128 channels ≈ 3.15M parameters. Batches are sharded over all local devices with parameters replicated, so multi-GPU works without changes.

### Configuration reference

Every field can be changed with `--set key=value`. The resolved config is written to `<run-dir>/config.json`.

| key | default | meaning |
|---|---|---|
| **network** | | |
| `blocks`, `channels` | 10, 128 | ResNet size. Checkpoints only load into the same size |
| **optimisation** | | |
| `batch_size` | 1024 | must be divisible by the device count |
| `lr`, `weight_decay` | 1e-3, 1e-4 | AdamW |
| `lambda_exact` | 3.0 | value-loss weight of solver-proven targets |
| `alpha_q` | 0.5 | `y = (1-α) z + α q` for unproven positions |
| `lr_drops` | (0.6, 0.85) | lr ×0.1 at these fractions of `iterations` (`--set "lr_drops=(0.5,0.8)"`) |
| **phase 0** | | |
| `p0_positions` | 60,000 | solver-labelled warm-start positions |
| `p0_min_blocks`, `p0_max_blocks` | 24, 55 | block-count range |
| `p0_root_budget`, `p0_child_budget` | 50,000, 1,000 | df-pn expansions per position / per candidate move |
| `p0_epochs` | 4 | |
| `p0_mix` | 0.1 | share of every self-play batch drawn from phase-0 data |
| **test set** | | |
| `test_positions` | 2,000 | |
| `test_min_blocks`, `test_max_blocks` | 20, 45 | |
| `test_root_budget`, `test_child_budget` | 200,000, 5,000 | |
| **self-play** | | |
| `n_games` | 512 | games in lock-step = inference batch size |
| `sims` | 200 | MCTS simulations per move |
| `temp_plies` | 8 | plies sampled by visit count before greedy play |
| `label_min_blocks` | 22 | exact root solve from this many blocks (26 on ≤4-core hosts) |
| `sp_root_budget`, `sp_child_budget` | 20,000, 2,000 | in-game solver budgets (10k / 1k on ≤4-core hosts) |
| `iterations` | 300 | self-play iterations |
| `games_per_iter` | 512 | |
| `sample_reuse` | 4.0 | training samples per new self-play sample |
| `replay_capacity` | 500,000 | |
| **evaluation** | | |
| `arena_games` | 128 | |
| `arena_every` | 5 | set very large (e.g. `1000000000`) to disable arenas |
| **system** | | |
| `label_threads` | CPUs − 1 | solver threads for data generation |
| `sp_threads` | min(16, CPUs − 1) | solver threads during self-play |
| `tt_gb` | 0.25 | transposition table per solver thread, GB (self-play caps it at 0.125) |
| `ckpt_dir` | runs/default | same as `--run-dir` |
| `seed` | 0 | |
| `log_every` | 20 | seconds between progress lines |

The biggest levers on wall-clock time are `sims`, `n_games` (larger batches use the GPU better) and the solver budgets. Memory use for data generation is about `label_threads × tt_gb` GB.

### Outputs, checkpoints and resuming

```
runs/default/
├── config.json             resolved config of the last launch
├── testset.npz             held-out solver test set
├── phase0.npz              phase-0 dataset
├── *_parts_c*/             per-chunk generation results (resume points)
├── phase0_state.msgpack    weights after phase 0
├── latest.pkl              newest self-play checkpoint (see below)
├── weights_0025.msgpack …  weights snapshot every 25 iterations
└── progress.png            losses, test-set accuracy, arena win rates
```

`latest.pkl` is a pickled dict:

| key | contents |
|---|---|
| `state` | Flax `TrainState` serialised with `flax.serialization.to_bytes` (params, batch stats, optimiser state) |
| `iter` | last completed iteration |
| `history` | list of per-iteration metric dicts (phase 0 entry first) |
| `cfg` | the config used |
| `backend` | `cpu` / `gpu` / `tpu` |
| `replay` | newest ≤200k replay samples (`state`, `pi`, `y`, `w`) |

**Resume rules** (in order, each step skips work that's already on disk):
1. `testset.npz` / `phase0.npz` are loaded if complete; otherwise generation resumes from the saved chunks.
2. If `latest.pkl` exists, phase 0 is skipped and phase 1 continues from `iter + 1`, with the replay buffer restored.
3. Otherwise `phase0_state.msgpack` is loaded if present, or phase 0 is trained.

**Continuing from the included checkpoint**

```bash
mkdir -p runs/continue
cp checkpoints/latest.pkl runs/continue/
python train.py --run-dir runs/continue        # keep blocks=10, channels=128
```

The test and phase-0 datasets are regenerated in the new run dir. To get a shorter phase-0 dataset, like the one the shipped model used, add `--set p0_positions=7000`.

### Reading the logs

Every line is stamped `[clock time | minutes since start]`. Lines from a `--smoke` run on a laptop:

```
[11:29:31 |    0.3 min] iter 1 self-play: done 16/16 games | 16 active | 116.0 moves/s | 1,697 net evals/s (559 rounds)
    | root solves 123 (119 proven) | proof moves 64 | time C++ 90% / net 10% | 5s
[11:29:31 |    0.3 min] iter 1: 421 new samples (21% solver-exact) from 16 games, first player won 69% | replay now 421
[11:29:31 |    0.3 min] iter 1 training: step 26/26 (100%) | loss 4.525 (policy 3.794, value 0.731) | top1 in target 0.44
    | lr 1.00e-03 | 222.3 steps/s, 14,225 samples/s | batch prep 15% of time | ETA 0s
[11:29:31 |    0.3 min] iter 1 test set: value_sign_acc 0.672 | won_top1_winning 0.349 | value_mse 0.839 | ...
[11:29:31 |    0.3 min] checkpoint saved: iteration 1, 2 MB, 421 replay samples, 0s
```

- **time C++ / net**: if C++ dominates, the solver budgets or `sims` are the bottleneck (more CPU cores help). If net dominates, the GPU is the bottleneck.
- **solver-exact %**: the share of samples with a proven value target. It rises as games get more decisive.
- **top1 in target**: how often the network's top move has non-zero mass in the training policy target.

---

## The included checkpoint

`checkpoints/latest.pkl` (≈96 MB, mostly the bundled replay buffer) was trained on a Google Colab GPU with `notebooks/Santorini_AlphaZero_GPU.ipynb`:

- 10 blocks × 128 channels (3.15M parameters);
- phase 0 on 7,000 solver-labelled positions, then **13 self-play iterations** of 512 games × 200 simulations;
- settings that differ from the defaults: `p0_positions=7000`, `label_threads=2`, `sp_threads=1`, `arena_every=1000000000` (arenas off), `label_min_blocks=26`, `sp_root_budget=10000`, `sp_child_budget=1000` (the 2-core Colab host).

To reproduce that setup locally:

```bash
python train.py --run-dir runs/colab_repro --set p0_positions=7000 --set arena_every=1000000000 \
    --set label_min_blocks=26 --set sp_root_budget=10000 --set sp_child_budget=1000
```

Held-out solver test set (2,000 positions):

| | after phase 0 | after iteration 13 |
|---|---|---|
| value sign accuracy | 0.765 | **0.873** |
| top move is a proven win (won positions) | 0.337 | **0.516** |
| value MSE | 0.940 | **0.404** |

It is an early checkpoint: the loop was set for 300 iterations, so continuing training should keep improving it.

---

## Tests

```bash
make test-cpp          # C++ correctness suite, ~1 min on an M-series Mac
make test-bindings     # Python binding checks, a few seconds
make smoke             # tiny end-to-end training run, ~1 min
```

The C++ suite cross-checks df-pn against a brute-force reference, forced-move pruning against all legal moves, symmetry hashing and the training labels, and ends with `ALL CHECKS PASSED`.

---

## Colab notebooks

`notebooks/` holds the original self-contained Colab notebooks. Each one writes the C++ sources with `%%writefile`, builds the module and stores checkpoints on Google Drive (`MyDrive/santorini_az`).

- `Santorini_AlphaZero_GPU.ipynb`: the training notebook this repo's `train.py` is derived from (includes the training log of the shipped checkpoint).
- `Santorini_Play.ipynb`: loads a checkpoint from Drive or an upload and opens the play widget inside Colab.

Use them if you'd rather train on a Colab GPU. Checkpoints are interchangeable with the local scripts.

---

## Technical details

**Board and state.** Cells 0–24 = `row*5 + col`, written `A1`…`E5` (column letter, row number, row 1 on top). A state is 29 bytes: 25 heights (0–3, 4 = dome), the two workers of the side to move (sorted), then the opponent's two. States are always from the side to move's perspective. The official piece supply is enforced (22/18/14 blocks, 18 domes).

**Actions.** 153 policy outputs: `64·k + 8·move_dir + build_dir` for worker k ∈ {0,1} and 8 directions each (128 move+build actions), plus 25 placement actions (reserved, always masked). A step up to level 3 wins immediately and has no build.

**Input planes (5×5×15).** One-hot height (5), my workers, opponent workers, cells where I / the opponent threaten to step up to level 3, buildable cells, and constant planes for blocks on the board and remaining level-1/2/3 pieces, plus a move-phase flag.

**Solver.** df-pn with the 1+ε threshold trick over a 4-way-bucket transposition table keyed by canonical (symmetry-reduced) Zobrist hashes. Santorini's game graph is a DAG (every turn adds a block), so there are no repetition or graph-history-interaction issues.

**Python API** (module `santorini`): `generate`, `legal_mask`, `encode`, `transform`, `pretty`, `all_moves`, `action_index`, `Config`/`SelfPlay` (batched self-play), `SearchParams`/`Search` (single-position search with introspection).

---

## License

MIT. See [LICENSE](LICENSE).
