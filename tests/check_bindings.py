"""Quick checks of the Python bindings (run after building the module):

    python tests/check_bindings.py

1. solver labels: values are +-1 and every proven winning move is legal;
2. symmetry: all 8 D4 transforms permute the legal mask consistently;
3. self-play: 8 short games with a uniform "network" produce valid policy/value targets.
"""
import time

import numpy as np

import santorini as S


def main():
    t = time.time()
    st, v, w = S.generate(64, 25, 50, root_budget=50_000, child_budget=5_000, threads=2, seed=99)
    legal = S.legal_mask(st)
    assert set(np.unique(v)) <= {-1, 1} and np.all(w <= legal) and np.all(w[v == 1].sum(1) > 0)
    print(f"1/3 labels ok ({time.time() - t:.1f}s): values in {{-1,+1}}, winning moves are legal")

    for s in range(8):
        syms = np.full(len(st), s, np.int32)
        planes, leg_t, perm = S.encode(st, syms)
        assert np.array_equal(leg_t, S.legal_mask(S.transform(st, syms)))
        for i in range(len(st)):
            assert np.array_equal(leg_t[i][perm[i]], legal[i])
    print("2/3 symmetries ok")

    cfg = S.Config(); cfg.n_games = 8; cfg.sims = 16; cfg.max_games = 8; cfg.threads = 2
    sp = S.SelfPlay(cfg)
    buf = np.zeros((8, 5, 5, S.NPLANES), np.float32)
    while True:
        n, gi, _ = sp.prepare(buf)
        if n == 0:
            break
        sp.apply(gi, np.zeros((n, S.NACT), np.float32), np.zeros(n, np.float32))
    d = sp.harvest()
    assert np.allclose(d["pi"].sum(1), 1, atol=1e-5) and set(np.unique(d["z"])) <= {-1, 1}
    print(f"3/3 self-play ok: {sp.games_finished} games, {len(d['z'])} samples")
    print(S.pretty(st[0]), "\nvalue for side to move:", v[0])


if __name__ == "__main__":
    main()
