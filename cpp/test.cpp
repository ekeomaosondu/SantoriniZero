// Correctness checks:
//  1. canonical key is invariant under all 8 board symmetries
//  2. forced-move pruning never drops a non-losing move (vs. full move list)
//  3. df-pn agrees with an independent, unpruned brute-force minimax on random
//     mid/late-game positions
//  4. the screenshot position from the conversation is a loss for the mover
#include "solver.h"

// ---- Independent reference implementation (plain arrays, no pruning) ----------
struct Ref {
    int h[25];
    int w[2][2];  // w[0] = side to move
};
static bool refOcc(const Ref& r, int c) {
    return r.w[0][0] == c || r.w[0][1] == c || r.w[1][0] == c || r.w[1][1] == c;
}
static bool adjacent(int a, int b) {
    return a != b && abs(a / 5 - b / 5) <= 1 && abs(a % 5 - b % 5) <= 1;
}
static int refCount(const Ref& r, int lvl) {  // pieces of level `lvl` in use
    int n = 0;
    for (int c = 0; c < 25; c++) n += r.h[c] >= lvl;
    return n;
}
static bool refCanBuild(const Ref& r, int c) {
    static const int lim[5] = {0, LIMIT1, LIMIT2, LIMIT3, LIMITD};
    return r.h[c] < 4 && refCount(r, r.h[c] + 1) < lim[r.h[c] + 1];
}
// +1 mover wins, -1 loses, 0 undecided within `d` plies.
static int brute(Ref& r, int d) {
    bool any = false, unk = false;
    for (int k = 0; k < 2; k++) {
        int f = r.w[0][k];
        for (int t = 0; t < 25; t++) {
            if (!adjacent(f, t) || refOcc(r, t) || r.h[t] == 4 || r.h[t] > r.h[f] + 1) continue;
            if (r.h[f] == 2 && r.h[t] == 3) return 1;  // moving up to level 3 wins
            r.w[0][k] = t;
            for (int b = 0; b < 25; b++) {
                if (!adjacent(t, b) || refOcc(r, b) || !refCanBuild(r, b)) continue;
                any = true;
                if (d == 0) { unk = true; continue; }
                r.h[b]++;
                Ref c = r;
                std::swap(c.w[0][0], c.w[1][0]);
                std::swap(c.w[0][1], c.w[1][1]);
                int v = -brute(c, d - 1);
                r.h[b]--;
                if (v == 1) { r.w[0][k] = f; return 1; }
                if (v == 0) unk = true;
            }
            r.w[0][k] = f;
        }
    }
    if (!any) return -1;  // no legal move+build: lose
    return unk ? 0 : -1;
}
static Ref toRef(const Pos& p) {
    Ref r;
    for (int c = 0; c < 25; c++) r.h[c] = p.height(c);
    int i = 0;
    for (BB x = p.me; x; x &= x - 1) r.w[0][i++] = lsb(x);
    i = 0;
    for (BB x = p.op; x; x &= x - 1) r.w[1][i++] = lsb(x);
    return r;
}

static Pos transform(const Pos& p, int s) {
    Pos q;
    for (int c = 0; c < 25; c++)
        for (int i = 0; i < p.height(c); i++) q.raise(T.sym[s][c]);
    for (BB x = p.me; x; x &= x - 1) q.me |= bit(T.sym[s][lsb(x)]);
    for (BB x = p.op; x; x &= x - 1) q.op |= bit(T.sym[s][lsb(x)]);
    return q;
}

// Random game from a random placement; returns false if the game ended first.
static bool randomPosition(std::mt19937_64& rng, int blocks, Pos& out) {
    Pos p;
    int cells[25];
    for (int i = 0; i < 25; i++) cells[i] = i;
    std::shuffle(cells, cells + 25, rng);
    p.me = bit(cells[0]) | bit(cells[1]);
    p.op = bit(cells[2]) | bit(cells[3]);
    while (p.blocks() < blocks) {
        if (canWinNow(p, p.me)) return false;
        // all legal moves (unpruned)
        std::vector<Move> ms;
        BB occ = p.me | p.op, bmAllowed = p.buildable();
        for (BB ws = p.me; ws; ws &= ws - 1) {
            int w = lsb(ws);
            for (BB d = T.adj[w] & ~occ & climbMask(p, p.height(w)); d; d &= d - 1) {
                int t = lsb(d);
                for (BB b = T.adj[t] & ~(occ ^ bit(w) ^ bit(t)) & bmAllowed; b; b &= b - 1)
                    ms.push_back(Move{(uint8_t)w, (uint8_t)t, (uint8_t)lsb(b)});
            }
        }
        if (ms.empty()) return false;
        p = makeChild(p, ms[rng() % ms.size()]);
    }
    if (canWinNow(p, p.me)) return false;
    out = p;
    return true;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    initTables();
    std::mt19937_64 rng(12345);
    int fails = 0;

    auto t0 = std::chrono::steady_clock::now();
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    // 1. symmetry invariance
    printf("[1/5] symmetry + incremental hashing on 2000 random positions...\n");
    for (int i = 0; i < 2000; i++) {
        Pos p;
        if (!randomPosition(rng, 5 + rng() % 40, p)) continue;
        uint64_t k = canonicalKey(makeRoot(p));
        for (int s = 0; s < 8; s++)
            if (canonicalKey(makeRoot(transform(p, s))) != k) { fails++; break; }
        // incremental child keys == keys computed from scratch
        Gen g;
        genSafe(p, g);
        Node n = makeRoot(p);
        KeyCtx kc;
        kc.init(n);
        for (int j = 0; j < g.n; j++) {
            Node c = childNode(n, g.m[j]);
            if (kc.childKey(n, g.m[j]) != canonicalKey(c) ||
                canonicalKey(c) != canonicalKey(makeRoot(c.p))) { fails++; break; }
        }
    }
    printf("      symmetry/hash checks: %s (%.1fs)\n", fails ? "FAIL" : "ok", secs());
    printf("[2-3/5] df-pn vs brute force and pruning exactness on 300 positions...\n");

    // 2+3. df-pn vs brute force
    Solver S;
    S.verbose = false;
    S.tt.alloc(0.25);
    int agree = 0, decided = 0, tested = 0, prunedOk = 0;
    for (int i = 0; tested < 300; i++) {
        Pos p;
        int blocks = 34 + rng() % 30;
        if (!randomPosition(rng, blocks, p)) continue;
        tested++;
        if (tested % 50 == 0) printf("      %d/300 positions checked, %d mismatches so far (%.0fs)\n", tested, fails, secs());
        Ref r = toRef(p);
        int bf = brute(r, blocks > 45 ? 5 : 3);
        int dp = S.solve(p, 5);
        if (dp == 0) { printf("df-pn unresolved in 5s at %d blocks\n", blocks); }
        if (bf != 0) {
            decided++;
            if (bf == dp) agree++;
            else { fails++; printf("MISMATCH brute=%d dfpn=%d\n", bf, dp); printPos(p); }
        }
        // pruning check: every unpruned legal move that is NOT in the safe list must lose
        // in one ply (opponent can step up), confirmed with the reference implementation.
        Gen g;
        genSafe(p, g);
        bool ok = true;
        BB occ = p.me | p.op, bmAllowed = p.buildable();
        for (BB ws = p.me; ws; ws &= ws - 1) {
            int w = lsb(ws);
            for (BB d = T.adj[w] & ~occ & climbMask(p, p.height(w)); d; d &= d - 1) {
                int t = lsb(d);
                for (BB b = T.adj[t] & ~(occ ^ bit(w) ^ bit(t)) & bmAllowed; b; b &= b - 1) {
                    Move m{(uint8_t)w, (uint8_t)t, (uint8_t)lsb(b)};
                    bool inSafe = false;
                    for (int j = 0; j < g.n; j++)
                        if (g.m[j].f == m.f && g.m[j].t == m.t && g.m[j].b == m.b) inSafe = true;
                    Ref c = toRef(makeChild(p, m));
                    bool oppWinsNow = brute(c, 0) == 1;
                    if (inSafe == oppWinsNow) ok = false;
                }
            }
        }
        if (ok) prunedOk++;
        else { fails++; printf("PRUNING MISMATCH\n"); printPos(p); }
    }
    printf("df-pn vs brute force: %d/%d decided positions agree (%d tested)\n", agree, decided,
           tested);
    printf("forced-move pruning exact on %d/%d positions\n", prunedOk, tested);

    // 4. the screenshot position (Green to move): expect a loss
    printf("[4/5] screenshot position...\n");
    Pos shot = parsePos("0003100011000010300022110", "D3,C5", "C2,B5");
    int v = S.solve(shot, 60);
    printf("screenshot position, Green to move: %s\n", v == -1 ? "LOSS (expected)" : "UNEXPECTED");
    if (v != -1) fails++;

    // 5. training labels: every listed winning move must leave the opponent lost
    //    (checked by brute force where decisive, else by a fresh unbudgeted solve),
    //    and a won position must list at least one winning move.
    printf("[5/5] training labels on 150 won positions...\n");
    int labelled = 0, labelOk = 0;
    for (int i = 0; labelled < 150; i++) {
        Pos p;
        int blocks = 30 + rng() % 30;
        if (!randomPosition(rng, blocks, p)) continue;
        Solver::Label L = S.label(p, 200000, 20000);
        if (L.value != 1) continue;
        labelled++;
        if (labelled % 50 == 0) printf("      %d/150 labelled positions checked (%.0fs)\n", labelled, secs());
        bool ok = !L.winning.empty();
        for (Move m : L.winning) {
            if (p.height(m.f) == 2 && p.height(m.t) == 3) continue;  // immediate step-up win
            Ref c = toRef(makeChild(p, m));
            int bf = brute(c, 3);
            int ex = bf != 0 ? bf : S.solve(makeChild(p, m), 60);
            if (ex != -1) printf("  move %s: brute=%d solve=%d\n", moveStr(m).c_str(), bf, ex);
            ok &= ex == -1;
        }
        if (ok) labelOk++;
        else { fails++; printf("LABEL MISMATCH\n"); printPos(p); }
    }
    printf("training labels: winning-move sets correct on %d/%d won positions\n", labelOk, labelled);

    printf("%s (%.0fs total)\n", fails ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED", secs());
    return fails != 0;
}