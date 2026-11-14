// Transposition table + depth-first proof-number search (df-pn) with the
// 1+epsilon threshold trick. Values are from the side-to-move's perspective:
//   pn == 0  -> mover wins (proven)      dn == 0 -> mover loses (proven)
// Santorini's game graph is a DAG (every turn adds a block), so there is no
// repetition / graph-history-interaction problem for df-pn.
#pragma once
#include "engine.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <vector>

constexpr uint32_t INF = 0x3FFFFFFF;
inline uint32_t sat(uint64_t x) { return x >= INF ? INF : (uint32_t)x; }

// ---- Transposition table -------------------------------------------------------
// 4-way buckets of 16-byte entries (one cache line). The 48-bit tag plus the
// bucket index together verify all 64 key bits. Replacement keeps the entries
// that cost the most work to compute; proven results get a large bonus.
struct TT {
    struct Entry {
        uint64_t tagWork;  // high 48 bits: key tag, low 16 bits: log2(work) * 1024
        uint32_t pn, dn;
    };
    Entry* t = nullptr;
    uint64_t nBuckets = 0, mask = 0;
    uint64_t stores = 0, used = 0;

    TT() = default;
    TT(const TT&) = delete;
    TT& operator=(const TT&) = delete;
    ~TT() { free(t); }

    void alloc(double gb) {
        free(t);
        t = nullptr;
        uint64_t bytes = (uint64_t)(gb * (1ULL << 30));
        nBuckets = 1;
        while (nBuckets * 2 * 64 <= bytes) nBuckets *= 2;
        mask = nBuckets - 1;
        t = (Entry*)aligned_alloc(64, nBuckets * 64);
        memset(t, 0, nBuckets * 64);
    }
    static uint64_t tagOf(uint64_t key) { return key & ~0xFFFFULL; }
    Entry* bucket(uint64_t key) { return t + (key & mask) * 4; }

    bool lookup(uint64_t key, uint32_t& pn, uint32_t& dn) {
        Entry* b = bucket(key);
        uint64_t tag = tagOf(key);
        for (int i = 0; i < 4; i++)
            if ((b[i].tagWork & ~0xFFFFULL) == tag && (b[i].pn | b[i].dn)) {
                pn = b[i].pn;
                dn = b[i].dn;
                return true;
            }
        return false;
    }
    static uint32_t prio(const Entry& e) {
        uint32_t w = e.tagWork & 0xFFFF;
        if (e.pn == 0 || e.dn == 0) w += 40000;  // keep proofs
        return w;
    }
    void store(uint64_t key, uint32_t pn, uint32_t dn, uint64_t work) {
        stores++;
        Entry* b = bucket(key);
        uint64_t tag = tagOf(key);
        uint32_t w16 = std::min<uint32_t>(0xFFFF, (uint32_t)(std::log2((double)work + 1) * 1024));
        Entry* victim = nullptr;
        for (int i = 0; i < 4; i++)
            if ((b[i].tagWork & ~0xFFFFULL) == tag && (b[i].pn | b[i].dn)) { victim = &b[i]; break; }
        if (!victim) {
            for (int i = 0; i < 4; i++)
                if (!(b[i].pn | b[i].dn)) { victim = &b[i]; used++; break; }
        }
        if (!victim) {
            victim = &b[0];
            for (int i = 1; i < 4; i++)
                if (prio(b[i]) < prio(*victim)) victim = &b[i];
        }
        victim->tagWork = tag | w16;
        victim->pn = pn;
        victim->dn = dn;
    }
    double fill() const { return (double)used / (nBuckets * 4); }
};

// ---- df-pn -------------------------------------------------------------------
struct Solver {
    TT tt;
    double eps = 0.25;
    uint64_t expansions = 0;
    uint64_t nextReport = 1ULL << 22;
    std::chrono::steady_clock::time_point start, deadline;
    bool aborted = false;
    bool verbose = true;
    uint64_t budget = 0;       // max expansions per solve() call; 0 = unlimited
    uint64_t budgetStart = 0;

    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    // Initial (pn, dn) for an unexplored child. Plain 1/1; children whose mover
    // is under an immediate threat get a smaller disproof estimate because
    // their reply is forced.
    int heuristic = 1;
    void initChild(const Pos& c, uint32_t& pn, uint32_t& dn) {
        pn = 1;
        dn = 1;
        if (!heuristic) return;
        // Threats the child's mover must answer (open level-3 squares next to an
        // opposing worker on level 2).
        BB occ = c.me | c.op, L2 = c.lvl(2), L3 = c.lvl(3);
        BB adj = 0;
        for (BB x = c.op & L2; x; x &= x - 1) adj |= T.adj[lsb(x)];
        int threats = pc(adj & L3 & ~occ);
        if (threats >= 2) { pn = 4; dn = 1; return; }
        if (threats == 1) { pn = 2; dn = 1; return; }
        int d = 0;
        for (BB x = c.me; x; x &= x - 1) d += c.height(lsb(x));
        for (BB x = c.op; x; x &= x - 1) d -= c.height(lsb(x));
        if (d > 0) { pn = 1; dn = 1 + std::min(d, 3); }
        else if (d < 0) { pn = 1 + std::min(-d, 3); dn = 1; }
    }

    uint64_t mid(const Node& n, uint64_t key, uint32_t thpn, uint32_t thdn, uint32_t& outPn,
                 uint32_t& outDn) {
        expansions++;
        if (budget && expansions - budgetStart >= budget) aborted = true;  // per-position cap
        if (expansions >= nextReport) {
            nextReport += 1ULL << 22;
            if (verbose)
                fprintf(stderr, "  [%.0fs] %llu expansions, %.1f M/s, TT fill %.1f%%\n", elapsed(),
                        (unsigned long long)expansions, expansions / elapsed() / 1e6,
                        100 * tt.fill());
            if (std::chrono::steady_clock::now() > deadline) aborted = true;
        }
        Gen g;
        genSafe(n.p, g);
        if (g.win) { outPn = 0; outDn = INF; tt.store(key, 0, INF, 1); return 1; }
        if (g.n == 0) { outPn = INF; outDn = 0; tt.store(key, INF, 0, 1); return 1; }

        KeyCtx kc;
        kc.init(n);
        uint64_t ck[128];
        uint32_t cpn[128], cdn[128];
        for (int i = 0; i < g.n; i++) {
            ck[i] = kc.childKey(n, g.m[i]);
            if (!tt.lookup(ck[i], cpn[i], cdn[i])) initChild(makeChild(n.p, g.m[i]), cpn[i], cdn[i]);
        }
        uint64_t work = 1;
        uint32_t pn, dn;
        for (;;) {
            // pn(node) = min over children of dn(child); dn(node) = sum of pn(child)
            uint32_t best = INF + 1, second = INF;
            int bi = -1;
            uint64_t sum = 0;
            for (int i = 0; i < g.n; i++) {
                sum += cpn[i];
                if (cdn[i] < best) { second = best; best = cdn[i]; bi = i; }
                else if (cdn[i] < second) second = cdn[i];
            }
            pn = best;
            dn = sat(sum);
            if (second > INF) second = INF;
            if (pn >= thpn || dn >= thdn || aborted) break;
            uint32_t cthpn = sat((uint64_t)thdn - dn + cpn[bi]);
            uint64_t widened = std::max<uint64_t>((uint64_t)second + 1,
                                                  (uint64_t)std::ceil(second * (1.0 + eps)));
            uint32_t cthdn = (uint32_t)std::min<uint64_t>(thpn, widened);
            Node c = childNode(n, g.m[bi]);
            work += mid(c, ck[bi], cthpn, cthdn, cpn[bi], cdn[bi]);
        }
        tt.store(key, pn, dn, work);
        outPn = pn;
        outDn = dn;
        return work;
    }

    // Returns +1 mover wins, -1 mover loses, 0 unresolved (time limit).
    int solve(const Pos& p, double seconds, uint32_t* pnOut = nullptr, uint32_t* dnOut = nullptr) {
        start = std::chrono::steady_clock::now();
        deadline = start + std::chrono::milliseconds((int64_t)(seconds * 1000));
        aborted = false;
        budgetStart = expansions;
        Node root = makeRoot(p);
        uint64_t key = canonicalKey(root);
        uint32_t pn, dn;
        mid(root, key, INF, INF, pn, dn);
        if (pnOut) *pnOut = pn;
        if (dnOut) *dnOut = dn;
        return pn == 0 ? 1 : dn == 0 ? -1 : 0;
    }

    // Training label for one position.
    //   value: +1 mover wins, -1 mover loses, 0 unresolved within rootBudget expansions.
    //   winning: when value == +1, every move proven to win. An immediate step-up
    //   win has no build; it is encoded with b = f (build back on the vacated cell),
    //   so it maps to a fixed (worker, move dir, build dir) action index.
    // Children are checked with childBudget each; a child left unresolved is simply
    // not listed, so `winning` is a subset of the true winning set (never wrong).
    struct Label {
        int value;
        std::vector<Move> winning;
    };
    Label label(const Pos& p, uint64_t rootBudget, uint64_t childBudget) {
        Label L{0, {}};
        bool v0 = verbose;
        verbose = false;
        Gen g;
        genSafe(p, g);
        if (g.win) {
            L.value = 1;
            BB occ = p.me | p.op, L3 = p.lvl(3);
            for (BB x = p.me & p.lvl(2); x; x &= x - 1)
                for (BB t = T.adj[lsb(x)] & L3 & ~occ; t; t &= t - 1)
                    L.winning.push_back(Move{(uint8_t)lsb(x), (uint8_t)lsb(t), (uint8_t)lsb(x)});
            verbose = v0;
            return L;
        }
        budget = rootBudget;
        L.value = solve(p, 1e9);
        if (L.value == 1) {
            budget = childBudget;
            for (int i = 0; i < g.n; i++) {
                Pos c = makeChild(p, g.m[i]);
                uint32_t pn, dn;
                bool known = tt.lookup(canonicalKey(makeRoot(c)), pn, dn);
                if ((known && dn == 0) || ((!known || pn != 0) && solve(c, 1e9) == -1))
                    L.winning.push_back(g.m[i]);
            }
        }
        budget = 0;
        verbose = v0;
        return L;
    }

    // Value of a position from the TT, solving it (re-proving) if it fell out.
    int value(const Pos& p, double seconds = 1e9) {
        Node n = makeRoot(p);
        uint32_t pn, dn;
        if (tt.lookup(canonicalKey(n), pn, dn) && (pn == 0 || dn == 0)) return pn == 0 ? 1 : -1;
        bool v = verbose;
        verbose = false;
        int r = solve(p, seconds);
        verbose = v;
        return r;
    }
};