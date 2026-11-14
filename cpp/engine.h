// Santorini (base game, no god powers) engine: bitboards, move generation,
// forced-move pruning, D4 symmetry and canonical Zobrist keys.
//
// Board: 25 cells, index = row*5 + col, row 0 = top (row "1"), col 0 = "A".
// Positions are stored from the side-to-move's perspective: `me` moves next.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <random>
#include <algorithm>

typedef uint32_t BB;
constexpr BB FULL = (1u << 25) - 1;
inline int lsb(BB b) { return __builtin_ctz(b); }
inline int pc(BB b) { return __builtin_popcount(b); }
inline BB bit(int c) { return 1u << c; }

// Official piece supply: 22 level-1, 18 level-2, 14 level-3 blocks, 18 domes.
constexpr int LIMIT1 = 22, LIMIT2 = 18, LIMIT3 = 14, LIMITD = 18;

struct Tables {
    BB adj[25];
    uint8_t sym[8][25];          // D4 symmetry: sym[s][cell] -> image cell
    uint64_t zh[25][5];          // height keys (zh[c][0] = 0)
    uint64_t zme[25], zop[25];   // worker keys (mover / opponent)
};
inline Tables T;

inline void initTables() {
    for (int c = 0; c < 25; c++) {
        int r = c / 5, k = c % 5;
        BB m = 0;
        for (int dr = -1; dr <= 1; dr++)
            for (int dk = -1; dk <= 1; dk++) {
                if (!dr && !dk) continue;
                int rr = r + dr, kk = k + dk;
                if (rr >= 0 && rr < 5 && kk >= 0 && kk < 5) m |= bit(rr * 5 + kk);
            }
        T.adj[c] = m;
        for (int s = 0; s < 8; s++) {
            int x = k, y = r;              // rotate s&3 times, then mirror if s>=4
            for (int i = 0; i < (s & 3); i++) { int nx = 4 - y, ny = x; x = nx; y = ny; }
            if (s >= 4) x = 4 - x;
            T.sym[s][c] = (uint8_t)(y * 5 + x);
        }
    }
    std::mt19937_64 rng(0x5a170421ULL);
    for (int c = 0; c < 25; c++) {
        T.zh[c][0] = 0;
        for (int h = 1; h < 5; h++) T.zh[c][h] = rng();
        T.zme[c] = rng();
        T.zop[c] = rng();
    }
}

struct Pos {
    BB h1 = 0, h2 = 0, h3 = 0, dm = 0;  // height >=1, >=2, >=3, dome
    BB me = 0, op = 0;                  // worker sets

    int height(int c) const {
        return ((h1 >> c) & 1) + ((h2 >> c) & 1) + ((h3 >> c) & 1) + ((dm >> c) & 1);
    }
    BB lvl(int l) const {
        switch (l) {
            case 0: return FULL & ~h1;
            case 1: return h1 & ~h2;
            case 2: return h2 & ~h3;
            case 3: return h3 & ~dm;
            default: return dm;
        }
    }
    int blocks() const { return pc(h1) + pc(h2) + pc(h3) + pc(dm); }
    void raise(int c) {
        BB b = bit(c);
        if (!(h1 & b)) h1 |= b;
        else if (!(h2 & b)) h2 |= b;
        else if (!(h3 & b)) h3 |= b;
        else dm |= b;
    }
    // Squares on which a block can currently be placed (supply limits respected).
    BB buildable() const {
        BB m = 0;
        if (pc(h1) < LIMIT1) m |= lvl(0);
        if (pc(h2) < LIMIT2) m |= lvl(1);
        if (pc(h3) < LIMIT3) m |= lvl(2);
        if (pc(dm) < LIMITD) m |= lvl(3);
        return m;
    }
};

struct Move { uint8_t f, t, b; };

// Squares a worker standing at height `lw` may step to (ignoring occupancy).
inline BB climbMask(const Pos& p, int lw) {
    if (lw == 0) return ~p.h2 & ~p.dm & FULL;
    if (lw == 1) return ~p.h3 & ~p.dm & FULL;
    return ~p.dm & FULL;
}

// Can the side owning `w` win right now by stepping from level 2 onto level 3?
inline bool canWinNow(const Pos& p, BB w) {
    BB occ = p.me | p.op, L2 = p.lvl(2), L3 = p.lvl(3);
    for (BB x = w & L2; x; x &= x - 1)
        if (T.adj[lsb(x)] & L3 & ~occ) return true;
    return false;
}

// Result of move generation with forced-move pruning.
//   win  : mover can win immediately (no need to look at moves)
//   n == 0 (and !win): mover is lost (no legal move, or every move hands the
//          opponent an immediate win)
// Otherwise m[0..n) are exactly the moves that do NOT let the opponent win
// on the very next step. Dropping the others is value-preserving: such a move
// loses, so the mover only needs them if nothing else exists (then: loss).
struct Gen {
    bool win;
    int n;
    Move m[128];
};

inline void genSafe(const Pos& p, Gen& g) {
    g.win = false;
    g.n = 0;
    if (canWinNow(p, p.me)) { g.win = true; return; }
    BB occ = p.me | p.op;
    BB L2 = p.lvl(2), L3 = p.lvl(3);
    BB oppAdj = 0;  // squares adjacent to an opponent worker standing on level 2
    for (BB x = p.op & L2; x; x &= x - 1) oppAdj |= T.adj[lsb(x)];
    BB danger2 = L2 & oppAdj;       // raising these to level 3 hands over a win
    BB bmAllowed = p.buildable();
    for (BB ws = p.me; ws; ws &= ws - 1) {
        int w = lsb(ws);
        BB dests = T.adj[w] & ~occ & climbMask(p, p.height(w));
        for (; dests; dests &= dests - 1) {
            int t = lsb(dests);
            BB occ2 = occ ^ bit(w) ^ bit(t);
            BB bm = T.adj[t] & ~occ2 & bmAllowed;
            BB threats = oppAdj & L3 & ~occ2;   // open level-3 squares the opponent can step onto
            BB safe;
            if (!threats) safe = bm & ~danger2;
            else if (!(threats & (threats - 1))) safe = bm & threats;  // must dome the single threat
            else safe = 0;                                              // two threats: cannot stop both
            for (; safe; safe &= safe - 1)
                g.m[g.n++] = Move{(uint8_t)w, (uint8_t)t, (uint8_t)lsb(safe)};
        }
    }
}

// Apply a move and flip perspective (opponent becomes the mover).
inline Pos makeChild(const Pos& p, Move m) {
    Pos c = p;
    c.raise(m.b);
    BB me2 = p.me ^ bit(m.f) ^ bit(m.t);
    c.me = p.op;
    c.op = me2;
    return c;
}

// ---- Symmetry-aware hashing ------------------------------------------------
// A Node carries the height hash under each of the 8 symmetries so a child's
// canonical key costs ~16 table lookups instead of re-hashing the board.
struct Node {
    Pos p;
    uint64_t hb[8];
};

inline Node makeRoot(const Pos& p) {
    Node n;
    n.p = p;
    for (int s = 0; s < 8; s++) {
        uint64_t h = 0;
        for (int c = 0; c < 25; c++) h ^= T.zh[T.sym[s][c]][p.height(c)];
        n.hb[s] = h;
    }
    return n;
}

inline uint64_t workersKey(int s, BB me, BB op) {
    uint64_t k = 0;
    for (BB x = me; x; x &= x - 1) k ^= T.zme[T.sym[s][lsb(x)]];
    for (BB x = op; x; x &= x - 1) k ^= T.zop[T.sym[s][lsb(x)]];
    return k;
}

inline uint64_t canonicalKey(const Node& n) {
    uint64_t best = ~0ULL;
    for (int s = 0; s < 8; s++) best = std::min(best, n.hb[s] ^ workersKey(s, n.p.me, n.p.op));
    return best;
}

inline Node childNode(const Node& n, Move m) {
    Node c;
    c.p = makeChild(n.p, m);
    int h0 = n.p.height(m.b);
    for (int s = 0; s < 8; s++) {
        int img = T.sym[s][m.b];
        c.hb[s] = n.hb[s] ^ T.zh[img][h0] ^ T.zh[img][h0 + 1];
    }
    return c;
}

// Fast canonical key of a child without building the full Node.
struct KeyCtx {
    uint64_t hb[8], meAsOp[8];  // child's opponent = parent's mover (moved worker added per move)
    uint64_t opAsMe[8];         // child's mover = parent's opponent (unchanged)
    void init(const Node& n) {
        for (int s = 0; s < 8; s++) {
            hb[s] = n.hb[s];
            opAsMe[s] = 0;
            for (BB x = n.p.op; x; x &= x - 1) opAsMe[s] ^= T.zme[T.sym[s][lsb(x)]];
        }
    }
    uint64_t childKey(const Node& n, Move m) const {
        int h0 = n.p.height(m.b);
        int other = lsb(n.p.me ^ bit(m.f));
        uint64_t best = ~0ULL;
        for (int s = 0; s < 8; s++) {
            const uint8_t* S = T.sym[s];
            uint64_t k = hb[s] ^ T.zh[S[m.b]][h0] ^ T.zh[S[m.b]][h0 + 1] ^ opAsMe[s] ^
                         T.zop[S[m.t]] ^ T.zop[S[other]];
            best = std::min(best, k);
        }
        return best;
    }
};

// ---- I/O helpers -------------------------------------------------------------
inline std::string sq(int c) {
    std::string s;
    s += char('A' + c % 5);
    s += char('1' + c / 5);
    return s;
}
inline int parseSq(const std::string& s) {
    return (s[1] - '1') * 5 + (toupper(s[0]) - 'A');
}
inline std::string moveStr(Move m) { return sq(m.f) + "-" + sq(m.t) + " build " + sq(m.b); }

// heights: 25 digits, row 1 (top) first. me/op: "D3,C5".
inline Pos parsePos(const std::string& heights, const std::string& me, const std::string& op) {
    Pos p;
    for (int c = 0; c < 25; c++)
        for (int i = 0; i < heights[c] - '0'; i++) p.raise(c);
    auto workers = [](const std::string& s) {
        BB b = 0;
        b |= bit(parseSq(s.substr(0, 2)));
        b |= bit(parseSq(s.substr(3, 2)));
        return b;
    };
    p.me = workers(me);
    p.op = workers(op);
    return p;
}

inline void printPos(const Pos& p, const char* meName = "X", const char* opName = "O") {
    printf("     A   B   C   D   E\n");
    for (int r = 0; r < 5; r++) {
        printf("  %d ", r + 1);
        for (int k = 0; k < 5; k++) {
            int c = r * 5 + k, h = p.height(c);
            char w = (p.me >> c & 1) ? meName[0] : (p.op >> c & 1) ? opName[0] : ' ';
            if (h == 4) printf(" [] ");
            else printf(" %d%c ", h, w);
        }
        printf("\n");
    }
    printf("  (%c = side to move, %c = opponent, [] = dome)\n", meName[0], opName[0]);
}