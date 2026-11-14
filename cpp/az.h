// AlphaZero layer on top of engine.h / solver.h:
//   * compact state <-> Pos, action encoding, D4 action permutations
//   * network input planes (NHWC, 5x5x15)
//   * threaded solver labelling and Phase-0 position generation
//   * SelfPlay: many games in lock-step; each prepare() call returns one leaf per
//     game for the network, apply() expands/backs up. Uses forced-move pruning,
//     MCTS-Solver proof propagation and an exact df-pn root solve late in the game.
#pragma once
#include "solver.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace az {

constexpr int NPLANES = 15;
constexpr int NMOVE = 128;    // move+build actions: 64*k + 8*dirMove + dirBuild
constexpr int NACT = 153;     // + 25 placement actions (reserved, always masked for now)
constexpr int STATE_LEN = 29; // 25 heights, me0 < me1, op0 < op1 (cell indices)

const int DR[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
const int DC[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

struct Tabs {
    int dirOf[25][25];  // direction index from a to b, -1 if not adjacent
    int dirMap[8][8];   // dirMap[s][d]: direction d after symmetry s
};
inline Tabs A;

inline void init() {
    initTables();
    for (int a = 0; a < 25; a++)
        for (int b = 0; b < 25; b++) {
            A.dirOf[a][b] = -1;
            for (int d = 0; d < 8; d++)
                if (a / 5 + DR[d] == b / 5 && a % 5 + DC[d] == b % 5) A.dirOf[a][b] = d;
        }
    for (int s = 0; s < 8; s++)
        for (int d = 0; d < 8; d++) {
            int c = 12, n = (2 + DR[d]) * 5 + (2 + DC[d]);
            A.dirMap[s][d] = A.dirOf[T.sym[s][c]][T.sym[s][n]];
        }
}

// ---- state conversion ----------------------------------------------------------
inline Pos fromState(const int8_t* s) {
    Pos p;
    for (int c = 0; c < 25; c++)
        for (int i = 0; i < s[c]; i++) p.raise(c);
    p.me = bit(s[25]) | bit(s[26]);
    p.op = bit(s[27]) | bit(s[28]);
    return p;
}
inline void toState(const Pos& p, int8_t* s) {
    for (int c = 0; c < 25; c++) s[c] = (int8_t)p.height(c);
    s[25] = (int8_t)lsb(p.me);
    s[26] = (int8_t)lsb(p.me & (p.me - 1));
    s[27] = (int8_t)lsb(p.op);
    s[28] = (int8_t)lsb(p.op & (p.op - 1));
}
inline Pos transformPos(const Pos& p, int s) {
    Pos q;
    for (int c = 0; c < 25; c++)
        for (int i = 0; i < p.height(c); i++) q.raise(T.sym[s][c]);
    for (BB x = p.me; x; x &= x - 1) q.me |= bit(T.sym[s][lsb(x)]);
    for (BB x = p.op; x; x &= x - 1) q.op |= bit(T.sym[s][lsb(x)]);
    return q;
}

// ---- actions ---------------------------------------------------------------------
// Worker k = 0 is the mover's worker on the lower cell index. An immediate step-up
// win has no build; it is encoded with build = the vacated cell (b = f).
inline int encodeMove(const Pos& p, Move m) {
    int k = (m.f == lsb(p.me)) ? 0 : 1;
    return 64 * k + 8 * A.dirOf[m.f][m.t] + A.dirOf[m.t][m.b];
}
// perm[a] = index of action a after applying symmetry s to position p.
inline void actionPerm(const Pos& p, int s, int32_t* perm) {
    int w0 = lsb(p.me), w1 = lsb(p.me & (p.me - 1));
    int nk0 = T.sym[s][w0] < T.sym[s][w1] ? 0 : 1;
    int nk[2] = {nk0, 1 - nk0};
    for (int k = 0; k < 2; k++)
        for (int dm = 0; dm < 8; dm++)
            for (int db = 0; db < 8; db++)
                perm[64 * k + 8 * dm + db] = 64 * nk[k] + 8 * A.dirMap[s][dm] + A.dirMap[s][db];
}
// Legal (safe) actions. All zero means the mover is lost.
inline int legalMask(const Pos& p, uint8_t* mask) {
    memset(mask, 0, NMOVE);
    Gen g;
    genSafe(p, g);
    int n = 0;
    if (g.win) {
        BB occ = p.me | p.op, L3 = p.lvl(3);
        for (BB x = p.me & p.lvl(2); x; x &= x - 1)
            for (BB t = T.adj[lsb(x)] & L3 & ~occ; t; t &= t - 1) {
                mask[encodeMove(p, Move{(uint8_t)lsb(x), (uint8_t)lsb(t), (uint8_t)lsb(x)})] = 1;
                n++;
            }
        return n;
    }
    for (int i = 0; i < g.n; i++) mask[encodeMove(p, g.m[i])] = 1;
    return g.n;
}
// Decode an action index into a Move (no legality check).
inline Move decode(const Pos& p, int a) {
    int k = a / 64, dm = (a / 8) % 8, db = a % 8;
    int f = k == 0 ? lsb(p.me) : lsb(p.me & (p.me - 1));
    int t = f + DR[dm] * 5 + DC[dm];
    int b = t + DR[db] * 5 + DC[db];
    return Move{(uint8_t)f, (uint8_t)t, (uint8_t)b};
}

// ---- input planes (NHWC: out[cell * NPLANES + plane]) ------------------------------
inline BB threatCells(const Pos& p, BB workers) {
    BB occ = p.me | p.op, adj = 0;
    for (BB x = workers & p.lvl(2); x; x &= x - 1) adj |= T.adj[lsb(x)];
    return adj & p.lvl(3) & ~occ;
}
inline void encodePlanes(const Pos& p, float* out) {
    memset(out, 0, sizeof(float) * 25 * NPLANES);
    BB myT = threatCells(p, p.me), opT = threatCells(p, p.op);
    BB build = p.buildable() & ~(p.me | p.op);
    float sc[5] = {p.blocks() / 68.0f, (LIMIT1 - pc(p.h1)) / 22.0f, (LIMIT2 - pc(p.h2)) / 18.0f,
                   (LIMIT3 - pc(p.h3)) / 14.0f, 1.0f /* move phase */};
    for (int c = 0; c < 25; c++) {
        float* o = out + c * NPLANES;
        o[p.height(c)] = 1;
        o[5] = (p.me >> c) & 1;
        o[6] = (p.op >> c) & 1;
        o[7] = (myT >> c) & 1;
        o[8] = (opT >> c) & 1;
        o[9] = (build >> c) & 1;
        for (int i = 0; i < 5; i++) o[10 + i] = sc[i];
    }
}

// ---- random positions --------------------------------------------------------------
inline Pos randomPlacement(std::mt19937_64& rng) {
    int cells[25];
    for (int i = 0; i < 25; i++) cells[i] = i;
    std::shuffle(cells, cells + 25, rng);
    Pos p;
    p.me = bit(cells[0]) | bit(cells[1]);
    p.op = bit(cells[2]) | bit(cells[3]);
    return p;
}
// Random safe play to exactly `blocks` blocks; false if the game ended first.
inline bool randomPosition(std::mt19937_64& rng, int blocks, Pos& out) {
    Pos p = randomPlacement(rng);
    for (;;) {
        Gen g;
        genSafe(p, g);
        if (g.win || g.n == 0) return false;
        if (p.blocks() >= blocks) { out = p; return true; }
        p = makeChild(p, g.m[rng() % g.n]);
    }
}

// ---- threaded labelling ---------------------------------------------------------------
struct LabelOut {
    int8_t value;       // +1 / -1 / 0 (unresolved)
    uint8_t win[NMOVE]; // proven winning actions (subset of the true set)
};
inline void labelOne(Solver& S, const Pos& p, uint64_t rootBudget, uint64_t childBudget,
                     LabelOut& out) {
    Solver::Label L = S.label(p, rootBudget, childBudget);
    out.value = (int8_t)L.value;
    memset(out.win, 0, NMOVE);
    for (Move m : L.winning) out.win[encodeMove(p, m)] = 1;
}
inline std::unique_ptr<Solver> makeSolver(double ttGb) {
    auto S = std::make_unique<Solver>();
    S->verbose = false;
    S->tt.alloc(ttGb);
    return S;
}
inline void labelBatch(const std::vector<Pos>& ps, uint64_t rootBudget, uint64_t childBudget,
                       int threads, double ttGb, std::vector<LabelOut>& out) {
    out.resize(ps.size());
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < threads; t++)
        th.emplace_back([&]() {
            auto S = makeSolver(ttGb);
            for (size_t i; (i = next++) < ps.size();) labelOne(*S, ps[i], rootBudget, childBudget, out[i]);
        });
    for (auto& t : th) t.join();
}
// Phase 0: n resolved positions with blocks uniform in [minB, maxB].
inline void generateLabeled(int n, int minB, int maxB, uint64_t rootBudget, uint64_t childBudget,
                            int threads, double ttGb, uint64_t seed, std::vector<Pos>& ps,
                            std::vector<LabelOut>& out) {
    ps.resize(n);
    out.resize(n);
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < threads; t++)
        th.emplace_back([&, t]() {
            auto S = makeSolver(ttGb);
            std::mt19937_64 rng(seed * 1000003 + t);
            for (;;) {
                if (next.load() >= n) return;
                Pos p;
                int target = minB + (int)(rng() % (maxB - minB + 1));
                if (!randomPosition(rng, target, p)) continue;
                LabelOut lo;
                labelOne(*S, p, rootBudget, childBudget, lo);
                if (lo.value == 0) continue;
                int i = next++;
                if (i >= n) return;
                ps[i] = p;
                out[i] = lo;
            }
        });
    for (auto& t : th) t.join();
}

// ---- self-play MCTS ---------------------------------------------------------------------
struct Config {
    int nGames = 256;           // games played in lock-step (= inference batch)
    int sims = 200;             // simulations per move
    float cpuct = 1.5f;
    float dirAlpha = 0.3f, dirFrac = 0.25f;
    bool noise = true;          // Dirichlet noise at the root (off for arena games)
    int tempPlies = 8;          // sample moves proportional to visits for this many plies
    int labelMinBlocks = 22;    // exact root solve (and label) from this many blocks
    uint64_t rootBudget = 20000, childBudget = 2000;
    int threads = 4;            // solver threads
    double ttGb = 0.125;        // TT per solver thread
    int maxGames = 1 << 30;     // stop starting new games after this many
    bool arena = false;         // two networks: net id per side, colours alternate
    uint64_t seed = 1;
};

struct Sample {
    int8_t state[STATE_LEN];
    float pi[NMOVE];
    float q;          // root value estimate (mover's view)
    int8_t exact;     // 1 if v is a proven value
    int8_t v;         // proven value (+1/-1) when exact
    int8_t player;    // 0 = first player, 1 = second player
    float z;          // final outcome for `player`, filled at game end
};

struct Edge {
    Move m;
    int16_t a;        // action index
    int child = -1;
    float P = 0, W = 0;  // W summed from the parent's mover perspective
    int N = 0;
};
struct MNode {
    Pos p;
    int first = 0, n = 0;  // edges [first, first+n)
    int parent = -1, pedge = -1;
    int N = 0;
    int8_t status = 0;     // +1 mover wins, -1 mover loses (proven)
    bool expanded = false;
};

class SelfPlay {
   public:
    Config cfg;
    explicit SelfPlay(const Config& c) : cfg(c), games(c.nGames) {
        rng.seed(c.seed);
        for (int t = 0; t < cfg.threads; t++) solvers.push_back(makeSolver(cfg.ttGb));
        for (int i = 0; i < cfg.nGames; i++) startGame(i);
    }

    // Advance all games until each needs a network evaluation (or is finished).
    // Fills planes[count][25][NPLANES], gameIdx[count], netId[count]; returns count.
    int prepare(float* planes, int32_t* gameIdx, int32_t* netId) {
        for (;;) {
            std::vector<int> needLabel;
            for (int i = 0; i < (int)games.size(); i++) {
                int st = advance(i);
                if (st == NEED_LABEL) needLabel.push_back(i);
            }
            if (needLabel.empty()) break;
            runLabels(needLabel);
            rootProven = provenCount.load();
        }
        int cnt = 0;
        for (int i = 0; i < (int)games.size(); i++) {
            Game& g = games[i];
            if (!g.active || g.leaf < 0) continue;
            encodePlanes(g.nodes[g.leaf].p, planes + (size_t)cnt * 25 * NPLANES);
            gameIdx[cnt] = i;
            netId[cnt] = g.netOf[g.ply & 1];
            cnt++;
        }
        return cnt;
    }

    // logits[count][NACT], values[count] for the leaves returned by prepare().
    void apply(int count, const int32_t* gameIdx, const float* logits, const float* values) {
        leafEvals += count;
        for (int j = 0; j < count; j++) {
            Game& g = games[gameIdx[j]];
            int leaf = g.leaf;
            expand(g, leaf, logits + (size_t)j * NACT);
            g.leaf = -1;
            backup(g, leaf, values[j]);
        }
    }

    // Finished-game samples since the last call.
    std::vector<Sample> harvest() {
        std::vector<Sample> out;
        out.swap(done);
        return out;
    }
    int gamesStarted() const { return started; }
    size_t pendingSamples() const { return done.size(); }
    // counters for logging
    long movesPlayed = 0, rootSolves = 0, rootProven = 0, solverMoves = 0, leafEvals = 0;
    int gamesFinished() const { return finished; }
    int activeGames() const {
        int a = 0;
        for (auto& g : games) a += g.active;
        return a;
    }
    // Arena results: wins by net 0 / net 1, and first-player wins.
    int netWins[2] = {0, 0};
    int firstPlayerWins = 0;
    std::vector<int> gameLengths;

   private:
    enum { NEED_EVAL, NEED_LABEL, IDLE };
    struct Game {
        bool active = false;
        Pos pos;
        int ply = 0;
        int netOf[2] = {0, 0};  // network id playing first / second player
        std::vector<MNode> nodes;
        std::vector<Edge> edges;
        int leaf = -1;
        bool labelled = false;
        LabelOut label{};
        std::vector<Sample> samples;
    };
    std::vector<Game> games;
    std::vector<std::unique_ptr<Solver>> solvers;
    std::vector<Sample> done;
    std::mt19937_64 rng;
    int started = 0, finished = 0;
    std::atomic<long> provenCount{0};

    void startGame(int i) {
        Game& g = games[i];
        g = Game();
        if (started >= cfg.maxGames) return;
        g.active = true;
        g.pos = randomPlacement(rng);
        if (cfg.arena) {  // alternate colours between the two networks
            g.netOf[0] = started & 1;
            g.netOf[1] = 1 - g.netOf[0];
        }
        started++;
    }

    void finishGame(int i, int winnerPlayer) {
        Game& g = games[i];
        for (Sample& s : g.samples) {
            s.z = s.player == winnerPlayer ? 1.0f : -1.0f;
            done.push_back(s);
        }
        netWins[g.netOf[winnerPlayer]]++;
        firstPlayerWins += winnerPlayer == 0;
        gameLengths.push_back(g.ply);
        finished++;
        startGame(i);
    }

    void record(Game& g, const float* pi, float q, int exact, int v) {
        if (!cfg.arena) {
            Sample s;
            toState(g.pos, s.state);
            memcpy(s.pi, pi, sizeof(s.pi));
            s.q = q;
            s.exact = (int8_t)exact;
            s.v = (int8_t)v;
            s.player = (int8_t)(g.ply & 1);
            s.z = 0;
            g.samples.push_back(s);
        }
    }

    // Play move m in game i; returns true if the game ended.
    bool play(int i, Move m) {
        movesPlayed++;
        Game& g = games[i];
        int mover = g.ply & 1;
        if (g.pos.height(m.f) == 2 && g.pos.height(m.t) == 3) {  // step up: win
            g.ply++;
            finishGame(i, mover);
            return true;
        }
        g.pos = makeChild(g.pos, m);
        g.ply++;
        g.nodes.clear();
        g.edges.clear();
        g.labelled = false;
        return false;
    }

    void runLabels(const std::vector<int>& idx) {
        rootSolves += (long)idx.size();
        std::atomic<size_t> next{0};
        std::vector<std::thread> th;
        int nt = std::min<int>(cfg.threads, (int)idx.size());
        for (int t = 0; t < nt; t++)
            th.emplace_back([&, t]() {
                for (size_t k; (k = next++) < idx.size();) {
                    Game& g = games[idx[k]];
                    labelOne(*solvers[t], g.pos, cfg.rootBudget, cfg.childBudget, g.label);
                    g.labelled = true;
                    if (g.label.value != 0) provenCount++;
                }
            });
        for (auto& t : th) t.join();
    }

    int newNode(Game& g, const Pos& p, int parent, int pedge) {
        MNode n;
        n.p = p;
        n.parent = parent;
        n.pedge = pedge;
        Gen gen;
        genSafe(p, gen);
        if (gen.win) n.status = 1;
        else if (gen.n == 0) n.status = -1;
        else {
            n.first = (int)g.edges.size();
            n.n = gen.n;
            for (int k = 0; k < gen.n; k++) {
                Edge e;
                e.m = gen.m[k];
                e.a = (int16_t)encodeMove(p, gen.m[k]);
                g.edges.push_back(e);
            }
        }
        g.nodes.push_back(n);
        return (int)g.nodes.size() - 1;
    }

    // Returns NEED_EVAL (g.leaf set), NEED_LABEL or IDLE.
    int advance(int i) {
        for (;;) {
            Game& g = games[i];
            if (!g.active) return IDLE;
            if (g.leaf >= 0) return NEED_EVAL;
            if (g.nodes.empty()) {
                // New move. Exact root solve late in the game.
                if (g.pos.blocks() >= cfg.labelMinBlocks && !g.labelled) return NEED_LABEL;
                int root = newNode(g, g.pos, -1, -1);
                MNode& r = g.nodes[root];
                if (r.status == -1) { finishGame(i, 1 - (g.ply & 1)); continue; }  // no safe move
                bool labelWins = false;  // root proven won AND at least one winning move known
                if (g.labelled && g.label.value == 1)
                    for (int a = 0; a < NMOVE && !labelWins; a++) labelWins = g.label.win[a];
                if (r.status == 1 || labelWins) {
                    playProvenWin(i);
                    continue;
                }
                g.leaf = root;  // root itself needs a network evaluation first
                return NEED_EVAL;
            }
            MNode& root = g.nodes[0];
            if (root.status != 0 || root.N >= cfg.sims) {
                if (root.status == 1) { playProvenWin(i); continue; }
                chooseAndPlay(i);
                continue;
            }
            // One simulation: select down to an unexpanded node.
            int x = 0;
            for (;;) {
                MNode& n = g.nodes[x];
                if (n.status != 0) {  // proven / terminal: exact backup
                    backup(g, x, (float)n.status);
                    break;
                }
                if (!n.expanded) {  // freshly created child: needs the network
                    g.leaf = x;
                    return NEED_EVAL;
                }
                int e = select(g, x);
                if (e < 0) {  // every move loses: node is lost
                    g.nodes[x].status = -1;
                    propagate(g, g.nodes[x].parent);
                    continue;
                }
                Edge& ed = g.edges[e];
                if (ed.child < 0) {
                    Pos cp = makeChild(g.nodes[x].p, ed.m);
                    int c = newNode(g, cp, x, e);
                    g.edges[e].child = c;
                    if (g.nodes[c].status != 0) propagate(g, x);
                }
                x = g.edges[e].child;
            }
        }
    }

    // PUCT over edges, skipping moves into positions proven won for the opponent;
    // a move into a proven-lost position is taken immediately.
    int select(Game& g, int x) {
        MNode& n = g.nodes[x];
        float sq = std::sqrt((float)std::max(1, n.N));
        int best = -1;
        float bestV = -1e30f;
        for (int k = n.first; k < n.first + n.n; k++) {
            Edge& e = g.edges[k];
            if (e.child >= 0) {
                int st = g.nodes[e.child].status;
                if (st == -1) return k;
                if (st == 1) continue;
            }
            float q = e.N ? e.W / e.N : 0.0f;
            float u = cfg.cpuct * e.P * sq / (1 + e.N);
            if (q + u > bestV) { bestV = q + u; best = k; }
        }
        return best;
    }

    // MCTS-Solver: a node is won if some child is lost for its mover, and lost if
    // every child is won for its mover.
    void propagate(Game& g, int x) {
        while (x >= 0) {
            MNode& n = g.nodes[x];
            if (n.status != 0) return;
            bool anyLoss = false, allWin = true;
            for (int k = n.first; k < n.first + n.n; k++) {
                int c = g.edges[k].child;
                int st = c >= 0 ? g.nodes[c].status : 0;
                if (st == -1) anyLoss = true;
                if (st != 1) allWin = false;
            }
            if (anyLoss) n.status = 1;
            else if (allWin) n.status = -1;
            else return;
            x = n.parent;
        }
    }

    void expand(Game& g, int x, const float* logits) {
        MNode& n = g.nodes[x];
        n.expanded = true;
        float mx = -1e30f;
        for (int k = n.first; k < n.first + n.n; k++) mx = std::max(mx, logits[g.edges[k].a]);
        float sum = 0;
        for (int k = n.first; k < n.first + n.n; k++) {
            float p = std::exp(logits[g.edges[k].a] - mx);
            g.edges[k].P = p;
            sum += p;
        }
        for (int k = n.first; k < n.first + n.n; k++) g.edges[k].P /= sum;
        if (x == 0 && cfg.noise && n.n > 0) {
            std::gamma_distribution<float> gam(cfg.dirAlpha, 1.0f);
            std::vector<float> d(n.n);
            float ds = 0;
            for (float& v : d) { v = gam(rng); ds += v; }
            for (int k = 0; k < n.n; k++) {
                Edge& e = g.edges[n.first + k];
                e.P = (1 - cfg.dirFrac) * e.P + cfg.dirFrac * d[k] / std::max(ds, 1e-12f);
            }
        }
    }

    // v is from the mover's perspective at node x.
    void backup(Game& g, int x, float v) {
        for (;;) {
            MNode& n = g.nodes[x];
            n.N++;
            if (n.parent < 0) break;
            Edge& e = g.edges[n.pedge];
            e.N++;
            e.W += -v;
            v = -v;
            x = n.parent;
        }
    }

    void playProvenWin(int i) {
        Game& g = games[i];
        float pi[NMOVE] = {0};
        std::vector<int> wins;
        uint8_t mask[NMOVE];
        if (g.labelled && g.label.value == 1)
            for (int a = 0; a < NMOVE; a++)
                if (g.label.win[a]) wins.push_back(a);
        if (!wins.empty()) {
        } else if (legalMask(g.pos, mask) && canWinNow(g.pos, g.pos.me)) {
            for (int a = 0; a < NMOVE; a++)
                if (mask[a]) wins.push_back(a);
        } else if (!g.nodes.empty()) {  // proven by the tree (MCTS-Solver)
            MNode& r = g.nodes[0];
            for (int k = r.first; k < r.first + r.n; k++) {
                int c = g.edges[k].child;
                if (c >= 0 && g.nodes[c].status == -1) wins.push_back(g.edges[k].a);
            }
        }
        if (wins.empty()) {  // should not happen; fall back to the search
            chooseAndPlay(i);
            return;
        }
        solverMoves++;
        for (int a : wins) pi[a] = 1.0f / wins.size();
        record(g, pi, 1.0f, 1, 1);
        int a = wins[rng() % wins.size()];
        play(i, decode(g.pos, a));
    }

    void chooseAndPlay(int i) {
        Game& g = games[i];
        MNode& r = g.nodes[0];
        float pi[NMOVE] = {0};
        int total = 0;
        float wsum = 0;
        for (int k = r.first; k < r.first + r.n; k++) {
            total += g.edges[k].N;
            wsum += g.edges[k].W;
        }
        // Candidate moves: exclude moves proven to lose unless all of them do.
        std::vector<int> cand;
        for (int k = r.first; k < r.first + r.n; k++) {
            int c = g.edges[k].child;
            if (!(c >= 0 && g.nodes[c].status == 1)) cand.push_back(k);
        }
        if (cand.empty())
            for (int k = r.first; k < r.first + r.n; k++) cand.push_back(k);
        for (int k : cand) pi[g.edges[k].a] = (float)g.edges[k].N;
        float s = 0;
        for (float v : pi) s += v;
        if (s <= 0) {
            for (int k : cand) pi[g.edges[k].a] = 1;
            s = (float)cand.size();
        }
        for (float& v : pi) v /= s;
        float q = total ? wsum / total : 0.0f;
        bool exact = (g.labelled && g.label.value != 0) || r.status != 0;
        int v = r.status != 0 ? r.status : g.label.value;
        record(g, pi, q, exact, exact ? v : 0);

        int pick;
        if (g.ply < cfg.tempPlies) {
            std::uniform_real_distribution<float> U(0, 1);
            float x = U(rng), acc = 0;
            pick = cand.back();
            for (int k : cand) {
                acc += pi[g.edges[k].a];
                if (x <= acc) { pick = k; break; }
            }
        } else {
            pick = cand[0];
            for (int k : cand)
                if (g.edges[k].N > g.edges[pick].N) pick = k;
        }
        play(i, g.edges[pick].m);
    }
};

}  // namespace az