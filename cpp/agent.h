// Single-position search for playing and analysis.
//
// Same MCTS as self-play (PUCT, forced-move pruning, MCTS-Solver proofs, network
// priors/values) but for one position at a time, with:
//   * batched leaf collection using virtual loss, so one network call evaluates
//     many leaves;
//   * an optional exact df-pn solve of the root (value + every proven winning move);
//   * full introspection: per-move visits, value, prior, proof status and the
//     principal variation, for visualising what the agent is "thinking".
#pragma once
#include "az.h"
#include <climits>
#include <cmath>
#include <set>

namespace az {

struct SearchParams {
    float cpuct = 1.5f;
    uint64_t rootBudget = 200000;   // df-pn expansions for the root solve (0 = off)
    uint64_t childBudget = 2000;    // per move, to list the winning moves of a won root
    double ttGb = 0.25;
};

struct MoveStat {
    Move m;
    int a;              // action index (network policy index)
    bool stepUp;        // immediate win by stepping up to level 3
    int N;
    float Q;            // mean value for the root mover (-1..1), 0 if unvisited
    float P;            // network prior
    int proof;          // +1 proven win for the root mover, -1 proven loss, 0 unknown
    std::vector<Move> pv;  // continuation after this move (alternating sides)
    bool pvEndsInWin;   // last pv entry is a step-up win
};

class Search {
   public:
    Search(const Pos& root, const SearchParams& sp) : prm(sp) {
        rootPos = root;
        // Immediate step-up wins and positions with no safe move are decided at once.
        Gen g;
        genSafe(root, g);
        if (g.win) {
            rootStatus = 1;
            BB occ = root.me | root.op, L3 = root.lvl(3);
            for (BB x = root.me & root.lvl(2); x; x &= x - 1)
                for (BB t = T.adj[lsb(x)] & L3 & ~occ; t; t &= t - 1)
                    winNow.push_back(Move{(uint8_t)lsb(x), (uint8_t)lsb(t), (uint8_t)lsb(x)});
            return;
        }
        newNode(root, -1, -1);
        if (g.n == 0) { rootStatus = -1; nodes[0].status = -1; return; }
        if (prm.rootBudget) {
            solver = makeSolver(prm.ttGb);
            LabelOut lo;
            labelOne(*solver, root, prm.rootBudget, prm.childBudget, lo);
            solverValue = lo.value;
            if (lo.value == 1)
                for (int a = 0; a < NMOVE; a++)
                    if (lo.win[a]) solverWins.insert(a);
            if (lo.value != 0) rootStatus = lo.value;
            solver.reset();  // the transposition table is only needed for the root solve
        }
    }

    // Collect up to maxLeaves leaves that need the network; writes their planes.
    // Simulations ending in proven/terminal nodes are backed up immediately.
    int prepare(float* planes, int maxLeaves) {
        pending.clear();
        if (done()) return 0;
        int guard = 0;
        while ((int)pending.size() < maxLeaves && guard++ < maxLeaves * 4) {
            if (done()) break;  // proven by the tree
            std::vector<int> path;  // edge indices from the root
            int x = 0;
            bool collided = false;
            for (;;) {
                MNode& n = nodes[x];
                if (n.status != 0) {  // terminal / proven: exact value
                    addVirtual(path);
                    backup(path, (float)n.status);
                    sims++;
                    break;
                }
                if (!n.expanded) {
                    if (pendingNode.count(x)) { collided = true; break; }
                    addVirtual(path);
                    pendingNode.insert(x);
                    pending.push_back({x, path});
                    encodePlanes(n.p, planes + (size_t)(pending.size() - 1) * 25 * NPLANES);
                    break;
                }
                int e = select(x);
                if (e < 0) {  // every move loses: node is lost
                    nodes[x].status = -1;
                    propagate(nodes[x].parent);
                    continue;  // re-handle as a proven node
                }
                path.push_back(e);
                if (edges[e].child < 0) {
                    int c = newNode(makeChild(nodes[x].p, edges[e].m), x, e);
                    edges[e].child = c;
                    if (nodes[c].status != 0) propagate(x);
                }
                x = edges[e].child;
            }
            if (collided) break;
        }
        return (int)pending.size();
    }

    // logits[count][NACT], values[count] for the leaves of the last prepare().
    void apply(const float* logits, const float* values) {
        for (size_t j = 0; j < pending.size(); j++) {
            int x = pending[j].node;
            expand(x, logits + j * NACT);
            netV[x] = values[j];
            if (x == 0) rootNetV = values[j];
            backup(pending[j].path, values[j]);
            pendingNode.erase(x);
            sims++;
        }
        pending.clear();
    }

    // Nothing left to search: immediate win, no safe move, or the tree proved the root.
    bool done() const {
        if (winNow.size() || nodes.empty()) return true;
        const MNode& r = nodes[0];
        return r.status != 0 && (r.expanded || r.n == 0);
    }
    int simulations() const { return sims; }
    int rootN() const { return nodes.empty() ? 0 : nodes[0].N; }
    int status() const {  // +1 / -1 proven for the root mover (solver or tree), 0 unknown
        if (rootStatus) return rootStatus;
        return nodes.empty() ? 0 : nodes[0].status;
    }
    int solverResult() const { return solverValue; }
    float netValue() const { return rootNetV; }
    float searchValue() const {
        if (nodes.empty()) return winNow.size() ? 1.0f : 0.0f;
        const MNode& r = nodes[0];
        int n = 0;
        float w = 0;
        for (int k = r.first; k < r.first + r.n; k++) { n += edges[k].N; w += edges[k].W; }
        return n ? w / n : rootNetV;
    }
    size_t treeSize() const { return nodes.size(); }

    std::vector<MoveStat> stats(int pvDepth) const {
        std::vector<MoveStat> out;
        for (Move m : winNow) out.push_back(MoveStat{m, encodeMove(rootPos, m), true, 0, 1.0f, 1.0f, 1, {}, false});
        if (nodes.empty()) return out;
        const MNode& r = nodes[0];
        for (int k = r.first; k < r.first + r.n; k++) {
            const Edge& e = edges[k];
            MoveStat s{e.m, e.a, false, e.N, e.N ? e.W / e.N : 0.0f, e.P, 0, {}, false};
            if (e.child >= 0) s.proof = -nodes[e.child].status;
            if (solverValue == 1 && solverWins.count(e.a)) s.proof = 1;
            if (solverValue == -1) s.proof = -1;
            if (e.child >= 0) pv(e.child, pvDepth, s.pv, s.pvEndsInWin);
            out.push_back(std::move(s));
        }
        std::sort(out.begin(), out.end(), [](const MoveStat& a, const MoveStat& b) {
            if (a.stepUp != b.stepUp) return a.stepUp;
            if (a.N != b.N) return a.N > b.N;
            return a.P > b.P;
        });
        return out;
    }

    // Move to play: an immediate win; else a proven win (most visited); else the most
    // visited move that is not proven lost; ties broken by the prior.
    MoveStat best() const {
        auto st = stats(0);
        if (st.empty()) return MoveStat{Move{0, 0, 0}, -1, false, 0, 0, 0, 0, {}, false};
        for (auto& s : st) if (s.stepUp) return s;
        for (auto& s : st) if (s.proof == 1) return s;
        for (auto& s : st) if (s.proof != -1) return s;
        return st[0];
    }


    // ---- Step-by-step search (for teaching; no virtual loss, one simulation) -------
    // stepSelect: walk down by PUCT to a leaf. Returns 1 if the leaf needs the network
    // (its planes are written to `planes`), 2 if its value is already known (terminal or
    // proven), 0 if the search is finished. stepExpand / stepExpandKnown evaluate it,
    // stepBackup adds the value along the path.
    int stepSelect(float* planes) {
        stepPath.clear();
        stepLeaf = -1;
        stepState = 0;
        if (done() || !pending.empty()) return 0;
        int x = 0;
        for (;;) {
            MNode& n = nodes[x];
            if (n.status != 0) { stepLeaf = x; stepState = 2; stepValue = (float)n.status; return 2; }
            if (!n.expanded) {
                stepLeaf = x;
                stepState = 1;
                encodePlanes(n.p, planes);
                return 1;
            }
            int e = select(x);
            if (e < 0) { nodes[x].status = -1; propagate(nodes[x].parent); continue; }
            stepPath.push_back(e);
            if (edges[e].child < 0) {
                int c = newNode(makeChild(nodes[x].p, edges[e].m), x, e);
                edges[e].child = c;
                if (nodes[c].status != 0) propagate(x);
            }
            x = edges[e].child;
        }
    }
    void stepExpand(const float* logits, float value) {
        if (stepState != 1) return;
        expand(stepLeaf, logits);
        netV[stepLeaf] = value;
        if (stepLeaf == 0) rootNetV = value;
        stepValue = value;
        stepState = 3;
    }
    void stepExpandKnown() { if (stepState == 2) stepState = 3; }
    void stepBackup() {
        if (stepState != 3) return;
        nodes[0].N++;
        float v = stepValue;
        for (int i = (int)stepPath.size() - 1; i >= 0; i--) {
            Edge& e = edges[stepPath[i]];
            e.N++;
            e.W += -v;
            nodes[e.child].N++;
            v = -v;
        }
        sims++;
        stepState = 4;  // backed up; the path stays highlighted until the next select
    }
    int stepPhase() const { return stepState; }
    int stepLeafNode() const { return stepLeaf; }
    float stepLeafValue() const { return stepValue; }
    std::vector<int> stepNodes() const {  // node ids on the current step path
        std::vector<int> out;
        if (stepState == 0 || nodes.empty()) return out;
        out.push_back(0);
        for (int e : stepPath) out.push_back(edges[e].child);
        return out;
    }

    // ---- Tree export for the visualiser ----------------------------------------------
    struct TreeItem {
        int id;            // node index, or -(edge + 1) for a move that has no node yet
        int parent;        // id of the parent item, INT_MIN for the displayed root
        int depth;         // plies from the search root
        int N;             // visits (virtual loss removed)
        float Q;           // mean value from the parent mover's view (edge), or the root mover's view at the search root
        float P;           // network prior of the incoming move (1 for the root)
        float U;           // PUCT exploration bonus of the incoming move
        int status;        // +1 / -1 proven for the side to move at this node, 0 unknown
        bool expanded;     // evaluated by the network
        float netV;        // network value for the side to move at this node (NaN if not evaluated)
        bool hasMove;
        Move m;            // incoming move
        Pos p;             // position (side to move's perspective)
        bool onPV, onStep;
        int childCount;    // safe moves available at this node
    };

    std::vector<TreeItem> exportTree(int root, int maxNodes, float prunePct, int maxDepth, bool unvisited) const {
        std::vector<TreeItem> out;
        if (nodes.empty()) return out;
        if (root < 0 || root >= (int)nodes.size()) root = 0;
        std::set<int> pvSet, stepSet;
        for (int x = 0, guard = 0; x >= 0 && guard < 64; guard++) {
            pvSet.insert(x);
            const MNode& n = nodes[x];
            int best = -1;
            for (int k = n.first; k < n.first + n.n; k++)
                if (edges[k].child >= 0 && dispN(k) > 0 && (best < 0 || dispN(k) > dispN(best))) best = k;
            x = best < 0 ? -1 : edges[best].child;
        }
        for (int x : stepNodes()) stepSet.insert(x);
        struct Cand { int node, edge, parent, depth; float prio; };
        std::vector<Cand> cands;
        int rootDepth = depthOf[root];
        cands.push_back({root, nodes[root].pedge, INT_MIN, rootDepth, 1e30f});
        std::vector<int> stack{0};
        while (!stack.empty()) {
            Cand c = cands[stack.back()];
            stack.pop_back();
            if (c.node < 0 || c.depth - rootDepth >= maxDepth) continue;
            const MNode& n = nodes[c.node];
            if (!n.expanded) continue;
            std::vector<int> ks;
            int total = 0;
            for (int k = n.first; k < n.first + n.n; k++) { ks.push_back(k); total += dispN(k); }
            std::sort(ks.begin(), ks.end(), [&](int a, int b) {
                if (dispN(a) != dispN(b)) return dispN(a) > dispN(b);
                return edges[a].P > edges[b].P;
            });
            int acc = 0;
            for (int k : ks) {
                int ch = edges[k].child;
                bool forced = ch >= 0 && stepSet.count(ch);
                if (!forced && prunePct < 100 && total > 0 && acc >= prunePct / 100.0f * total) break;
                if (!forced && dispN(k) == 0 && !unvisited) continue;
                acc += dispN(k);
                float prio = forced ? 1e29f : dispN(k) > 0 ? (float)dispN(k) : edges[k].P * 0.999f;
                cands.push_back({ch, k, c.node, c.depth + 1, prio});
                if (ch >= 0) stack.push_back((int)cands.size() - 1);
            }
        }
        // global budget: most visited first, parents before children
        std::vector<int> order(cands.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            if (cands[a].prio != cands[b].prio) return cands[a].prio > cands[b].prio;
            return cands[a].depth < cands[b].depth;
        });
        std::set<int> keptNodes;
        for (int i : order) {
            const Cand& c = cands[i];
            if ((int)out.size() >= std::max(1, maxNodes) && c.prio < 1e29f) break;
            if (c.parent != INT_MIN && !keptNodes.count(c.parent)) continue;
            TreeItem t{};
            t.depth = c.depth;
            t.parent = c.parent;
            t.hasMove = c.edge >= 0;
            if (c.edge >= 0) {
                const Edge& e = edges[c.edge];
                int pn = nodes[c.parent == INT_MIN ? nodes[c.node].parent : c.parent].N;
                t.m = e.m;
                t.N = dispN(c.edge);
                t.Q = t.N ? (e.W + vl[c.edge]) / t.N : 0.0f;
                t.P = e.P;
                t.U = prm.cpuct * e.P * std::sqrt((float)std::max(1, pn)) / (1 + t.N);
            } else {
                t.N = nodes[0].N - rootVl;
                t.Q = searchValue();
                t.P = 1;
                t.U = 0;
            }
            if (c.node >= 0) {
                const MNode& n = nodes[c.node];
                t.id = c.node;
                t.status = n.status;
                t.expanded = n.expanded;
                t.netV = netV[c.node];
                t.p = n.p;
                t.childCount = n.n;
                t.onPV = pvSet.count(c.node) > 0;
                t.onStep = stepSet.count(c.node) > 0;
                keptNodes.insert(c.node);
            } else {
                t.id = -(c.edge + 1);
                t.status = 0;
                t.expanded = false;
                t.netV = NAN;
                t.p = makeChild(nodes[c.parent].p, edges[c.edge].m);
                t.childCount = -1;
                t.onPV = t.onStep = false;
            }
            out.push_back(t);
        }
        return out;
    }
    size_t nodeCount() const { return nodes.size(); }

   private:
    struct Pending {
        int node;
        std::vector<int> path;
    };
    SearchParams prm;
    Pos rootPos;
    std::vector<MNode> nodes;
    std::vector<Edge> edges;
    std::vector<Pending> pending;
    std::set<int> pendingNode;
    std::vector<Move> winNow;
    std::unique_ptr<Solver> solver;
    std::set<int> solverWins;
    int solverValue = 0, rootStatus = 0, sims = 0;
    float rootNetV = 0;
    std::vector<uint16_t> vl;      // virtual losses currently on each edge
    int rootVl = 0;
    std::vector<float> netV;       // network value per node (side to move), NaN until evaluated
    std::vector<int> depthOf;      // plies from the root per node
    std::vector<int> stepPath;     // edges of the step-by-step simulation
    int stepLeaf = -1, stepState = 0;
    float stepValue = 0;
    int dispN(int e) const { return edges[e].N - vl[e]; }

    int newNode(const Pos& p, int parent, int pedge) {
        MNode n;
        n.p = p;
        n.parent = parent;
        n.pedge = pedge;
        Gen gen;
        genSafe(p, gen);
        if (gen.win) n.status = 1;
        else if (gen.n == 0) n.status = -1;
        else {
            n.first = (int)edges.size();
            n.n = gen.n;
            for (int k = 0; k < gen.n; k++) {
                Edge e;
                e.m = gen.m[k];
                e.a = (int16_t)encodeMove(p, gen.m[k]);
                edges.push_back(e);
            }
        }
        nodes.push_back(n);
        vl.resize(edges.size(), 0);
        netV.push_back(NAN);
        depthOf.push_back(parent < 0 ? 0 : depthOf[parent] + 1);
        return (int)nodes.size() - 1;
    }

    int select(int x) const {
        const MNode& n = nodes[x];
        float sq = std::sqrt((float)std::max(1, n.N));
        int best = -1;
        float bestV = -1e30f;
        for (int k = n.first; k < n.first + n.n; k++) {
            const Edge& e = edges[k];
            if (e.child >= 0) {
                int st = nodes[e.child].status;
                if (st == -1) return k;
                if (st == 1) continue;
            }
            float q = e.N ? e.W / e.N : 0.0f;
            float u = prm.cpuct * e.P * sq / (1 + e.N);
            if (q + u > bestV) { bestV = q + u; best = k; }
        }
        return best;
    }

    void propagate(int x) {
        while (x >= 0) {
            MNode& n = nodes[x];
            if (n.status != 0) return;
            bool anyLoss = false, allWin = true;
            for (int k = n.first; k < n.first + n.n; k++) {
                int c = edges[k].child;
                int st = c >= 0 ? nodes[c].status : 0;
                if (st == -1) anyLoss = true;
                if (st != 1) allWin = false;
            }
            if (anyLoss) n.status = 1;
            else if (allWin) n.status = -1;
            else return;
            x = n.parent;
        }
    }

    void expand(int x, const float* logits) {
        MNode& n = nodes[x];
        n.expanded = true;
        float mx = -1e30f;
        for (int k = n.first; k < n.first + n.n; k++) mx = std::max(mx, logits[edges[k].a]);
        float sum = 0;
        for (int k = n.first; k < n.first + n.n; k++) {
            float p = std::exp(logits[edges[k].a] - mx);
            edges[k].P = p;
            sum += p;
        }
        for (int k = n.first; k < n.first + n.n; k++) edges[k].P /= std::max(sum, 1e-20f);
    }

    // Virtual loss: count the visit now and score it as a loss for the edge's mover,
    // so the next selections in the same batch spread to other branches.
    void addVirtual(const std::vector<int>& path) {
        nodes[0].N++;
        rootVl++;
        for (int e : path) {
            edges[e].N++;
            edges[e].W -= 1.0f;
            vl[e]++;
            nodes[edges[e].child].N++;
        }
    }
    // Replace the virtual loss by the real value v (mover's perspective at the leaf).
    void backup(const std::vector<int>& path, float v) {
        for (int i = (int)path.size() - 1; i >= 0; i--) {
            edges[path[i]].W += 1.0f - v;  // edge value is from the parent's mover: -v
            vl[path[i]]--;
            v = -v;
        }
        rootVl--;
    }

    void pv(int x, int depth, std::vector<Move>& out, bool& endsInWin) const {
        endsInWin = false;
        for (int d = 0; d < depth && x >= 0; d++) {
            const MNode& n = nodes[x];
            if (n.status == 1 && n.n == 0) {  // mover steps up and wins
                BB occ = n.p.me | n.p.op, L3 = n.p.lvl(3);
                for (BB w = n.p.me & n.p.lvl(2); w; w &= w - 1) {
                    BB t = T.adj[lsb(w)] & L3 & ~occ;
                    if (t) {
                        out.push_back(Move{(uint8_t)lsb(w), (uint8_t)lsb(t), (uint8_t)lsb(w)});
                        endsInWin = true;
                        return;
                    }
                }
                return;
            }
            if (!n.expanded || n.n == 0) return;
            int best = -1;
            for (int k = n.first; k < n.first + n.n; k++) {
                if (edges[k].child >= 0 && nodes[edges[k].child].status == -1) { best = k; break; }
                if (edges[k].N > 0 && (best < 0 || edges[k].N > edges[best].N)) best = k;
            }
            if (best < 0) return;
            out.push_back(edges[best].m);
            x = edges[best].child;
        }
    }
};

// Every legal move+build for the side to move (no pruning). Step-up wins have b = 255.
inline std::vector<Move> allMoves(const Pos& p) {
    std::vector<Move> out;
    BB occ = p.me | p.op, bmAllowed = p.buildable();
    for (BB ws = p.me; ws; ws &= ws - 1) {
        int w = lsb(ws);
        for (BB d = T.adj[w] & ~occ & climbMask(p, p.height(w)); d; d &= d - 1) {
            int t = lsb(d);
            if (p.height(w) == 2 && p.height(t) == 3) {
                out.push_back(Move{(uint8_t)w, (uint8_t)t, 255});
                continue;
            }
            for (BB b = T.adj[t] & ~(occ ^ bit(w) ^ bit(t)) & bmAllowed; b; b &= b - 1)
                out.push_back(Move{(uint8_t)w, (uint8_t)t, (uint8_t)lsb(b)});
        }
    }
    return out;
}

}  // namespace az