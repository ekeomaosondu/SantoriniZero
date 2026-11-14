// Python bindings: python3 -m pybind11 --includes; built as module `santorini`.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "agent.h"

namespace py = pybind11;
using namespace az;
template <class T>
using arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

static std::vector<Pos> toPositions(const arr<int8_t>& states) {
    if (states.ndim() != 2 || states.shape(1) != STATE_LEN)
        throw std::runtime_error("states must have shape [N, 29]");
    std::vector<Pos> ps(states.shape(0));
    const int8_t* s = states.data();
    for (size_t i = 0; i < ps.size(); i++) ps[i] = fromState(s + i * STATE_LEN);
    return ps;
}
static arr<int8_t> fromPositions(const std::vector<Pos>& ps) {
    arr<int8_t> out({(py::ssize_t)ps.size(), (py::ssize_t)STATE_LEN});
    for (size_t i = 0; i < ps.size(); i++) toState(ps[i], out.mutable_data() + i * STATE_LEN);
    return out;
}

PYBIND11_MODULE(santorini, m) {
    az::init();
    m.attr("NPLANES") = NPLANES;
    m.attr("NMOVE") = NMOVE;
    m.attr("NACT") = NACT;
    m.attr("STATE_LEN") = STATE_LEN;

    m.def("legal_mask", [](arr<int8_t> states) {
        auto ps = toPositions(states);
        arr<uint8_t> out({(py::ssize_t)ps.size(), (py::ssize_t)NMOVE});
        for (size_t i = 0; i < ps.size(); i++) legalMask(ps[i], out.mutable_data() + i * NMOVE);
        return out;
    }, "Safe (non-losing-in-one) actions per state, [N,128] uint8.");

    m.def("encode", [](arr<int8_t> states, arr<int32_t> syms) {
        auto ps = toPositions(states);
        py::ssize_t n = ps.size();
        if (syms.size() != n) throw std::runtime_error("syms must have length N");
        arr<float> planes({n, (py::ssize_t)5, (py::ssize_t)5, (py::ssize_t)NPLANES});
        arr<uint8_t> legal({n, (py::ssize_t)NMOVE});
        arr<int32_t> perm({n, (py::ssize_t)NMOVE});
        float* P = planes.mutable_data();
        uint8_t* L = legal.mutable_data();
        int32_t* Q = perm.mutable_data();
        const int32_t* S = syms.data();
        {
            py::gil_scoped_release nogil;
            for (py::ssize_t i = 0; i < n; i++) {
                Pos t = transformPos(ps[i], S[i]);
                encodePlanes(t, P + i * 25 * NPLANES);
                legalMask(t, L + i * NMOVE);
                actionPerm(ps[i], S[i], Q + i * NMOVE);
            }
        }
        return py::make_tuple(planes, legal, perm);
    }, "Planes [N,5,5,15], legal mask [N,128] and action permutation [N,128] after symmetry syms[i].\n"
       "A policy target pi over the original state maps to pi_t[perm[a]] = pi[a].");

    m.def("transform", [](arr<int8_t> states, arr<int32_t> syms) {
        auto ps = toPositions(states);
        for (size_t i = 0; i < ps.size(); i++) ps[i] = transformPos(ps[i], syms.data()[i]);
        return fromPositions(ps);
    }, "States after D4 symmetry syms[i] (0-3 rotations, 4-7 rotation+mirror).");

    m.def("apply_action", [](arr<int8_t> state, int a) {
        auto ps = toPositions(state.reshape({1, STATE_LEN}));
        Pos p = ps[0];
        Move mv = decode(p, a);
        bool win = p.height(mv.f) == 2 && p.height(mv.t) == 3;
        Pos c = win ? p : makeChild(p, mv);
        return py::make_tuple(fromPositions({c}), win);
    }, "Child state (opponent to move) after action a, and whether a is a winning step-up.");

    m.def("label", [](arr<int8_t> states, uint64_t rootBudget, uint64_t childBudget, int threads,
                      double ttGb) {
        auto ps = toPositions(states);
        std::vector<LabelOut> out;
        {
            py::gil_scoped_release nogil;
            labelBatch(ps, rootBudget, childBudget, threads, ttGb, out);
        }
        py::ssize_t n = ps.size();
        arr<int8_t> v(n);
        arr<uint8_t> w({n, (py::ssize_t)NMOVE});
        for (py::ssize_t i = 0; i < n; i++) {
            v.mutable_data()[i] = out[i].value;
            memcpy(w.mutable_data() + i * NMOVE, out[i].win, NMOVE);
        }
        return py::make_tuple(v, w);
    }, py::arg("states"), py::arg("root_budget") = 200000, py::arg("child_budget") = 20000,
       py::arg("threads") = 4, py::arg("tt_gb") = 0.25,
       "Exact values (+1/-1/0 unresolved) and proven winning actions.");

    m.def("generate", [](int n, int minB, int maxB, uint64_t rootBudget, uint64_t childBudget,
                         int threads, double ttGb, uint64_t seed) {
        std::vector<Pos> ps;
        std::vector<LabelOut> out;
        {
            py::gil_scoped_release nogil;
            generateLabeled(n, minB, maxB, rootBudget, childBudget, threads, ttGb, seed, ps, out);
        }
        arr<int8_t> v(n);
        arr<uint8_t> w({(py::ssize_t)n, (py::ssize_t)NMOVE});
        for (int i = 0; i < n; i++) {
            v.mutable_data()[i] = out[i].value;
            memcpy(w.mutable_data() + (size_t)i * NMOVE, out[i].win, NMOVE);
        }
        return py::make_tuple(fromPositions(ps), v, w);
    }, py::arg("n"), py::arg("min_blocks"), py::arg("max_blocks"), py::arg("root_budget") = 100000,
       py::arg("child_budget") = 5000, py::arg("threads") = 4, py::arg("tt_gb") = 0.25,
       py::arg("seed") = 1,
       "n solver-labelled positions from random safe play, blocks uniform in [min, max].");

    m.def("pretty", [](arr<int8_t> state) {
        Pos p = toPositions(state.reshape({1, STATE_LEN}))[0];
        std::string s = "    A  B  C  D  E\n";
        for (int r = 0; r < 5; r++) {
            s += " " + std::to_string(r + 1) + " ";
            for (int c = 0; c < 5; c++) {
                int i = r * 5 + c, h = p.height(i);
                char w = (p.me >> i & 1) ? 'X' : (p.op >> i & 1) ? 'O' : ' ';
                s += h == 4 ? " []" : " " + std::to_string(h) + w;
            }
            s += "\n";
        }
        return s + "(X = to move, O = opponent, [] = dome)";
    });

    m.def("all_moves", [](arr<int8_t> state) {
        Pos p = toPositions(state.reshape({1, STATE_LEN}))[0];
        auto ms = allMoves(p);
        arr<int32_t> out({(py::ssize_t)ms.size(), (py::ssize_t)3});
        for (size_t i = 0; i < ms.size(); i++) {
            out.mutable_data()[i * 3] = ms[i].f;
            out.mutable_data()[i * 3 + 1] = ms[i].t;
            out.mutable_data()[i * 3 + 2] = ms[i].b == 255 ? -1 : ms[i].b;
        }
        return out;
    }, "Every legal (from, to, build) for the side to move, unpruned; build = -1 for a step-up win.");

    m.def("action_index", [](arr<int8_t> state, int f, int t, int b) {
        Pos p = toPositions(state.reshape({1, STATE_LEN}))[0];
        return encodeMove(p, Move{(uint8_t)f, (uint8_t)t, (uint8_t)(b < 0 ? f : b)});
    }, "Policy index of (from, to, build); build = -1 for a step-up win.");

    py::class_<SearchParams>(m, "SearchParams")
        .def(py::init<>())
        .def_readwrite("cpuct", &SearchParams::cpuct)
        .def_readwrite("root_budget", &SearchParams::rootBudget)
        .def_readwrite("child_budget", &SearchParams::childBudget)
        .def_readwrite("tt_gb", &SearchParams::ttGb);

    auto moveDict = [](const MoveStat& s) {
        py::dict d;
        d["from"] = (int)s.m.f; d["to"] = (int)s.m.t; d["build"] = s.stepUp ? -1 : (int)s.m.b;
        d["action"] = s.a; d["step_up"] = s.stepUp; d["visits"] = s.N; d["q"] = s.Q;
        d["prior"] = s.P; d["proof"] = s.proof;
        py::list pv;
        for (size_t i = 0; i < s.pv.size(); i++) {
            bool win = s.pvEndsInWin && i + 1 == s.pv.size();
            pv.append(py::make_tuple((int)s.pv[i].f, (int)s.pv[i].t, win ? -1 : (int)s.pv[i].b));
        }
        d["pv"] = pv;
        d["pv_ends_in_win"] = s.pvEndsInWin;
        return d;
    };

    py::class_<Search>(m, "Search")
        .def(py::init([](arr<int8_t> state, const SearchParams& sp) {
            Pos p = toPositions(state.reshape({1, STATE_LEN}))[0];
            py::gil_scoped_release nogil;
            return new Search(p, sp);
        }), "Search from a state (side to move's perspective). Runs the root solve if enabled.")
        .def("prepare", [](Search& s, arr<float> planes, int maxLeaves) {
            if (planes.ndim() != 4 || planes.shape(0) < maxLeaves)
                throw std::runtime_error("planes buffer must be [>=max_leaves, 5, 5, 15]");
            float* P = planes.mutable_data();
            py::gil_scoped_release nogil;
            return s.prepare(P, maxLeaves);
        }, "Collect up to max_leaves leaves for the network; returns how many.")
        .def("apply", [](Search& s, arr<float> logits, arr<float> values) {
            s.apply(logits.data(), values.data());
        })
        .def_property_readonly("done", &Search::done)
        .def_property_readonly("simulations", &Search::simulations)
        .def_property_readonly("status", &Search::status)
        .def_property_readonly("solver_result", &Search::solverResult)
        .def_property_readonly("net_value", &Search::netValue)
        .def_property_readonly("search_value", &Search::searchValue)
        .def_property_readonly("tree_size", &Search::treeSize)
        .def("moves", [moveDict](Search& s, int pvDepth) {
            py::list out;
            for (auto& st : s.stats(pvDepth)) out.append(moveDict(st));
            return out;
        }, py::arg("pv_depth") = 8, "Per-move statistics, most visited first.")
        .def("best", [moveDict](Search& s) { return moveDict(s.best()); })
        .def("step_select", [](Search& s, arr<float> planes) {
            if (planes.ndim() != 4 || planes.shape(0) < 1) throw std::runtime_error("planes must be [>=1, 5, 5, 15]");
            return s.stepSelect(planes.mutable_data());
        }, "1 = leaf needs the network (planes[0] written), 2 = value already known, 0 = finished")
        .def("step_expand", [](Search& s, arr<float> logits, float value) { s.stepExpand(logits.data(), value); })
        .def("step_expand_known", &Search::stepExpandKnown)
        .def("step_backup", &Search::stepBackup)
        .def_property_readonly("step_phase", &Search::stepPhase)
        .def_property_readonly("step_leaf", &Search::stepLeafNode)
        .def_property_readonly("step_value", &Search::stepLeafValue)
        .def_property_readonly("step_nodes", &Search::stepNodes)
        .def_property_readonly("node_count", &Search::nodeCount)
        .def("tree", [](Search& s, int root, int maxNodes, float prunePct, int maxDepth, bool unvisited) {
            auto items = s.exportTree(root, maxNodes, prunePct, maxDepth, unvisited);
            py::list out;
            for (auto& t : items) {
                py::dict d;
                d["id"] = t.id; d["parent"] = t.parent == INT_MIN ? py::object(py::none()) : py::object(py::int_(t.parent));
                d["depth"] = t.depth; d["n"] = t.N; d["q"] = t.Q; d["p"] = t.P; d["u"] = t.U;
                d["status"] = t.status; d["expanded"] = t.expanded;
                d["net"] = std::isnan(t.netV) ? py::object(py::none()) : py::object(py::float_(t.netV));
                d["move"] = t.hasMove ? py::object(py::make_tuple((int)t.m.f, (int)t.m.t, (int)t.m.b)) : py::object(py::none());
                py::list h;
                for (int c = 0; c < 25; c++) h.append(t.p.height(c));
                d["heights"] = h;
                d["me"] = py::make_tuple(lsb(t.p.me), lsb(t.p.me & (t.p.me - 1)));
                d["op"] = py::make_tuple(lsb(t.p.op), lsb(t.p.op & (t.p.op - 1)));
                d["pv"] = t.onPV; d["step"] = t.onStep; d["moves"] = t.childCount;
                out.append(d);
            }
            return out;
        }, py::arg("root") = 0, py::arg("max_nodes") = 150, py::arg("prune_pct") = 100.0f,
           py::arg("max_depth") = 8, py::arg("unvisited") = false,
           "Nodes of the search tree for display, most visited first, parents before children.");

    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("n_games", &Config::nGames)
        .def_readwrite("sims", &Config::sims)
        .def_readwrite("cpuct", &Config::cpuct)
        .def_readwrite("dir_alpha", &Config::dirAlpha)
        .def_readwrite("dir_frac", &Config::dirFrac)
        .def_readwrite("noise", &Config::noise)
        .def_readwrite("temp_plies", &Config::tempPlies)
        .def_readwrite("label_min_blocks", &Config::labelMinBlocks)
        .def_readwrite("root_budget", &Config::rootBudget)
        .def_readwrite("child_budget", &Config::childBudget)
        .def_readwrite("threads", &Config::threads)
        .def_readwrite("tt_gb", &Config::ttGb)
        .def_readwrite("max_games", &Config::maxGames)
        .def_readwrite("arena", &Config::arena)
        .def_readwrite("seed", &Config::seed);

    py::class_<SelfPlay>(m, "SelfPlay")
        .def(py::init<const Config&>())
        .def("prepare", [](SelfPlay& sp, arr<float> planes) {
            if (planes.ndim() != 4 || planes.shape(0) < sp.cfg.nGames)
                throw std::runtime_error("planes buffer must be [n_games, 5, 5, 15]");
            std::vector<int32_t> gi(sp.cfg.nGames), ni(sp.cfg.nGames);
            int cnt;
            float* P = planes.mutable_data();
            {
                py::gil_scoped_release nogil;
                cnt = sp.prepare(P, gi.data(), ni.data());
            }
            gi.resize(cnt);
            ni.resize(cnt);
            return py::make_tuple(cnt, arr<int32_t>(cnt, gi.data()), arr<int32_t>(cnt, ni.data()));
        }, "Advance games; writes leaf planes into the buffer. Returns (count, game_idx, net_id).")
        .def("apply", [](SelfPlay& sp, arr<int32_t> gameIdx, arr<float> logits, arr<float> values) {
            int cnt = (int)gameIdx.size();
            if (logits.ndim() != 2 || logits.shape(0) < cnt || logits.shape(1) != NACT || values.size() < cnt)
                throw std::runtime_error("logits must be [>=count, 153], values [>=count]");
            py::gil_scoped_release nogil;
            sp.apply(cnt, gameIdx.data(), logits.data(), values.data());
        })
        .def("harvest", [](SelfPlay& sp) {
            auto smp = sp.harvest();
            py::ssize_t n = smp.size();
            arr<int8_t> st({n, (py::ssize_t)STATE_LEN});
            arr<float> pi({n, (py::ssize_t)NMOVE});
            arr<float> q(n), z(n);
            arr<int8_t> ex(n), v(n), pl(n);
            for (py::ssize_t i = 0; i < n; i++) {
                memcpy(st.mutable_data() + i * STATE_LEN, smp[i].state, STATE_LEN);
                memcpy(pi.mutable_data() + i * NMOVE, smp[i].pi, sizeof(float) * NMOVE);
                q.mutable_data()[i] = smp[i].q;
                z.mutable_data()[i] = smp[i].z;
                ex.mutable_data()[i] = smp[i].exact;
                v.mutable_data()[i] = smp[i].v;
                pl.mutable_data()[i] = smp[i].player;
            }
            py::dict d;
            d["state"] = st; d["pi"] = pi; d["q"] = q; d["z"] = z;
            d["exact"] = ex; d["v"] = v; d["player"] = pl;
            return d;
        }, "Samples from finished games since the last call.")
        .def_property_readonly("games_started", &SelfPlay::gamesStarted)
        .def_property_readonly("games_finished", &SelfPlay::gamesFinished)
        .def_property_readonly("active_games", &SelfPlay::activeGames)
        .def_property_readonly("net_wins", [](SelfPlay& sp) { return py::make_tuple(sp.netWins[0], sp.netWins[1]); })
        .def_property_readonly("first_player_wins", [](SelfPlay& sp) { return sp.firstPlayerWins; })
        .def_property_readonly("game_lengths", [](SelfPlay& sp) { return sp.gameLengths; })
        .def_property_readonly("stats", [](SelfPlay& sp) {
            py::dict d;
            d["games_started"] = sp.gamesStarted(); d["games_finished"] = sp.gamesFinished();
            d["active_games"] = sp.activeGames(); d["pending_samples"] = sp.pendingSamples();
            d["moves_played"] = sp.movesPlayed; d["root_solves"] = sp.rootSolves;
            d["root_proven"] = sp.rootProven; d["solver_moves"] = sp.solverMoves;
            d["leaf_evals"] = sp.leafEvals;
            return d;
        }, "Counters: moves played, exact root solves (and how many were proven), moves played "
           "straight from a proof, network leaf evaluations.");
}