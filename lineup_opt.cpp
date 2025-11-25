#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
using namespace std;


/*
  optimize_lineup_winprob.cpp

  - Stage1: Expected runs per half-inning (ER) via value iteration with optimal action (swing/bunt/steal)
            Beam search to shortlist top orders
  - Stage2: Game-level DP on reachable states (visitor acts with 3 actions, home swings only)
  - No external libs. C++17 only.

  Build:
    g++ -O3 -std=c++17 lineup_opt.cpp -o lineup_opt

  Run (defaults similar to Python):
    ./lineup_opt --topk 50 --beam 30 --iters 150 --randseeds 20 --topfinal 10
*/

// =============================================================================
// Constants & settings
// =============================================================================
static constexpr int   COLD_FROM_INNING = 5;
static constexpr int   COLD_DIFF        = 8;
static constexpr int   MAX_DIFF         = 12;

static constexpr int   MAX_SWEEPS_ER    = 300;     // half-inning ER VI
static constexpr double TOL_ER          = 1e-10;

static constexpr int   MAX_SWEEPS_GAME  = 150;     // game VI
static constexpr double TOL_GAME        = 1e-8;

static constexpr int   MAX_A            = 3;       // actions for away: 0 swing, 1 bunt, 2 steal
static constexpr int   MAX_BRANCH       = 6;       // up to 6 for swing

// =============================================================================
// Players data (Hawks 2014, Table 1)
// =============================================================================
struct Player {
  // probabilities
  double OUT, _1B, _2B, _3B, HR, BB, SAC, SB;
};

static unordered_map<string, Player> players = {
  {"Y. Honda",    {0.648, 0.217, 0.032, 0.016, 0.000, 0.087, 0.941, 0.793}},
  {"A. Nakamura", {0.627, 0.231, 0.035, 0.006, 0.006, 0.095, 0.800, 0.833}},
  {"Y. Yanagita", {0.593, 0.211, 0.029, 0.007, 0.025, 0.136, 0.000, 0.846}},
  {"S. Uchikawa", {0.653, 0.199, 0.049, 0.002, 0.034, 0.063, 0.000, 0.0  }},
  {"Lee Dae-Ho",  {0.637, 0.195, 0.048, 0.000, 0.031, 0.090, 0.000, 0.0  }},
  {"Y. Hasegawa", {0.624, 0.193, 0.056, 0.006, 0.011, 0.110, 0.000, 0.500}},
  {"N. Matsuda",  {0.655, 0.185, 0.048, 0.007, 0.043, 0.062, 0.500, 0.667}},
  {"S. Tsuruoka", {0.750, 0.167, 0.024, 0.018, 0.000, 0.042, 0.944, 0.0  }},
  {"K. Imamiya",  {0.698, 0.174, 0.044, 0.002, 0.005, 0.077, 0.873, 0.667}}
};

static vector<string> default_lineup = {
  "Y. Honda","A. Nakamura","Y. Yanagita","S. Uchikawa","Lee Dae-Ho","Y. Hasegawa","N. Matsuda","S. Tsuruoka","K. Imamiya"
};
static vector<string> worst_lineup = {
  "S. Tsuruoka","Y. Hasegawa","K. Imamiya","Y. Honda","N. Matsuda","S. Uchikawa","Lee Dae-Ho","A. Nakamura","Y. Yanagita"
};
static vector<string> opt_lineup_doc = {
  "A. Nakamura","Y. Yanagita","S. Uchikawa","Lee Dae-Ho","Y. Hasegawa","N. Matsuda","Y. Honda","S. Tsuruoka","K. Imamiya"
};

// =============================================================================
// Utilities
// =============================================================================
inline int clamp_diff(int v) {
  if (v < -MAX_DIFF) return -MAX_DIFF;
  if (v >  MAX_DIFF) return  MAX_DIFF;
  return v;
}
inline bool has_runner(int bases, int base) { // base: 1..3
  return ((bases >> (base-1)) & 1) == 1;
}
inline int set_runner(int bases, int base, bool present) {
  int mask = (1 << (base-1));
  return present ? (bases | mask) : (bases & ~mask);
}
inline int prev_idx(int bidx) { return (bidx + 8) % 9; } // (bidx-1)%9

// =============================================================================
// Base transitions
// =============================================================================
static pair<int,int> walk_transition(int bases) {
  int runs = (has_runner(bases,1) && has_runner(bases,2) && has_runner(bases,3)) ? 1 : 0;
  int nb = 0;
  nb = set_runner(nb,3, has_runner(bases,3) || has_runner(bases,2));
  nb = set_runner(nb,2, has_runner(bases,2) || has_runner(bases,1));
  nb = set_runner(nb,1, true);
  return {nb, runs};
}
static pair<int,int> single_transition(int bases) {
  int runs = (int)has_runner(bases,2) + (int)has_runner(bases,3);
  int nb = 0;
  if (has_runner(bases,1)) nb = set_runner(nb,3, true);
  nb = set_runner(nb,1, true);
  return {nb, runs};
}
static pair<int,int> double_transition(int bases) {
  int runs = (int)has_runner(bases,1) + (int)has_runner(bases,2) + (int)has_runner(bases,3);
  int nb = 0;
  nb = set_runner(nb,2, true);
  return {nb, runs};
}
static pair<int,int> triple_transition(int bases) {
  int runs = (int)has_runner(bases,1) + (int)has_runner(bases,2) + (int)has_runner(bases,3);
  int nb = 0;
  nb = set_runner(nb,3, true);
  return {nb, runs};
}
static pair<int,int> hr_transition(int bases) {
  int runs = 1 + (int)has_runner(bases,1) + (int)has_runner(bases,2) + (int)has_runner(bases,3);
  return {0, runs};
}
static pair<int,int> bunt_success_transition(int bases) {
  bool r1 = has_runner(bases,1);
  bool r2 = has_runner(bases,2);
  bool r3 = has_runner(bases,3);

  int runs = 0;
  int nb = 0;

  // runner on 3rd scores
  if (r3) {
    runs += 1;
  }
  // runner on 2nd -> 3rd
  if (r2) {
    nb = set_runner(nb,3,true);
  }
  // runner on 1st -> 2nd
  if (r1) {
    nb = set_runner(nb,2,true);
  }

  // batter is out and does NOT occupy any base
  return {nb, runs};
}
static pair<int,int> bunt_fail_transition(int bases) {
  bool r1 = has_runner(bases,1);
  bool r2 = has_runner(bases,2);
  bool r3 = has_runner(bases,3);
  int nb = 0;
  if (r3) {
    if (r2) nb = set_runner(nb,3,true);
    if (r1) nb = set_runner(nb,2,true);
  } else if (r2) {
    if (r1) nb = set_runner(nb,2,true);
  } else if (r1) {
    // runner on 1st out; others none
  }
  nb = set_runner(nb,1,true); // batter to 1st
  return {nb, 0};
}

static bool bunt_allowed(int bases, int outs, int run_diff=0) {
  if (outs > 1) return false;
  if (bases == 0) return false;
  return (run_diff < 3); // don't bunt if leading by >=3
}
static bool steal_allowed(int bases, int outs) {
  return has_runner(bases,1) && (!has_runner(bases,2));
}

// =============================================================================
// Half-inning transitions (precompile) and value iteration for ER
// =============================================================================
struct KeyHalf {
  int bidx, bases, outs;
  bool operator==(const KeyHalf& o) const {
    return bidx==o.bidx && bases==o.bases && outs==o.outs;
  }
};
struct KeyHalfHash {
  size_t operator()(const KeyHalf& k) const {
    return (k.bidx*97u) ^ (k.bases*17u) ^ (k.outs*131u);
  }
};
struct BranchHalf {
  double p;
  int next_outs;
  int next_bases;
  int runs;
};

static void compile_half_transitions(
  const array<Player,9>& lineup,
  unordered_map<KeyHalf, vector<BranchHalf>, KeyHalfHash>& SW,
  unordered_map<KeyHalf, vector<BranchHalf>, KeyHalfHash>& BU,
  unordered_map<KeyHalf, vector<BranchHalf>, KeyHalfHash>& ST
) {
  SW.clear(); BU.clear(); ST.clear();
  for (int bidx=0;bidx<9;++bidx){
    const auto& batter = lineup[bidx];
    for (int bases=0;bases<8;++bases){
      for (int outs=0;outs<=2;++outs){
        KeyHalf key{bidx,bases,outs};
        vector<BranchHalf> dist;

        auto add_swing = [&](double p, int no, int nb, int r){
          if (p>0) dist.push_back({p, min(no,3), nb, r});
        };

        add_swing(batter.OUT, outs+1, bases, 0);
        if (batter.BB>0){
          auto w = walk_transition(bases);
          add_swing(batter.BB, outs, w.first, w.second);
        }
        if (batter._1B>0){
          auto tr = single_transition(bases);
          add_swing(batter._1B, outs, tr.first, tr.second);
        }
        if (batter._2B>0){
          auto tr = double_transition(bases);
          add_swing(batter._2B, outs, tr.first, tr.second);
        }
        if (batter._3B>0){
          auto tr = triple_transition(bases);
          add_swing(batter._3B, outs, tr.first, tr.second);
        }
        if (batter.HR>0){
          auto tr = hr_transition(bases);
          add_swing(batter.HR, outs, tr.first, tr.second);
        }
        SW[key]=dist;

        // bunt
        if (bunt_allowed(bases,outs) && batter.SAC>0){
          auto bs = bunt_success_transition(bases);
          auto bf = bunt_fail_transition(bases);
          BU[key] = {
            // success: all runners advance 1 base, batter out
            {batter.SAC,               min(outs+1,3), bs.first, bs.second},
            // fail: leading runner out, others advance 1, batter to 1st
            {1.0 - batter.SAC,         min(outs+1,3), bf.first, 0}
          };
        } else {
          BU[key] = SW[key];
        }

        // steal: runner uses prev batter's SB
        if (steal_allowed(bases,outs)){
          const auto& runner = lineup[prev_idx(bidx)];
          double ps = runner.SB;
          if (ps>0.0){
            int nb_s = set_runner(bases,1,false);
            nb_s = set_runner(nb_s,2,true);
            int nb_f = set_runner(bases,1,false);
            ST[key] = {
              {ps, outs, nb_s, 0},
              {1.0-ps, min(outs+1,3), nb_f, 0}
            };
          } else {
            ST[key] = SW[key];
          }
        } else {
          ST[key] = SW[key];
        }
      }
    }
  }
}

static double expected_runs_half_fast(const vector<string>& order_names){
  array<Player,9> lineup;
  for (int i=0;i<9;++i) lineup[i]=players.at(order_names[i]);

  unordered_map<KeyHalf, vector<BranchHalf>, KeyHalfHash> SW,BU,ST;
  compile_half_transitions(lineup, SW,BU,ST);

  // V[outs][bases][bidx]
  double V[4][8][9] = {}; // terminal outs=3 included
  for (int sweep=0; sweep<MAX_SWEEPS_ER; ++sweep){
    double delta = 0.0;
    for (int outs=0;outs<=2;++outs){
      for (int bases=0;bases<8;++bases){
        for (int bidx=0;bidx<9;++bidx){
          int nbidx = (bidx+1)%9;
          KeyHalf key{bidx,bases,outs};

          auto Q = [&](const vector<BranchHalf>& dist){
            double s=0.0;
            for (const auto& br: dist){
              if (br.next_outs>=3){
                s += br.p * br.runs;
              } else {
                s += br.p * (br.runs + V[br.next_outs][br.next_bases][nbidx]);
              }
            }
            return s;
          };

          double newv = max({Q(SW[key]), Q(BU[key]), Q(ST[key])});
          double old = V[outs][bases][bidx];
          delta = max(delta, fabs(newv-old));
          V[outs][bases][bidx]=newv;
        }
      }
    }
    if (delta < TOL_ER) break;
  }
  return V[0][0][0];
}

// =============================================================================
// Stage1: Beam search by half-inning ER
// =============================================================================
struct Cand { double er; array<string,9> order; };

static vector<pair<double, vector<string>>> shortlist_by_halfER_beam(
  const vector<string>& names, int top_k=50, int beam=30, int iters=150, int rand_seeds=20
){
  vector<vector<string>> seeds;
  seeds.push_back(default_lineup);
  seeds.push_back(worst_lineup);
  seeds.push_back(opt_lineup_doc);
  std::mt19937_64 rng(42);

  for (int t=0;t<rand_seeds;++t){
    vector<string> perm = names;
    shuffle(perm.begin(), perm.end(), rng);
    seeds.push_back(std::move(perm));
  }
  struct VecHash {
    size_t operator()(const vector<string>& v) const {
      size_t h=0;
      for (auto& s: v) h ^= hash<string>{}(s)+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
      return h;
    }
  };
  unordered_set<vector<string>, VecHash> seen;

  vector<pair<double, vector<string>>> beam_list;
  beam_list.reserve(beam + rand_seeds + 3);

  for (auto& ord0: seeds){
    if (!seen.insert(ord0).second) continue;
    double er0 = expected_runs_half_fast(ord0);
    beam_list.push_back({er0, ord0});
  }
  sort(beam_list.begin(), beam_list.end(), [](auto& a, auto& b){return a.first>b.first;});
  if ((int)beam_list.size()>beam) beam_list.resize(beam);

  uniform_int_distribution<int> dist01(0,8);
  for (int it=0; it<iters; ++it){
    vector<pair<double, vector<string>>> cand;
    for (auto& [er, order] : beam_list){
      for (int rep=0; rep<4; ++rep){
        int i=dist01(rng), j=dist01(rng);
        if (i==j) { j=(j+1)%9; }
        auto neword = order;
        swap(neword[i], neword[j]);
        if (!seen.insert(neword).second) continue;
        double er_new = expected_runs_half_fast(neword);
        cand.push_back({er_new, std::move(neword)});
      }
    }
    for (auto& c: cand) beam_list.push_back(std::move(c));
    sort(beam_list.begin(), beam_list.end(), [](auto& a, auto& b){return a.first>b.first;});
    if ((int)beam_list.size()>beam) beam_list.resize(beam);
  }

  sort(beam_list.begin(), beam_list.end(), [](auto& a, auto& b){return a.first>b.first;});
  if ((int)beam_list.size()>top_k) beam_list.resize(top_k);

  return beam_list;
}

// =============================================================================
// Stage2: Game transitions (dict) and DP (dict V) — Python fast version equivalent
// =============================================================================
struct State {
  int inn, half, outs, bases, bA, bH, rd;
  // runner on 1st for the batting team
  // -1: no runner on 1st, 0..8: index in the batting team's lineup array
  int r1;
  bool operator==(const State& o) const {
    return inn==o.inn && half==o.half && outs==o.outs &&
           bases==o.bases && bA==o.bA && bH==o.bH &&
           rd==o.rd && r1==o.r1;
  }
};
struct StateHash {
  size_t operator()(const State& s) const {
    size_t h = s.inn;
    h = h*131 + s.half;
    h = h*131 + s.outs;
    h = h*131 + s.bases;
    h = h*131 + s.bA;
    h = h*131 + s.bH;
    h = h*131 + (s.rd + 32);
    h = h*131 + (s.r1 + 2); // shift -1 to positive
    return h;
  }
};


enum R1Mode {
  R1_KEEP = 0,        // keep current runner on 1st
  R1_SET_BATTER = 1,  // set runner on 1st to current batter
  R1_CLEAR = 2        // clear runner on 1st (no runner)
};

struct BranchFull {
  double p;
  int no;
  int nb;
  int runs;
  int r1_mode; // one of R1Mode
};


static pair<int,int> next_half(int inn, int half){
  return (half==0) ? make_pair(inn, 1) : make_pair(inn+1, 0);
}
static pair<bool,double> terminal_value(int inn, int half, int outs, int rd){
  // cold
  if (inn >= COLD_FROM_INNING && half == 1 && abs(rd) >= COLD_DIFF){
    return {true, (rd>0)?1.0:0.0};
  }
  // 9th top end: if home leads, game ends
  if (inn==9 && half==0 && outs==3 && rd<0){
    return {true, 0.0};
  }
  // 9th bottom end
  if (inn==9 && half==1 && outs==3){
    if (rd>0) return {true,1.0};
    if (rd<0) return {true,0.0};
    return {true,0.5};
  }
  return {false,0.0};
}

struct KeyFull {
  // side "A"/"H" is implicit by half (0=away,1=home) but we prebuild with tags like Python; C++ uses these keys directly per side
  int bidx, bases, outs;
  bool operator==(const KeyFull& o) const {
    return bidx==o.bidx && bases==o.bases && outs==o.outs;
  }
};
struct KeyFullHash {
  size_t operator()(const KeyFull& k) const {
    return (k.bidx*97u) ^ (k.bases*17u) ^ (k.outs*131u);
  }
};

static vector<BranchFull> swing_transitions_full(const Player& batter, int bases, int outs){
  vector<BranchFull> dist;
  auto add = [&](double p, int no, int nb, int r, int r1_mode){
    if (p>0) dist.push_back({p, min(no,3), nb, r, r1_mode});
  };

  // out: runner on 1st stays
  add(batter.OUT, outs+1, bases, 0, R1_KEEP);

  // walk: old runner on 1st goes to 2nd, batter to 1st
  if (batter.BB>0) {
    auto w = walk_transition(bases);
    add(batter.BB, outs, w.first, w.second, R1_SET_BATTER);
  }

  // single: runner on 1st goes to 3rd, batter to 1st
  if (batter._1B>0){
    auto tr = single_transition(bases);
    add(batter._1B, outs, tr.first, tr.second, R1_SET_BATTER);
  }

  // double: all runners score, batter to 2nd → 1st is empty
  if (batter._2B>0){
    auto tr = double_transition(bases);
    add(batter._2B, outs, tr.first, tr.second, R1_CLEAR);
  }

  // triple: all runners score, batter to 3rd → 1st is empty
  if (batter._3B>0){
    auto tr = triple_transition(bases);
    add(batter._3B, outs, tr.first, tr.second, R1_CLEAR);
  }

  // HR: all score, bases empty
  if (batter.HR>0){
    auto tr = hr_transition(bases);
    add(batter.HR, outs, tr.first, tr.second, R1_CLEAR);
  }

  return dist;
}

static vector<BranchFull> bunt_transitions_full(const Player& batter, int bases, int outs){
  if (!(bunt_allowed(bases,outs) && batter.SAC>0)) {
    return swing_transitions_full(batter, bases, outs);
  }

  auto bs = bunt_success_transition(bases);
  auto bf = bunt_fail_transition(bases);

  vector<BranchFull> dist;
  // success: all runners advance 1 base, batter out → no runner on 1st
  dist.push_back({batter.SAC,               min(outs+1,3), bs.first,  bs.second, R1_CLEAR});
  // fail: leading runner out, others advance 1, batter becomes runner on 1st
  dist.push_back({1.0 - batter.SAC,         min(outs+1,3), bf.first,  0,         R1_SET_BATTER});
  return dist;
}

struct TransFull {
  array<Player,9> lineup;
  unordered_map<KeyFull, vector<BranchFull>, KeyFullHash> SW, BU;
};

static vector<BranchFull> steal_transitions_full(const TransFull& T, const State& s){
  vector<BranchFull> dist;

  // steal not allowed or no runner on 1st → no-op
  if (!steal_allowed(s.bases, s.outs) || s.r1 < 0) {
    dist.push_back({1.0, s.outs, s.bases, 0, R1_KEEP});
    return dist;
  }

  int runner_idx = s.r1;
  double ps = T.lineup[runner_idx].SB;
  if (ps <= 0.0) {
    dist.push_back({1.0, s.outs, s.bases, 0, R1_KEEP});
    return dist;
  }

  int nb_s = set_runner(s.bases,1,false);
  nb_s = set_runner(nb_s,2,true);
  int nb_f = set_runner(s.bases,1,false);

  dist.push_back({ps,            s.outs,             nb_s, 0, R1_CLEAR});
  dist.push_back({1.0 - ps,      min(s.outs+1,3),    nb_f, 0, R1_CLEAR});

  return dist;
}

static TransFull compile_game_transitions(const array<Player,9>& lineup){
  TransFull T;
  T.lineup = lineup;

  for (int bidx=0;bidx<9;++bidx){
    for (int bases=0;bases<8;++bases){
      for (int outs=0;outs<=2;++outs){
        KeyFull key{bidx,bases,outs};
        T.SW[key] = swing_transitions_full(lineup[bidx], bases, outs);
        T.BU[key] = bunt_transitions_full(lineup[bidx], bases, outs);
      }
    }
  }
  return T;
}

static vector<State> reachable_states(const TransFull& A, const TransFull& H){
  State start{1,0,0,0,0,0,0,-1}; // no runner on 1st at start
  deque<State> Q; Q.push_back(start);
  unordered_set<State, StateHash> seen; seen.insert(start);

  while (!Q.empty()){
    State s = Q.front(); Q.pop_front();
    auto [term, _tv] = terminal_value(s.inn, s.half, s.outs, s.rd);
    if (term) continue;

    if (s.outs >= 3){
      auto nh = next_half(s.inn, s.half);
      // new half: bases empty, no runner on 1st
      State s2{nh.first, nh.second, 0, 0, s.bA, s.bH, s.rd, -1};
      if (!seen.count(s2)){ seen.insert(s2); Q.push_back(s2); }
      continue;
    }

    if (s.half==0){
      // away acts: 3 actions (swing, bunt, steal)
      KeyFull key{s.bA, s.bases, s.outs};
      int nbA = (s.bA+1)%9;

      auto apply_branch = [&](const BranchFull& br){
        if (br.p==0.0) return;
        int nrd = clamp_diff(s.rd + br.runs);
        auto t2 = terminal_value(s.inn, s.half, br.no, nrd);
        int nr1;
        if (br.r1_mode == R1_KEEP)       nr1 = s.r1;
        else if (br.r1_mode == R1_SET_BATTER) nr1 = s.bA;
        else                              nr1 = -1;

        if (t2.first) return; // terminal → no successor state for BFS
        State s2{s.inn, s.half, br.no, br.nb, nbA, s.bH, nrd, nr1};
        if (!seen.count(s2)){ seen.insert(s2); Q.push_back(s2); }
      };

      const auto& v0 = A.SW.at(key);
      const auto& v1 = A.BU.at(key);
      auto v2 = steal_transitions_full(A, s);

      for (const auto& br: v0) apply_branch(br);
      for (const auto& br: v1) apply_branch(br);
      for (const auto& br: v2) apply_branch(br);

    } else {
      // home swings only
      KeyFull key{s.bH, s.bases, s.outs};
      int nbH = (s.bH+1)%9;
      const auto& branches = H.SW.at(key);
      for (const auto& br: branches){
        if (br.p==0.0) continue;
        int nrd = clamp_diff(s.rd - br.runs);
        auto t2 = terminal_value(s.inn, s.half, br.no, nrd);
        int nr1;
        if (br.r1_mode == R1_KEEP)       nr1 = s.r1;
        else if (br.r1_mode == R1_SET_BATTER) nr1 = s.bH;
        else                              nr1 = -1;

        if (t2.first) continue;
        State s2{s.inn, s.half, br.no, br.nb, s.bA, nbH, nrd, nr1};
        if (!seen.count(s2)){ seen.insert(s2); Q.push_back(s2); }
      }
    }
  }

  // dump to vector
  vector<State> R; R.reserve(seen.size());
  for (const auto& st: seen) R.push_back(st);
  // for reproducibility:
  sort(R.begin(), R.end(), [](const State& a, const State& b){
    return tie(a.inn,a.half,a.outs,a.bases,a.bA,a.bH,a.rd) < tie(b.inn,b.half,b.outs,b.bases,b.bA,b.bH,b.rd);
  });
  return R;
}

static pair<double, unordered_map<State,double,StateHash>>
evaluate_winprob_fast(const vector<string>& away_names, const vector<string>& home_names, const string& log_prefix="")
{
  array<Player,9> away, home;
  for (int i=0;i<9;++i){ away[i]=players.at(away_names[i]); home[i]=players.at(home_names[i]); }

  auto TA = compile_game_transitions(away);
  auto TH = compile_game_transitions(home);

  auto R = reachable_states(TA, TH);
  cerr << log_prefix << "Reachable states: " << R.size() << "\n";

  unordered_map<State,double,StateHash> V;
  V.reserve(R.size()*2);
  for (auto& s: R) V[s]=0.0;

  for (int sweep=1; sweep<=MAX_SWEEPS_GAME; ++sweep){
    double delta=0.0;

    for (const auto& s: R){
      auto [term, tv] = terminal_value(s.inn, s.half, s.outs, s.rd);
      double newv=0.0;

      if (term){
        newv = tv;
      } else if (s.outs==3){
        auto nh = next_half(s.inn, s.half);
        State s2{nh.first, nh.second, 0, 0, s.bA, s.bH, s.rd, -1};
        auto it = V.find(s2);
        newv = (it==V.end()) ? tv : it->second;
        } else if (s.half==0){
        // away: max over 3 actions (swing, bunt, steal)
        int nbA = (s.bA+1)%9;

        auto q_of = [&](const vector<BranchFull>& brs){
          double q=0.0;
          for (const auto& br: brs){
            if (br.p==0.0) continue;
            int nrd = clamp_diff(s.rd + br.runs);
            auto t2 = terminal_value(s.inn, s.half, br.no, nrd);

            int nr1;
            if (br.r1_mode == R1_KEEP)       nr1 = s.r1;
            else if (br.r1_mode == R1_SET_BATTER) nr1 = s.bA;
            else                              nr1 = -1;

            if (t2.first) q += br.p * t2.second;
            else {
              State s2{s.inn, s.half, br.no, br.nb, nbA, s.bH, nrd, nr1};
              q += br.p * V[s2];
            }
          }
          return q;
        };

        KeyFull key{s.bA,s.bases,s.outs};
        double q0 = q_of(TA.SW.at(key));
        double q1 = q_of(TA.BU.at(key));
        auto steal_branches = steal_transitions_full(TA, s);
        double q2 = q_of(steal_branches);

        newv = max({q0,q1,q2});
      } else {
        // home swing only
        int nbH = (s.bH+1)%9;
        double q=0.0;
        KeyFull key{s.bH,s.bases,s.outs};
        for (const auto& br: TH.SW.at(key)){
          if (br.p==0.0) continue;
          int nrd = clamp_diff(s.rd - br.runs);
          auto t2 = terminal_value(s.inn, s.half, br.no, nrd);

          int nr1;
          if (br.r1_mode == R1_KEEP)       nr1 = s.r1;
          else if (br.r1_mode == R1_SET_BATTER) nr1 = s.bH;
          else                              nr1 = -1;

          if (t2.first) q += br.p * t2.second;
          else {
            State s2{s.inn, s.half, br.no, br.nb, s.bA, nbH, nrd, nr1};
            q += br.p * V[s2];
          }
        }
        newv = q;
      }

      double old = V[s];
      V[s] = newv;
      delta = max(delta, fabs(newv-old));
    }

    if (sweep==1 || sweep%20==0){
      cerr << log_prefix << "  sweep " << setw(3) << sweep << ": Δ=" << scientific << setprecision(3) << delta << "\n";
    }
    if (delta < TOL_GAME) break;
  }

  State start{1,0,0,0,0,0,0,-1};
  return {V[start], std::move(V)};
}

// =============================================================================
// Orchestration
// =============================================================================
struct Ranked { double wp, er, t; array<string,9> order; };

static vector<Ranked> optimize_lineup_by_winprob(
  const vector<string>& names,
  const vector<string>& home_ref,
  int top_k=50, int beam=30, int iters=150, int rand_seeds=20, int top_final=10
){
  auto t0 = chrono::high_resolution_clock::now();
  auto shortlist = shortlist_by_halfER_beam(names, top_k, beam, iters, rand_seeds);
  auto t1 = chrono::high_resolution_clock::now();
  cerr << "[Time] Stage1 (beam="<<beam<<", iters="<<iters<<", top_k="<<top_k<<"): "
       << chrono::duration<double>(t1-t0).count() << "s\n";

  vector<Ranked> ranked;
  int rank=0;
  for (auto& item: shortlist){
    ++rank;
    auto ts = chrono::high_resolution_clock::now();
    auto [wp, _V] = evaluate_winprob_fast(item.second, home_ref, "[Cand "+to_string(rank)+"] ");
    auto te = chrono::high_resolution_clock::now();
    double spent = chrono::duration<double>(te-ts).count();

    Ranked r; r.wp=wp; r.er=item.first; r.t=spent;
    for (int i=0;i<9;++i) r.order[i]=item.second[i];
    ranked.push_back(std::move(r));

    cout << fixed << setprecision(6);
    cout << "[Cand " << setw(2) << setfill('0') << rank << "] "
         << "WP="<< setprecision(6) << wp
         << "  ER_half="<< setprecision(6) << item.first
         << "  time="<< setprecision(2) << spent
         << "s  lineup=[";
    for (int i=0;i<9;++i){ cout << r.order[i] << (i==8?"]\n":", "); }
  }

  sort(ranked.begin(), ranked.end(), [](const Ranked& a, const Ranked& b){return a.wp>b.wp;});
  auto t2 = chrono::high_resolution_clock::now();
  cerr << "[Time] Stage2 (K="<<shortlist.size()<<"): " << chrono::duration<double>(t2-t1).count() << "s\n";
  cerr << "[Time] Total: " << chrono::duration<double>(t2-t0).count() << "s\n";

  if ((int)ranked.size()>top_final) ranked.resize(top_final);
  return ranked;
}

// =============================================================================
// CLI
// =============================================================================
static int get_flag_int(char** begin, char** end, const string& flag, int defv){
  char** it = std::find(begin, end, flag);
  if (it!=end && ++it!=end) return stoi(*it);
  return defv;
}

int main(int argc, char** argv){
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  int topk     = get_flag_int(argv, argv+argc, string("--topk"),     50);
  int beam     = get_flag_int(argv, argv+argc, string("--beam"),     30);
  int iters    = get_flag_int(argv, argv+argc, string("--iters"),    150);
  int randseeds= get_flag_int(argv, argv+argc, string("--randseeds"),20);
  int topfinal = get_flag_int(argv, argv+argc, string("--topfinal"), 10);

  vector<string> names = default_lineup;
  vector<string> home_ref = default_lineup;

  auto result = optimize_lineup_by_winprob(names, home_ref, topk, beam, iters, randseeds, topfinal);

  cout << "\n=== Best lineups by Win Probability (visitor vs Default home; steal=paper-like) ===\n";
  int i=0;
  for (const auto& r: result){
    cout << "[" << setw(2) << setfill(' ') << (++i) << "] "
         << "WP=" << fixed << setprecision(9) << r.wp
         << "  ER_half=" << fixed << setprecision(6) << r.er
         << "  time=" << fixed << setprecision(2) << r.t << "s  lineup=[";
    for (int k=0;k<9;++k) cout << r.order[k] << (k==8?"]\n":", ");
  }
  return 0;
}