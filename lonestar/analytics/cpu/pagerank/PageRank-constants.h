#ifndef LONESTAR_PAGERANK_CONSTANTS_H
#define LONESTAR_PAGERANK_CONSTANTS_H

#include <iostream>

#define DEBUG 0

static const char* name = "Page Rank";
static const char* url  = nullptr;

//! All PageRank algorithm variants use the same constants for ease of
//! comparison.
constexpr static const float ALPHA         = 0.85;
constexpr static const float INIT_RESIDUAL = 1 - ALPHA;

constexpr static const float TOLERANCE   = 1.0e-3;
constexpr static const unsigned MAX_ITER = 1000;

constexpr static const unsigned PRINT_TOP = 20;

namespace cll = llvm::cl;

static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"),
                                 cll::init(TOLERANCE));
static cll::opt<unsigned int> maxIterations(
    "maxIterations",
    cll::desc("Maximum iterations, applies round-based versions only"),
    cll::init(MAX_ITER));

//! Type definitions.
typedef float PRTy;

template <typename GNode>
struct TopPair {
  float value;
  GNode id;

  TopPair(float v, GNode i) : value(v), id(i) {}

  bool operator<(const TopPair& b) const {
    if (value == b.value)
      return id > b.id;
    return value < b.value;
  }
};

//! Helper functions.

PRTy atomicAdd(std::atomic<PRTy>& v, PRTy delta) {
  PRTy old;
  do {
    old = v;
  } while (!v.compare_exchange_strong(old, old + delta));
  return old;
}

template <typename Graph>
void printTop(Graph& graph, unsigned topn = PRINT_TOP) {

  using GNode = typename Graph::GraphNode;
  typedef TopPair<GNode> Pair;
  typedef std::map<Pair, GNode> TopMap;

  TopMap top;

  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src  = *ii;
    auto& n    = graph.getData(src);
    PRTy value = n.value;
    Pair key(value, src);

    if (top.size() < topn) {
      top.insert(std::make_pair(key, src));
      continue;
    }

    if (top.begin()->first < key) {
      top.erase(top.begin());
      top.insert(std::make_pair(key, src));
    }
  }

  int rank = 1;
  std::cout << "Rank PageRank Id\n";
  for (auto ii = top.rbegin(), ei = top.rend(); ii != ei; ++ii, ++rank) {
    std::cout << rank << ": " << ii->first.value << " " << ii->first.id << "\n";
  }
}

#if DEBUG
template <typename Graph>
void printPageRank(Graph& graph) {
  std::cout << "Id\tPageRank\n";
  int counter = 0;
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ii++) {
    auto& sd = graph.getData(*ii);
    std::cout << counter << " " << sd.value << "\n";
    counter++;
  }
}
#endif

#endif
