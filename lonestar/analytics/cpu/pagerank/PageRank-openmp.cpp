#include "Lonestar/BoilerPlate.h"
#include "llvm/Support/CommandLine.h"
#include "galois/Galois.h"

#include <iostream>
#include <fstream>
#include <stack>
#include <omp.h>

namespace cll = llvm::cl;

static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);

constexpr static const unsigned CHUNK_SIZE = 64U;

constexpr static const float ALPHA         = 0.85;
constexpr static const float INIT_RESIDUAL = 1 - ALPHA;

constexpr static const float TOLERANCE   = 1.0e-5;
constexpr static const unsigned MAX_ITER = 1000;

typedef std::size_t Node;
typedef std::pair<Node, size_t> edge;
typedef std::vector<std::vector<edge>> edge_set;

std::size_t N;
edge_set edges;

std::vector<float> pagerank_vec;
std::vector<size_t> outdegree_vec;
std::vector<omp_lock_t> degree_locks;

//! Initialize nodes for the topological algorithm.
void initNodeDataTopological() {
  float init_value = 1.0F / N;

  pagerank_vec.resize(N);
  outdegree_vec.resize(N);
  degree_locks.resize(N);

#pragma omp parallel for default(none) shared(N, init_value, pagerank_vec, outdegree_vec, degree_locks)
  for (Node i = 0; i < N; ++i) {
    pagerank_vec[i]  = init_value;
    outdegree_vec[i] = 0;
    omp_init_lock(&degree_locks[i]);
  }

  //! compute out degree
#pragma omp parallel for default(none) shared(N, edges, outdegree_vec, degree_locks)
  for (Node i = 0; i < N; ++i) {
    for (auto nbr : edges[i]) {
      Node dst = nbr.first;
      omp_set_lock(&degree_locks[dst]);
      outdegree_vec[dst] += 1;
      omp_unset_lock(&degree_locks[dst]);
    }
  }
}

/**
 * PageRank pull topological.
 * Always calculate the new pagerank for each iteration.
 */
void runPageank() {
  unsigned int iteration = 0;

  float base_score = (1.0f - ALPHA) / N;
  while (true) {

    float accum = 0;
#pragma omp parallel for schedule(static, CHUNK_SIZE) reduction(+: accum) default(none) shared(CHUNK_SIZE, N, edges, pagerank_vec, outdegree_vec, base_score)
    for (Node src = 0; src < N; ++src) {
        float sum    = 0.0;

        for (auto nbr : edges[src]) {
          Node dst = nbr.first;
          sum += pagerank_vec[dst] / static_cast<float>(outdegree_vec[dst]);
        }

        //! New value of pagerank after computing contributions from
        //! incoming edges in the original graph.
        float value = sum * ALPHA + base_score;
        //! Find the delta in new and old pagerank values.
        float diff = std::fabs(value - pagerank_vec[src]);

        //! Do not update pagerank before the diff is computed since
        //! there is a data dependence on the pagerank value.
        pagerank_vec[src] = value;
        accum += diff;
    }

    iteration += 1;
    if (accum <= TOLERANCE || iteration >= MAX_ITER) {
      break;
    }

  } ///< End while(true).


  std::cout << "Total Rounds: " << iteration << "\n";

  if (iteration >= MAX_ITER) {
    std::cerr << "ERROR: failed to converge in " << iteration
              << " iterations\n";
  }
}

void destroy() {
#pragma omp parallel for default(none) shared(N, degree_locks)
  for (Node i = 0; i < N; ++i) {
    omp_destroy_lock(&degree_locks[i]);
  }
}

void sanity() {
  double accumMax = 0;
  double accumMin = 0;
  double accumSum = 0;

  // get max, min, sum of BC values using accumulators and reducers
#pragma omp parallel for default(none) shared(N, pagerank_vec) reduction(max:accumMax)
  for (Node i = 0; i < N; ++i) {
    double rank = pagerank_vec[i];
    accumMax = std::max(accumMax, rank);
  }
#pragma omp parallel for default(none) shared(N, pagerank_vec) reduction(min:accumMin)
  for (Node i = 0; i < N; ++i) {
    double rank = pagerank_vec[i];
    accumMin = std::min(accumMin, rank);
  }
#pragma omp parallel for default(none) shared(N, pagerank_vec) reduction(+:accumSum)
  for (Node i = 0; i < N; ++i) {
    double rank = pagerank_vec[i];
    accumSum += rank;
  }
  galois::gPrint("Max Rank is ", accumMax, "\n");
  galois::gPrint("Min Rank is ", accumMin, "\n");
  galois::gPrint("Rank sum is ", accumSum, "\n");
}
void readGraphFile(const std::string& filePath) {
  std::ifstream file (filePath, std::ios::in);
  std::string line;

  Node start;
  Node end;
  size_t weight;

  size_t numNodes   = 0;
  size_t numEdges   = 0;

  while (getline(file, line)) {
    if (line[0] == '#') continue;

    std::stringstream ss(line);

    ss >> start;
    ss.ignore(1);
    ss >> end;
    ss.ignore(1);
    ss >> weight;

    numEdges++;
    numNodes =  std::max(numNodes, start);
  }

  numNodes++;
  N = numNodes;
  std::cout << "numNodes= " << numNodes << " numEdges= " << numEdges << std::endl;

  file.clear();
  file.seekg(0, std::ios::beg);

  edges.resize(numNodes);
  while (getline(file, line)) {
    if (line[0] == '#') continue;

    std::stringstream ss(line);

    ss >> start;
    ss.ignore(1);
    ss >> end;
    ss.ignore(1);
    ss >> weight;

    if (weight <= 0) {
      weight = 1;
    }
    // Note that we reverse the edge start and end
    edges[end].push_back({start, weight});
  }
  file.close();
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  std::cout << "Reading from file: " << inputFile << "\n";
  readGraphFile(inputFile);

  omp_set_dynamic(0);
  omp_set_num_threads(numThreads);

  // initial
  initNodeDataTopological();

  std::cout << "Running PageRank OpenMP algorithm\n";
  // execute algorithm
  galois::Timer execTime;
  execTime.start();

  runPageank();

  execTime.stop();

  std::cout << "Execute time: " << (execTime.get()) << "ms \n";

  // validate
  sanity();
  // destroy locks
  destroy();

}