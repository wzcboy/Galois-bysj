#include "Lonestar/BoilerPlate.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <fstream>
#include <omp.h>

namespace cll = llvm::cl;


static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int>
    startNode("startNode",
              cll::desc("Node to start search from (default value 0)"),
              cll::init(0));

constexpr static const unsigned CHUNK_SIZE = 64U;

typedef std::size_t vertex_id;
typedef std::pair<vertex_id, size_t> edge;
typedef std::vector<std::vector<edge>> edge_set;


std::size_t N;
std::size_t src = 0;      // start of path
size_t INF = 10000000;      // infinity

edge_set edges;

std::vector<size_t> curr_distance_vec; // current shortest distances from src vertex
std::vector<size_t> last_distance_vec;
std::vector<omp_lock_t> dist_locks;

void initial(int numOfThreads) {
  curr_distance_vec.resize(N);
  last_distance_vec.resize(N);
  dist_locks.resize(N);

#pragma omp parallel for num_threads(numOfThreads) default(none) shared(N, INF, curr_distance_vec, last_distance_vec, dist_locks)
  for (vertex_id i = 0; i < N; ++i) {
    curr_distance_vec[i] = INF;
    last_distance_vec[i] = INF;
    omp_init_lock(&dist_locks[i]);
  }
  curr_distance_vec[src] = 0;
}

void topoOmpAlgo(int numOfThreads) {

  bool changed = false;
  size_t rounds = 0;

  omp_set_dynamic(0);
  omp_set_num_threads(numOfThreads);

  do {

    ++rounds;
    changed = false;

#pragma omp parallel for schedule(static, CHUNK_SIZE) reduction(|:changed) default(none) shared(N, edges, CHUNK_SIZE, curr_distance_vec, last_distance_vec, dist_locks)
    for (vertex_id u = 0; u < N; ++u) {

      omp_set_lock(&dist_locks[u]);
      auto uDist = curr_distance_vec[u];
      omp_unset_lock(&dist_locks[u]);

      if (last_distance_vec[u] > uDist) {
        last_distance_vec[u] = uDist;

        changed |= true;

        for (auto nbr : edges[u]) {
          vertex_id v = nbr.first;
          const auto newDist = uDist + nbr.second;

          omp_set_lock(&dist_locks[v]);
          if (curr_distance_vec[v] > newDist) {
            curr_distance_vec[v] = newDist;
          }
          omp_unset_lock(&dist_locks[v]);
        }
      }
    }

  } while (changed);

  std::cout << "rounds: " << rounds << std::endl;
}

void destroy(int numOfThreads) {
#pragma omp parallel for num_threads(numOfThreads) default(none) shared(N, dist_locks)
  for (vertex_id i = 0; i < N; ++i) {
    omp_destroy_lock(&dist_locks[i]);
  }
}
bool verify() {
  for (vertex_id u = 0; u < N; ++u) {
    auto& uDist = curr_distance_vec[u];
    if (uDist == INF)
        continue;
    for (auto nbr : edges[u]) {
      vertex_id v = nbr.first;
      auto& vDist= curr_distance_vec[v];
      if (vDist > uDist + nbr.second) {
        std::cout << "Wrong label: " << vDist << ", on node: " << v
                  << ", correct label from src node " << u << " is "
                  << uDist + nbr.second << "\n";
        return false;
      }
    }
  }
  return true;
}

void readGraphFile(const std::string& filePath) {
  std::ifstream file (filePath, std::ios::in);
  std::string line;

  vertex_id start;
  vertex_id end;
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
    edges[start].push_back({end, weight});
  }
  std::cout << "edges[src].size=" << edges[src].size() << "\n";
  file.close();
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  src = startNode;

  std::cout << "Reading from file: " << inputFile << "\n";
  readGraphFile(inputFile);

  initial(numThreads);

  std::cout << "Running topoOmp algorithm\n";

  galois::Timer execTime;
  execTime.start();
  topoOmpAlgo(numThreads);
  execTime.stop();

  destroy(numThreads);

  if (!skipVerify) {
    if (verify()) {
      std::cout << "Verification successful.\n";
    } else {
      GALOIS_DIE("verification failed");
    }
  }

  std::cout << "Execute time: " << (execTime.get()) << "ms \n";

  return 0;
}