#include "Lonestar/BoilerPlate.h"
#include "llvm/Support/CommandLine.h"
#include "galois/Galois.h"

#include <iostream>
#include <fstream>
#include <stack>
#include <iomanip>
#include <omp.h>

namespace cll = llvm::cl;

static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);

constexpr static const unsigned CHUNK_SIZE = 64U;

typedef std::size_t Node;
typedef std::pair<Node, size_t> edge;
typedef std::vector<std::vector<edge>> edge_set;

std::size_t N;
edge_set edges;

std::vector<size_t> dist_vec;
std::vector<double> sigma_vec;
std::vector<double> delta_vec;
std::vector<double> bc_vec;
std::vector<omp_lock_t> bc_locks;

/**
 * Initialize node fields all to 0
 */
void InitializeNodeData() {
  dist_vec.resize(N);
  sigma_vec.resize(N);
  delta_vec.resize(N);
  bc_vec.resize(N);
  bc_locks.resize(N);

#pragma omp parallel for default(none) shared(N, dist_vec, sigma_vec, delta_vec, bc_vec, bc_locks)
  for (Node i = 0; i < N; i++) {
    dist_vec[i] = 0;
    sigma_vec[i] = 0;
    delta_vec[i] = 0;
    bc_vec[i] = 0;
    omp_init_lock(&bc_locks[i]);
  }
}


void run() {
#pragma omp parallel for schedule(static, CHUNK_SIZE) default(none) shared(CHUNK_SIZE, N, edges, bc_vec, bc_locks, std::cout) firstprivate(dist_vec, sigma_vec, delta_vec)
  for (Node curSource = 0; curSource < N; ++curSource) {
    if (curSource % 1000 == 0) {
      std::cout << "curSource= " << curSource << " thread_id=" << omp_get_thread_num() << "\n";
    }
    std::queue<Node> Q;
    std::stack<Node> S;
    std::vector<Node>* pre = new std::vector<Node>[N]();

    dist_vec[curSource]  = 1;
    sigma_vec[curSource] = 1;

    Q.push(curSource);

    // Do bfs while computing number of shortest paths (saved into sigma)
    // and successors of nodes;
    // Note this bfs makes it so source has distance of 1 instead of 0
    while (!Q.empty()) {
      Node src = Q.front();
      Q.pop();
      S.push(src);

      for (auto nbr : edges[src]) {
        Node dest = nbr.first;

        if (!dist_vec[dest]) {
          Q.push(dest);
          dist_vec[dest] = dist_vec[src] + 1;
        }

        if (dist_vec[dest] == dist_vec[src] + 1) {
          sigma_vec[dest] += sigma_vec[src];
          pre[dest].push_back(src);
        }
      }
    }

    // Back-propogate the dependency values (delta) along the BFS DAG
    while (!S.empty()) {
      Node w = S.top();
      S.pop();


      auto pre_list = pre[w];

      for (Node v : pre_list) {
        delta_vec[v] += (sigma_vec[v] / sigma_vec[w]) *
                               (1.0 + delta_vec[w]);
      }
      // ignore the source
      if (w != curSource) {
        // avoid race writing
        omp_set_lock(&bc_locks[w]);
        bc_vec[w] += delta_vec[w];
        omp_unset_lock(&bc_locks[w]);
      }
    }

    // save result of this source's BC, reset all local values for next source
    for (Node i = 0; i < N; ++i) {
      delta_vec[i] = 0;
      sigma_vec[i] = 0;
      dist_vec[i]  = 0;
    }
  }
}

void destroy() {
#pragma omp parallel for default(none) shared(N, bc_locks)
  for (Node i = 0; i < N; ++i) {
    omp_destroy_lock(&bc_locks[i]);
  }
}

/**
 * Print betweeness-centrality measures.
 *
 * @param begin first node to print BC measure of
 * @param end iterator after last node to print
 * @param out stream to output to
 * @param precision precision of the floating points outputted by the function
 */
void printBCValues(size_t begin, size_t end, std::ostream& out,
                   int precision = 6) {
  for (; begin != end; ++begin) {
    double bc = bc_vec[begin];

    out << begin << " " << std::setiosflags(std::ios::fixed)
        << std::setprecision(precision) << bc << "\n";
  }
}

/**
 * Print all betweeness centrality values in the graph.
 */
void printBCcertificate() {
  std::stringstream foutname;
  foutname << "outer_certificate_" << galois::getActiveThreads();

  std::ofstream outf(foutname.str().c_str());
  galois::gInfo("Writing certificate...");

  printBCValues(0, N, outf, 9);

  outf.close();
}

//! sanity check of BC values
void sanity() {
  double accumMax = 0;
  double accumMin = 0;
  double accumSum = 0;

  // get max, min, sum of BC values using accumulators and reducers
#pragma omp parallel for default(none) shared(N, bc_vec) reduction(max:accumMax)
  for (Node i = 0; i < N; ++i) {
    double bc = bc_vec[i];
    accumMax = std::max(accumMax, bc);
  }
#pragma omp parallel for default(none) shared(N, bc_vec) reduction(min:accumMin)
  for (Node i = 0; i < N; ++i) {
    double bc = bc_vec[i];
    accumMin = std::min(accumMin, bc);
  }
#pragma omp parallel for default(none) shared(N, bc_vec) reduction(+:accumSum)
  for (Node i = 0; i < N; ++i) {
    double bc = bc_vec[i];
    accumSum += bc;
  }
  galois::gPrint("Max BC is ", accumMax, "\n");
  galois::gPrint("Min BC is ", accumMin, "\n");
  galois::gPrint("BC sum is ", accumSum, "\n");
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
    edges[start].push_back({end, weight});
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
  InitializeNodeData();

  std::cout << "Running BetweennessCentrality OpenMP algorithm\n";
  // execute algorithm
  galois::Timer execTime;
  execTime.start();

  run();

  execTime.stop();

  std::cout << "Execute time: " << (execTime.get()) << "ms \n";

  printBCValues(0, std::min(10UL, N), std::cout, 6);

  // validate the result
  sanity();

  destroy();
  // printBCcertificate();

}

