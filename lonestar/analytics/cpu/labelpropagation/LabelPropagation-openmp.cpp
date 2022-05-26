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

static cll::opt<std::string> inputSourceLabel("inputSourceLabel",
                                              cll::desc("Whitespace separated list "
                                                        "of sources in a file to "
                                                        "0 represents no label"),
                                              cll::init(""));

static cll::opt<float>
    tolerance("tolerance",
              cll::desc("converge tolerance value(default value 1.0e-5"),
              cll::init(1.0e-5));
static cll::opt<unsigned int>
    maxIterations("maxIterations",
                  cll::desc("Maximum iterations(default value 1000)"),
                  cll::init(1000));

constexpr static const unsigned CHUNK_SIZE = 64U;

typedef uint32_t LPTy;
typedef std::size_t Node;
typedef std::pair<Node, size_t> edge;
typedef std::vector<std::vector<edge>> edge_set;

std::size_t N;
edge_set edges;

std::vector<LPTy> label_vec;

// for storing label and converting label to id
std::vector<std::string> srcLabelVec;
std::unordered_map<std::string, uint32_t> key2id;
std::vector<std::string> id2key;
std::set<std::string> labels;

std::vector<std::string> split(std::string& s, std::string c=",") {
  std::vector<std::string> v;
  if (s == "") {
    return v;
  }

  std::string::size_type pos1;
  std::string::size_type pos2;

  pos1 = 0;
  pos2 = s.find(c);
  while (pos2 != std::string::npos) {
    v.emplace_back(s.substr(pos1, pos2 - pos1));
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if (pos1 != s.length()) {
    v.emplace_back(s.substr(pos1));
  }
  return v;
}

void parseInputLabel(unsigned nnodes) {
  // split by ,
  srcLabelVec = split(inputSourceLabel);

  // no label
  id2key.emplace_back(" ");
  key2id[" "] = 0;

  if (srcLabelVec.empty()) {
    for (Node n = 0; n < nnodes; n++) {
      std::string srcLabel = std::to_string(n + 1);
      id2key.push_back(srcLabel);
      key2id[srcLabel] = id2key.size() - 1;
      labels.insert(srcLabel);
    }
  } else {
    for (const auto& s : srcLabelVec) {
      if (s != " " && labels.count(s) == 0) {
        id2key.push_back(s);
        key2id[s] = id2key.size() - 1;
        labels.insert(s);
      }
    }
  }
}

bool noInitialLabel(Node node) {
  return srcLabelVec.empty() || srcLabelVec[node] == " ";
}

void InitializeNodeData() {
  label_vec.resize(N);

#pragma omp parallel for default(none) shared(N, edges, label_vec, srcLabelVec, key2id)
  for (Node src = 0; src < N; ++src) {
    if (srcLabelVec.empty()) {
      LPTy labelId = src + 1;
      label_vec[src] = labelId;
    }
    else {
      std::string label = srcLabelVec[src];
      LPTy labelId = key2id[label];
      label_vec[src] = labelId;
    }
  }
}

void LabelPropagationAsync() {
  unsigned int iterations = 0;

  while (true) {

    float diff = 0;

    // update label and compute difference
#pragma omp parallel for schedule(static, CHUNK_SIZE) reduction(+: diff) default(none) shared(CHUNK_SIZE, N, edges, label_vec, labels)
    for (Node src = 0; src < N; ++src) {
      std::vector<size_t> count(labels.size() + 1, 0);
      size_t maxCount = 0;
      size_t edgeNum = edges[src].size();
      std::vector<std::vector<LPTy>> countLabelVec(edgeNum + 1, std::vector<LPTy>());

      for (auto nbr : edges[src]) {
        Node dest = nbr.first;
        auto dstDataLabelId = label_vec[dest];

        if (dstDataLabelId == 0) {
          continue;
        }
        count[dstDataLabelId] ++;
        maxCount = std::max(maxCount, count[dstDataLabelId]);
        countLabelVec[count[dstDataLabelId]].push_back(dstDataLabelId);
      }

      if (maxCount > 0 && noInitialLabel(src)) {
        std::vector<LPTy> candidateLabel = countLabelVec[maxCount];
        auto srcLabelId = label_vec[src];

        if(std::find(candidateLabel.begin(), candidateLabel.end(), srcLabelId) == candidateLabel.end()) {
          LPTy maxCountLabel = candidateLabel[std::rand() % candidateLabel.size()];
          label_vec[src] = maxCountLabel;
          diff += 1;
        }
      }
    }

    std::cout << "iterations=" << iterations << " diff=" << diff << std::endl;

    if (iterations >= maxIterations || diff < tolerance) {
      break;
    }

    iterations++;
  } ///< End while(true).
  std::cout << "LabelPropagationAsync Execute Rounds: " << iterations << "\n";
}

void printLPValues(size_t begin, size_t end, std::ostream& out) {
  for (; begin != end; ++begin) {
    LPTy labelId = label_vec[begin];
    out << begin << " " << id2key[labelId] << "\n";
  }
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
    if (line[0] == '#') {
      continue;
    }

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
    if (line[0] == '#') {
      continue;
    }

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
  parseInputLabel(N);
  InitializeNodeData();

  std::cout << "Running LabelPropagation OpenMP algorithm\n";
  // execute algorithm
  galois::Timer execTime;
  execTime.start();

  LabelPropagationAsync();

  execTime.stop();

  std::cout << "Execute time: " << (execTime.get()) << "ms \n";

  // validate
  printLPValues(0, std::min(10UL, N), std::cout);
}