#include "galois/Galois.h"
#include "galois/AtomicHelpers.h"
#include "galois/Reduction.h"
#include "galois/PriorityQueue.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "Lonestar/BoilerPlate.h"
#include "Lonestar/BFS_SSSP.h"
#include "Lonestar/Utils.h"

#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <boost/filesystem.hpp>

namespace cll = llvm::cl;

static const char* name = "Label Propagation";
static const char* desc =
    "Every node choose the label occurring with highest frequency among neighbors";
static const char* url = "label_propagation";

enum Algo { Sync = 0, Async, Priority, Serial, Schedule };

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

static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
                           cll::values(clEnumVal(Sync, "Sync"),
                                       clEnumVal(Async, "Async"),
                                       clEnumVal(Priority, "Priority"),
                                       clEnumVal(Serial, "Serial"),
                                       clEnumVal(Schedule, "Schedule")),
                           cll::init(Async));

constexpr static const unsigned CHUNK_SIZE = 64;
const uint32_t infinity = std::numeric_limits<uint32_t>::max() / 4;

//! [withnumaalloc]
using NodeType = uint32_t;
using Graph = galois::graphs::LC_CSR_Graph<NodeType, void>
    ::with_no_lockable<true>::type ::with_numa_alloc<true>::type;

typedef Graph::GraphNode GNode;

using LabelArray = galois::LargeArray<NodeType>;


// for storing label and converting label to id
std::vector<std::string> srcLabelVec;
std::unordered_map<std::string, uint32_t> key2id;
std::vector<std::string> id2key;
std::set<std::string> labels;

std::vector<std::string> split(std::string& s, std::string c=",") {
  std::vector<std::string> v;
  if (s == "") return v;

  std::string::size_type pos1, pos2;
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
  id2key.push_back(" ");
  key2id[" "] = 0;

  if (srcLabelVec.size() == 0) {
    for (GNode n = 0; n < nnodes; n++) {
      std::string srcLabel = std::to_string(n + 1);
      id2key.push_back(srcLabel);
      key2id[srcLabel] = id2key.size() - 1;
      labels.insert(srcLabel);
    }
  } else {
    for (auto s : srcLabelVec) {
      if (s != " " && labels.count(s) == 0) {
        id2key.push_back(s);
        key2id[s] = id2key.size() - 1;
        labels.insert(s);
      }
    }
  }
}

bool noInitialLabel(GNode node) {
  return srcLabelVec.size() == 0 || srcLabelVec[node] == " ";
}

void InitializeNodeData(Graph& graph) {
  galois::do_all(
      galois::iterate(graph),
      [&](const GNode& src) {
        if (srcLabelVec.size() == 0) {
          NodeType labelId = src + 1;
          graph.getData(src) = labelId;
        }
        else {
          std::string label = srcLabelVec[src];
          NodeType labelId = key2id[label];
          graph.getData(src) = labelId;
        }
      },
      galois::loopname("InitializeNodeData"), galois::no_stats(),
      galois::steal(), galois::chunk_size<CHUNK_SIZE>());
}

void LabelPropagationSync(Graph& graph) {
  LabelArray oldNodeLabels;
  oldNodeLabels.allocateInterleaved(graph.size());

  unsigned int iterations = 0;
  galois::GAccumulator<float> diff;
  float oldDiff = 0;

  while (true) {
    if (iterations % 2 == 0) {
      oldDiff = diff.reduce();
    }

    // assign all node labels to oldNodeLabels
    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          oldNodeLabels[src] = graph.getData(src);
        }, galois::steal(), galois::loopname("AssignToOldNodeLabels"));

    diff.reset();

    // update label and compute difference
    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          std::vector<size_t> count(labels.size() + 1, 0);
          size_t maxCount = 0ul;
          size_t edgeNum = std::distance(graph.edge_begin(src), graph.edge_end(src));
          std::vector<std::vector<NodeType>> countLabelVec(edgeNum + 1, std::vector<NodeType>());

          for (auto e : graph.edges(src)) {
            GNode dest = graph.getEdgeDst(e);
            auto& dstDataLabelId = oldNodeLabels[dest];

            if (dstDataLabelId == 0) {
              continue;
            }
            count[dstDataLabelId] ++;
            maxCount = std::max(maxCount, count[dstDataLabelId]);
            countLabelVec[count[dstDataLabelId]].push_back(dstDataLabelId);
          }

          if (maxCount > 0 && noInitialLabel(src)) {
            std::vector<NodeType> candidateLabel = countLabelVec[maxCount];
            auto& srcLabelId = graph.getData(src);

            if(std::find(candidateLabel.begin(), candidateLabel.end(), srcLabelId) == candidateLabel.end()) {
              NodeType maxCountLabel = candidateLabel[std::rand() % candidateLabel.size()];
              srcLabelId = maxCountLabel;
              diff += 1;
            }
          }
        },
        galois::steal(), galois::loopname("UpdateLabel"));

    // to eliminate vibration
    float diffValue = diff.reduce();
    if (diffValue == oldDiff) {
      galois::GAccumulator<float> deltaDiff;
      deltaDiff.reset();

      galois::do_all(
          galois::iterate(graph),
          [&](const GNode& src) {
            auto& srcLabelId = graph.getData(src);
            if (srcLabelId > oldNodeLabels[src]) {
              srcLabelId = oldNodeLabels[src];
              deltaDiff += 1;
            }
          }, galois::steal(), galois::loopname("EliminateVibration"));
      diff += (-deltaDiff.reduce());
    }

    std::cout << "iterations=" << iterations << " diff=" << diff.reduce() << std::endl;
    iterations++;
    if (iterations >= maxIterations || diff.reduce() < tolerance) {
      break;
    }
  } ///< End while(true).
  galois::runtime::reportStat_Single("LabelPropagationSync", "Rounds", iterations);
}

void LabelPropagationAsync(Graph& graph) {
  unsigned int iterations = 0;
  galois::GAccumulator<float> diff;

  while (true) {
    // update label and compute difference
    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          std::vector<size_t> count(labels.size() + 1, 0);
          size_t maxCount = 0ul;
          size_t edgeNum = std::distance(graph.edge_begin(src), graph.edge_end(src));
          std::vector<std::vector<NodeType>> countLabelVec(edgeNum + 1, std::vector<NodeType>());

          for (auto e : graph.edges(src)) {
            GNode dest = graph.getEdgeDst(e);
            auto& dstDataLabelId = graph.getData(dest);

            if (dstDataLabelId == 0) {
              continue;
            }
            count[dstDataLabelId] ++;
            maxCount = std::max(maxCount, count[dstDataLabelId]);
            countLabelVec[count[dstDataLabelId]].push_back(dstDataLabelId);
          }

          if (maxCount > 0 && noInitialLabel(src)) {
            std::vector<NodeType> candidateLabel = countLabelVec[maxCount];
            auto& srcLabelId = graph.getData(src);

            if(std::find(candidateLabel.begin(), candidateLabel.end(), srcLabelId) == candidateLabel.end()) {
              NodeType maxCountLabel = candidateLabel[std::rand() % candidateLabel.size()];
              srcLabelId = maxCountLabel;
              diff += 1;
            }
          }
        },
        galois::steal(), galois::loopname("UpdateLabel"));


    std::cout << "iterations=" << iterations << " diff=" << diff.reduce() << std::endl;

    if (iterations >= maxIterations || diff.reduce() < tolerance) {
      break;
    }

    iterations++;
    diff.reset();
  } ///< End while(true).
  galois::runtime::reportStat_Single("LabelPropagationAsync", "Rounds", iterations);
}

void LabelPropagationSchedule(Graph& graph) {
  unsigned int iterations = 0;
  galois::GAccumulator<float> diff;
  typedef galois::worklists::ChunkFIFO<CHUNK_SIZE> WL;

  while (true) {
    // update label and compute difference
    galois::for_each(
        galois::iterate(graph),
        [&](const GNode& src, auto& ctx) {
          std::vector<size_t> count(labels.size() + 1, 0);
          size_t maxCount = 0ul;
          size_t edgeNum = std::distance(graph.edge_begin(src), graph.edge_end(src));
          std::vector<std::vector<NodeType>> countLabelVec(edgeNum + 1, std::vector<NodeType>());

          for (auto e : graph.edges(src)) {
            GNode dest = graph.getEdgeDst(e);
            auto& dstDataLabelId = graph.getData(dest);

            if (dstDataLabelId == 0) {
              continue;
            }
            count[dstDataLabelId] ++;
            maxCount = std::max(maxCount, count[dstDataLabelId]);
            countLabelVec[count[dstDataLabelId]].push_back(dstDataLabelId);
          }

          if (maxCount > 0 && noInitialLabel(src)) {
            std::vector<NodeType> candidateLabel = countLabelVec[maxCount];
            auto& srcLabelId = graph.getData(src);

            if(std::find(candidateLabel.begin(), candidateLabel.end(), srcLabelId) == candidateLabel.end()) {
              NodeType maxCountLabel = candidateLabel[std::rand() % candidateLabel.size()];
              srcLabelId = maxCountLabel;
              diff += 1;
            }
          }
        },
        galois::no_stats(), galois::disable_conflict_detection(), galois::no_pushes(),
        galois::wl<WL>(), galois::loopname("UpdateLabel"));


    std::cout << "iterations=" << iterations << " diff=" << diff.reduce() << std::endl;

    if (iterations >= maxIterations || diff.reduce() < tolerance) {
      break;
    }

    iterations++;
    diff.reset();
  } ///< End while(true).
  galois::runtime::reportStat_Single("LabelPropagationSchedule", "Rounds", iterations);
}

void LabelPropagationSerial(Graph& graph) {
  unsigned int iterations = 0;
  galois::GAccumulator<float> diff;

  galois::StdForEach loop;
  while (true) {
    // update label and compute difference
    loop(
        galois::iterate(graph),
        [&](const GNode& src) {
          std::vector<size_t> count(labels.size() + 1, 0);
          size_t maxCount = 0ul;
          size_t edgeNum = std::distance(graph.edge_begin(src), graph.edge_end(src));
          std::vector<std::vector<NodeType>> countLabelVec(edgeNum + 1, std::vector<NodeType>());

          for (auto e : graph.edges(src)) {
            GNode dest = graph.getEdgeDst(e);
            auto& dstDataLabelId = graph.getData(dest);

            if (dstDataLabelId == 0) {
              continue;
            }
            count[dstDataLabelId] ++;
            maxCount = std::max(maxCount, count[dstDataLabelId]);
            countLabelVec[count[dstDataLabelId]].push_back(dstDataLabelId);
          }

          if (maxCount > 0 && noInitialLabel(src)) {
            std::vector<NodeType> candidateLabel = countLabelVec[maxCount];
            auto& srcLabelId = graph.getData(src);

            if(std::find(candidateLabel.begin(), candidateLabel.end(), srcLabelId) == candidateLabel.end()) {
              NodeType maxCountLabel = candidateLabel[std::rand() % candidateLabel.size()];
              srcLabelId = maxCountLabel;
              diff += 1;
            }
          }
        },
        galois::loopname("UpdateLabel"));


    std::cout << "iterations=" << iterations << " diff=" << diff.reduce() << std::endl;

    if (iterations >= maxIterations || diff.reduce() < tolerance) {
      break;
    }

    iterations++;
    diff.reset();
  } ///< End while(true).
  galois::runtime::reportStat_Single("LabelPropagationSerial", "Rounds", iterations);
}

struct LPWorkItem {
  uint32_t nodeID;
  uint32_t shift;
  LPWorkItem() : nodeID(0), shift(0){};
  LPWorkItem(uint32_t _node, uint32_t _diff) : nodeID(_node), shift(_diff) {};
};

struct LPWorkItemIndexer {
  uint32_t operator()(const LPWorkItem& item) {
    return item.shift;
  }
};

uint32_t changeShiftToValidRange(uint32_t nnodes, uint32_t shift) {
  // keep the sum of bags to 0~10
  return static_cast<uint32_t>(10 * shift / nnodes);
}

namespace gwl = galois::worklists;
using PSchunk = gwl::PerSocketChunkFIFO<CHUNK_SIZE>;
using OBIM    = gwl::OrderedByIntegerMetric<LPWorkItemIndexer, PSchunk>;

void LabelPropagationPriority(Graph& graph) {
  LabelArray labelShifts;
  labelShifts.allocateInterleaved(graph.size());

  galois::do_all(
      galois::iterate(graph),
      [&](const GNode& src) {
        labelShifts[src] = 0;
      },
      galois::steal(), galois::no_stats());

  unsigned int iterations = 0;
  galois::GAccumulator<float> diff;

  galois::InsertBag<LPWorkItem> activeNodes;

  while (true) {

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          auto validShift = changeShiftToValidRange(graph.size(), labelShifts[src]);
          activeNodes.push(LPWorkItem(src, validShift));
        },
        galois::steal(), galois::no_stats());

    // update label and compute difference
    galois::for_each(
        galois::iterate(activeNodes),
        [&](const LPWorkItem& item, auto& ctx) {
          GNode src = item.nodeID;

          std::vector<size_t> count(labels.size() + 1, 0);
          size_t maxCount = 0ul;
          size_t edgeNum = std::distance(graph.edge_begin(src), graph.edge_end(src));
          std::vector<std::vector<NodeType>> countLabelVec(edgeNum + 1, std::vector<NodeType>());

          for (auto e : graph.edges(src)) {
            GNode dest = graph.getEdgeDst(e);
            auto& dstDataLabelId = graph.getData(dest);

            if (dstDataLabelId == 0) {
              continue;
            }
            count[dstDataLabelId] ++;
            maxCount = std::max(maxCount, count[dstDataLabelId]);
            countLabelVec[count[dstDataLabelId]].push_back(dstDataLabelId);
          }

          if (maxCount > 0 && noInitialLabel(src)) {
            std::vector<NodeType> candidateLabel = countLabelVec[maxCount];
            auto& srcLabelId = graph.getData(src);

            if(std::find(candidateLabel.begin(), candidateLabel.end(), srcLabelId) == candidateLabel.end()) {
              NodeType maxCountLabel = candidateLabel[std::rand() % candidateLabel.size()];
              srcLabelId = maxCountLabel;
              diff += 1;
            }
          }
        },
        galois::disable_conflict_detection(), galois::no_pushes(),
        galois::wl<OBIM>(LPWorkItemIndexer()), galois::loopname("UpdateLabel"));


    std::cout << "iterations=" << iterations << " diff=" << diff.reduce() << std::endl;

    if (iterations >= maxIterations || diff.reduce() < tolerance) {
      break;
    }

    iterations++;
    diff.reset();
    activeNodes.clear();
  } ///< End while(true).
  galois::runtime::reportStat_Single("LabelPropagationPriority", "Rounds", iterations);
}

//! sanity check of labelpropagation
void LpSanity(Graph& graph) {
  std::vector<std::atomic<size_t>> count(labels.size() + 1);
  galois::GReduceMax<size_t> accumMax;
  galois::GReduceMin<size_t> accumMin;
  accumMax.reset();
  accumMin.reset();

  galois::do_all(
      galois::iterate(graph),
      [&](GNode n) {
        auto& srcLabelId = graph.getData(n);
        count[srcLabelId] ++;

        accumMax.update(count[srcLabelId]);
        accumMin.update(count[srcLabelId]);
      },
      galois::no_stats(), galois::loopname("LpSanity"));

  size_t accumMaxVal = accumMax.reduce();
  size_t accumMinVal = accumMin.reduce();

  std::vector<std::string> maxCountLabelVec;
  std::vector<std::string> minCountLabelVec;

  for (size_t i = 0; i < count.size(); i++) {
    if (count[i] == accumMaxVal) {
      maxCountLabelVec.push_back(id2key[i]);
    }
    if (count[i] == accumMinVal) {
      minCountLabelVec.push_back(id2key[i]);
    }
  }

  std::cout << "Max frequency is: " << accumMaxVal << ". All label: ";
  for (auto label : maxCountLabelVec) {
    std::cout << label << " ";
  }
  std::cout << "\n";

  std::cout << "Min frequency is: " << accumMinVal << std::endl;
}

void writeOutput(const std::string& outputDir, Graph& graph, std::string outputFileName = "output") {
  namespace fs = boost::filesystem;
  fs::path filename{outputDir};
  filename = filename.append(outputFileName);

  std::ofstream outputFile(filename.string().c_str());

  if (!outputFile) {
    std::cerr << "could not open file: " << filename << std::endl;
  }

  outputFile << "nodeID\tlabel\n";
  for (size_t i = 0; i < graph.size(); i++) {
    outputFile << i << " " << id2key[graph.getData(i)] << std::endl;
  }

  if (!outputFile) {
    std::cerr << "failed to write file: " << filename << std::endl;
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url, &inputFile);

  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  Graph graph;
  std::cout << "Reading graph: " << inputFile << "\n";

  galois::graphs::readGraph(graph, inputFile);
  std::cout << "Read " << graph.size() << " nodes" << graph.sizeEdges() << " edges\n";

  size_t approxNodeData = graph.size() * sizeof(typename Graph::node_data_type);
  galois::preAlloc(numThreads + approxNodeData / galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");


  // initial work
  parseInputLabel(graph.size());
  InitializeNodeData(graph);


  galois::StatTimer execTime("Timer_0");
  execTime.start();

  switch (algo) {
  case Sync:
    std::cout << "Running sync version, tolerance:" << tolerance
              << ", maxIterations:" << maxIterations << "\n";
    LabelPropagationSync(graph);
    break;
  case Async:
    std::cout << "Running async version, tolerance:" << tolerance
              << ", maxIterations:" << maxIterations << "\n";
    LabelPropagationAsync(graph);
    break;
  case Priority:
    std::cout << "Running Priority version, tolerance:" << tolerance
              << ", maxIterations:" << maxIterations << "\n";
    LabelPropagationPriority(graph);
    break;
  case Serial:
    std::cout << "Running Serial version, tolerance:" << tolerance
              << ", maxIterations:" << maxIterations << "\n";
    LabelPropagationSerial(graph);
    break;
  case Schedule:
    std::cout << "Running Schedule version, tolerance:" << tolerance
              << ", maxIterations:" << maxIterations << "\n";
    LabelPropagationSchedule(graph);
    break;
  default:
    std::abort();
  }

  execTime.stop();

  galois::reportPageAlloc("MeminfoPost");

  LpSanity(graph);

  if (outputToFile) {
    writeOutput(outputLocation, graph);
  }

  totalTime.stop();

  return 0;
}


