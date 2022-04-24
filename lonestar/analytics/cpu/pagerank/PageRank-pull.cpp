/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "Lonestar/BoilerPlate.h"
#include "PageRank-constants.h"
#include "galois/Galois.h"
#include "galois/LargeArray.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/gstl.h"

#include <boost/filesystem.hpp>

const char* desc =
    "Computes page ranks a la Page and Brin. This is a pull-style algorithm.";

enum Algo { Topo = 0, Residual, TopoSchedule, SerTopo, TopoPriority };

static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
                           cll::values(clEnumVal(Topo, "Topological"),
                                       clEnumVal(Residual, "Residual"),
                                       clEnumVal(TopoSchedule, "TopoSchedule"),
                                       clEnumVal(SerTopo, "SerTopo"),
                                       clEnumVal(TopoPriority, "TopoPriority")),
                           cll::init(Residual));

//! Flag that forces user to be aware that they should be passing in a
//! transposed graph.
static cll::opt<bool>
    transposedGraph("transposedGraph",
                    cll::desc("Specify that the input graph is transposed"),
                    cll::init(false));

constexpr static const unsigned CHUNK_SIZE = 32;

struct LNode {
  PRTy value;
  uint32_t nout;
  float diff;
};

typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_no_lockable<
    true>::type ::with_numa_alloc<true>::type Graph;
typedef typename Graph::GraphNode GNode;

using DeltaArray    = galois::LargeArray<PRTy>;
using ResidualArray = galois::LargeArray<PRTy>;

//! Initialize nodes for the topological algorithm.
void initNodeDataTopological(Graph& g) {
  PRTy init_value = 1.0f / g.size();
  galois::do_all(
      galois::iterate(g),
      [&](const GNode& n) {
        auto& sdata = g.getData(n, galois::MethodFlag::UNPROTECTED);
        sdata.value = init_value;
        sdata.nout  = 0;
        sdata.diff = 0;
      },
      galois::no_stats(), galois::loopname("initNodeData"));
}

//! Initialize nodes for the residual algorithm.
void initNodeDataResidual(Graph& g, DeltaArray& delta,
                          ResidualArray& residual) {
  galois::do_all(
      galois::iterate(g),
      [&](const GNode& n) {
        auto& sdata = g.getData(n, galois::MethodFlag::UNPROTECTED);
        sdata.value = 0;
        sdata.nout  = 0;
        sdata.diff  = 0;
        delta[n]    = 0;
        residual[n] = INIT_RESIDUAL;
      },
      galois::no_stats(), galois::loopname("initNodeData"));
}

//! Computing outdegrees in the tranpose graph is equivalent to computing the
//! indegrees in the original graph.
void computeOutDeg(Graph& graph) {
  galois::StatTimer outDegreeTimer("computeOutDegFunc");
  outDegreeTimer.start();

  galois::LargeArray<std::atomic<size_t>> vec;
  vec.allocateInterleaved(graph.size());

  galois::do_all(
      galois::iterate(graph),
      [&](const GNode& src) { vec.constructAt(src, 0ul); }, galois::no_stats(),
      galois::loopname("InitDegVec"));

  galois::do_all(
      galois::iterate(graph),
      [&](const GNode& src) {
        for (auto nbr : graph.edges(src)) {
          GNode dst = graph.getEdgeDst(nbr);
          vec[dst].fetch_add(1ul);
        };
      },
      galois::steal(), galois::chunk_size<CHUNK_SIZE>(), galois::no_stats(),
      galois::loopname("computeOutDeg"));

  galois::do_all(
      galois::iterate(graph),
      [&](const GNode& src) {
        auto& srcData = graph.getData(src, galois::MethodFlag::UNPROTECTED);
        srcData.nout  = vec[src];
      },
      galois::no_stats(), galois::loopname("CopyDeg"));

  outDegreeTimer.stop();
}

/**
 * It does not calculate the pagerank for each iteration,
 * but only calculate the residual to be added from the previous pagerank to
 * the current one.
 * If the residual is smaller than the tolerance, that is not reflected to
 * the next pagerank.
 */
//! [scalarreduction]
void computePRResidual(Graph& graph, DeltaArray& delta,
                       ResidualArray& residual) {
  unsigned int iterations = 0;
  galois::GAccumulator<unsigned int> accum;

  while (true) {
    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          auto& sdata = graph.getData(src);
          delta[src]  = 0;

          //! Only the residual higher than tolerance will be reflected
          //! to the pagerank.
          if (residual[src] > tolerance) {
            PRTy oldResidual = residual[src];
            residual[src]    = 0.0;
            sdata.value += oldResidual;
            if (sdata.nout > 0) {
              delta[src] = oldResidual * ALPHA / sdata.nout;
              accum += 1;
            }
          }
        },
        galois::no_stats(), galois::loopname("PageRank_delta"));

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          float sum = 0;
          for (auto nbr : graph.edges(src)) {
            GNode dst = graph.getEdgeDst(nbr);
            if (delta[dst] > 0) {
              sum += delta[dst];
            }
          }
          if (sum > 0) {
            residual[src] = sum;
          }
        },
        galois::steal(), galois::chunk_size<CHUNK_SIZE>(), galois::no_stats(),
        galois::loopname("PageRank"));

#if DEBUG
    std::cout << "iteration: " << iterations << "\n";
#endif
    iterations++;
    if (iterations >= maxIterations || !accum.reduce()) {
      break;
    }
    accum.reset();
  } ///< End while(true).
    //! [scalarreduction]

  galois::runtime::reportStat_Single("PageRank", "Rounds", iterations);

  if (iterations >= maxIterations) {
    std::cerr << "ERROR: failed to converge in " << iterations
              << " iterations\n";
  }

}

void computePRTopoSchedule(Graph& graph) {
  unsigned int iteration = 0;
  galois::GAccumulator<float> accum;

  typedef galois::worklists::PerThreadChunkFIFO<CHUNK_SIZE> WL;

  float base_score = (1.0f - ALPHA) / graph.size();
  while (true) {
    galois::for_each(
        galois::iterate(graph),
        [&](const GNode& src, auto& ctx) {
          constexpr const galois::MethodFlag flag =
              galois::MethodFlag::UNPROTECTED;

          LNode& sdata = graph.getData(src, flag);
          float sum    = 0.0;

          for (auto jj = graph.edge_begin(src, flag),
                    ej = graph.edge_end(src, flag);
               jj != ej; ++jj) {
            GNode dst = graph.getEdgeDst(jj);

            LNode& ddata = graph.getData(dst, flag);
            sum += ddata.value / ddata.nout;
          }

          //! New value of pagerank after computing contributions from
          //! incoming edges in the original graph.
          float value = sum * ALPHA + base_score;
          //! Find the delta in new and old pagerank values.
          float diff = std::fabs(value - sdata.value);

          //! Do not update pagerank before the diff is computed since
          //! there is a data dependence on the pagerank value.
          sdata.value = value;
          accum += diff;
        },
        galois::no_stats(), galois::disable_conflict_detection(), galois::no_pushes(),
        galois::loopname("PageRank"), galois::wl<WL>());

#if DEBUG
    std::cout << "iteration: " << iteration << " max delta: " << delta << "\n";
#endif

    iteration += 1;
    if (accum.reduce() <= tolerance || iteration >= maxIterations) {
      break;
    }
    accum.reset();

  } ///< End while(true).

  galois::runtime::reportStat_Single("PageRank", "Rounds", iteration);
  if (iteration >= maxIterations) {
    std::cerr << "ERROR: failed to converge in " << iteration
              << " iterations\n";
  }
}

void serComputePRTopological(Graph& graph) {
  unsigned int iteration = 0;
  float accum = 0;
  galois::StdForEach loop;

  float base_score = (1.0f - ALPHA) / graph.size();
  while (true) {
    loop(
        galois::iterate(graph),
        [&](const GNode& src) {
          constexpr const galois::MethodFlag flag =
              galois::MethodFlag::UNPROTECTED;

          LNode& sdata = graph.getData(src, flag);
          float sum    = 0.0;

          for (auto jj = graph.edge_begin(src, flag),
                    ej = graph.edge_end(src, flag);
               jj != ej; ++jj) {
            GNode dst = graph.getEdgeDst(jj);

            LNode& ddata = graph.getData(dst, flag);
            sum += ddata.value / ddata.nout;
          }

          //! New value of pagerank after computing contributions from
          //! incoming edges in the original graph.
          float value = sum * ALPHA + base_score;
          //! Find the delta in new and old pagerank values.
          float diff = std::fabs(value - sdata.value);

          //! Do not update pagerank before the diff is computed since
          //! there is a data dependence on the pagerank value.
          sdata.value = value;
          accum += diff;
        },
        galois::loopname("PageRank"));

    iteration += 1;
    if (accum <= tolerance || iteration >= maxIterations) {
      break;
    }
    accum = 0;

  } ///< End while(true).

  galois::runtime::reportStat_Single("PageRank", "Rounds", iteration);
  if (iteration >= maxIterations) {
    std::cerr << "ERROR: failed to converge in " << iteration
              << " iterations\n";
  }
}

// Work items for the OBIM PageRank
struct PageRankWorkItem {
  uint32_t nodeID;
  float diff;
  PageRankWorkItem() : nodeID(0), diff(0){};
  PageRankWorkItem(uint32_t _node, float _diff) : nodeID(_node), diff(_diff) {};
};

struct  PRWorkItemIndexer {
  uint32_t operator() (const PageRankWorkItem& item) {
    return static_cast<uint32_t> (1 / item.diff);
  }
};

namespace gwl = galois::worklists;
using PSchunk = gwl::PerSocketChunkFIFO<CHUNK_SIZE>;
using OBIM    = gwl::OrderedByIntegerMetric<PRWorkItemIndexer, PSchunk>;

void computePRTopoPriority(Graph& graph) {
  unsigned int iteration = 0;
  galois::GAccumulator<float> accum;

  float base_score = (1.0F - ALPHA) / graph.size();

  galois::InsertBag<PageRankWorkItem> activeNodes;

  while (true) {

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          auto& sdata = graph.getData(src);
          activeNodes.push(PageRankWorkItem(src, sdata.diff));
        });

    galois::for_each(
        galois::iterate(activeNodes),
        [&](const PageRankWorkItem& item, auto& ctx) {
          constexpr const galois::MethodFlag flag =
              galois::MethodFlag::UNPROTECTED;

          GNode src = item.nodeID;
          LNode& sdata = graph.getData(src, flag);
          float sum    = 0.0;

          for (auto jj = graph.edge_begin(src, flag),
                    ej = graph.edge_end(src, flag);
               jj != ej; ++jj) {
            GNode dst = graph.getEdgeDst(jj);

            LNode& ddata = graph.getData(dst, flag);
            sum += ddata.value / ddata.nout;
          }

          //! New value of pagerank after computing contributions from
          //! incoming edges in the original graph.
          float value = sum * ALPHA + base_score;
          //! Find the delta in new and old pagerank values.
          float diff = std::fabs(value - sdata.value);

          //! Do not update pagerank before the diff is computed since
          //! there is a data dependence on the pagerank value.
          sdata.value = value;
          sdata.diff  = diff;
          accum += diff;
        },
        galois::loopname("PageRank"), galois::disable_conflict_detection(), galois::no_pushes(),
        galois::wl<OBIM>(PRWorkItemIndexer()));

    iteration += 1;
    if (accum.reduce() <= tolerance || iteration >= maxIterations) {
      break;
    }
    activeNodes.clear();
    accum.reset();

  } ///< End while(true).

  galois::runtime::reportStat_Single("PageRank", "Rounds", iteration);
  if (iteration >= maxIterations) {
    std::cerr << "ERROR: failed to converge in " << iteration
              << " iterations\n";
  }
}
/**
 * PageRank pull topological.
 * Always calculate the new pagerank for each iteration.
 */
void computePRTopological(Graph& graph) {
  unsigned int iteration = 0;
  galois::GAccumulator<float> accum;

  float base_score = (1.0f - ALPHA) / graph.size();
  while (true) {
    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          constexpr const galois::MethodFlag flag =
              galois::MethodFlag::UNPROTECTED;

          LNode& sdata = graph.getData(src, flag);
          float sum    = 0.0;

          for (auto jj = graph.edge_begin(src, flag),
                    ej = graph.edge_end(src, flag);
               jj != ej; ++jj) {
            GNode dst = graph.getEdgeDst(jj);

            LNode& ddata = graph.getData(dst, flag);
            sum += ddata.value / ddata.nout;
          }

          //! New value of pagerank after computing contributions from
          //! incoming edges in the original graph.
          float value = sum * ALPHA + base_score;
          //! Find the delta in new and old pagerank values.
          float diff = std::fabs(value - sdata.value);

          //! Do not update pagerank before the diff is computed since
          //! there is a data dependence on the pagerank value.
          sdata.value = value;
          accum += diff;
        },
        galois::no_stats(), galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
        galois::loopname("PageRank"));

#if DEBUG
    std::cout << "iteration: " << iteration << " max delta: " << delta << "\n";
#endif

    iteration += 1;
    if (accum.reduce() <= tolerance || iteration >= maxIterations) {
      break;
    }
    accum.reset();

  } ///< End while(true).

  galois::runtime::reportStat_Single("PageRank", "Rounds", iteration);
  if (iteration >= maxIterations) {
    std::cerr << "ERROR: failed to converge in " << iteration
              << " iterations\n";
  }
}

void prTopoSerial(Graph& graph) {
  initNodeDataTopological(graph);
  computeOutDeg(graph);

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  serComputePRTopological(graph);
  execTime.stop();
}

void prTopological(Graph& graph) {
  initNodeDataTopological(graph);
  computeOutDeg(graph);

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  computePRTopological(graph);
  execTime.stop();
}

void prResidual(Graph& graph) {
  DeltaArray delta;
  delta.allocateInterleaved(graph.size());
  ResidualArray residual;
  residual.allocateInterleaved(graph.size());

  initNodeDataResidual(graph, delta, residual);
  computeOutDeg(graph);

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  computePRResidual(graph, delta, residual);
  execTime.stop();
}

void prTopoSchedule(Graph& graph) {
  initNodeDataTopological(graph);
  computeOutDeg(graph);

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  computePRTopoSchedule(graph);
  execTime.stop();
}

void prTopoPriority(Graph& graph) {
  initNodeDataTopological(graph);
  computeOutDeg(graph);

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  computePRTopoPriority(graph);
  execTime.stop();
}

void writeOutput(const std::string& output_dir, Graph& graph, std::string ouput_filename="output") {
  namespace fs = boost::filesystem;
  fs::path filename{output_dir};
  filename = filename.append(ouput_filename);

  std::ofstream outputFile(filename.string().c_str());

  if (!outputFile) {
    std::cerr << "could not open file: " << filename << std::endl;
  }

  for(size_t i = 0; i< graph.size(); ++i) {
    outputFile << i << " " << std::to_string(graph.getData(i).value) << std::endl;
  }

  if (!outputFile) {
    std::cerr << "failed tp write file: " << filename << std::endl;
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url, &inputFile);

  if (!transposedGraph) {
    GALOIS_DIE("This application requires a transposed graph input;"
               " please use the -transposedGraph flag "
               " to indicate the input is a transposed graph.");
  }
  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  Graph transposeGraph;
  std::cout << "WARNING: pull style algorithms work on the transpose of the "
               "actual graph\n"
            << "WARNING: this program assumes that " << inputFile
            << " contains transposed representation\n\n"
            << "Reading graph: " << inputFile << "\n";

  galois::graphs::readGraph(transposeGraph, inputFile);
  std::cout << "Read " << transposeGraph.size() << " nodes, "
            << transposeGraph.sizeEdges() << " edges\n";

  galois::preAlloc(2 * numThreads + (3 * transposeGraph.size() *
                                     sizeof(typename Graph::node_data_type)) /
                                        galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  switch (algo) {
  case Topo:
    std::cout << "Running Pull Topological version, tolerance:" << tolerance
              << ", maxIterations:" << maxIterations << "\n";
    prTopological(transposeGraph);
    break;
  case Residual:
    std::cout << "Running Pull Residual version, tolerance:" << tolerance
              << ", maxIterations:" << maxIterations << "\n";
    prResidual(transposeGraph);
    break;
  case TopoSchedule:
    std::cout << "Running Pull Topological Schedule version, tolerance:" << tolerance
              << ", maxIterations:" << maxIterations << "\n";
    prTopoSchedule(transposeGraph);
    break;
  case SerTopo:
    std::cout << "Running Pull Topological serial version, tolerance:" << tolerance
              << ", maxIterations:" << maxIterations << "\n";
    prTopoSerial(transposeGraph);
    break;
  case TopoPriority:
    std::cout << "Running Pull Topological Priority version, tolerance:" << tolerance
              << ", maxIterations:" << maxIterations << "\n";
    prTopoPriority(transposeGraph);
    break;
  default:
    std::abort();
  }

  galois::reportPageAlloc("MeminfoPost");

  //! Sanity checking code.
  galois::GReduceMax<PRTy> maxRank;
  galois::GReduceMin<PRTy> minRank;
  galois::GAccumulator<PRTy> distanceSum;
  maxRank.reset();
  minRank.reset();
  distanceSum.reset();

  //! [example of no_stats]
  galois::do_all(
      galois::iterate(transposeGraph),
      [&](uint64_t i) {
        PRTy rank = transposeGraph.getData(i).value;

        maxRank.update(rank);
        minRank.update(rank);
        distanceSum += rank;
      },
      galois::loopname("Sanity check"), galois::no_stats());
  //! [example of no_stats]

  PRTy rMaxRank = maxRank.reduce();
  PRTy rMinRank = minRank.reduce();
  PRTy rSum     = distanceSum.reduce();
  galois::gInfo("Max rank is ", rMaxRank);
  galois::gInfo("Min rank is ", rMinRank);
  galois::gInfo("Sum is ", rSum);

  if (!skipVerify) {
    printTop(transposeGraph);
  }

#if DEBUG
  printPageRank(transposeGraph);
#endif

  if (outputToFile) {
    writeOutput(outputLocation, transposeGraph);
  }

  totalTime.stop();

  return 0;
}
