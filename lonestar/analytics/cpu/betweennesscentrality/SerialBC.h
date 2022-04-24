#ifndef GALOIS_SERIALBC_H
#define GALOIS_SERIALBC_H

#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "Lonestar/BoilerPlate.h"
#include <boost/iterator/filter_iterator.hpp>

#include <iomanip>
#include <fstream>

using ShortPathType = double;

struct NodeData {
  uint32_t currentDistance;
  ShortPathType numShortestPaths;
  float dependency;
  float bc;
};

using Graph =
    galois::graphs::LC_CSR_Graph<NodeData, void>::with_no_lockable<
        true>::type::with_numa_alloc<true>::type;
using GNode        = Graph::GraphNode;

class BCSerial {
  Graph* G;
  int NumNodes;

public:
  /**
   * Constructor initializes thread local storage.
   */
  BCSerial(Graph& g) : G(&g), NumNodes(g.size()) { InitializeNodeData(); }

  /**
   * Constructor destroys thread local storage.
   */
  ~BCSerial(void) { }

  /**
   * Initialize node fields all to 0
   */
  void InitializeNodeData() {
    galois::do_all(
        galois::iterate(*G),
        [&](GNode n) {
          NodeData& nodeData = G->getData(n);
          nodeData.currentDistance  = 0;
          nodeData.numShortestPaths = 0;
          nodeData.dependency       = 0;
          nodeData.bc               = 0;
        },
        galois::no_pushes(), galois::loopname("InitializeGraph"));
  }


  //! Function that does BC for a single souce; called by a thread
  void doBC(const GNode curSource) {
    galois::gdeque<GNode> SQ;
    galois::gdeque<GNode>* succ = new galois::gdeque<GNode>[NumNodes]();

    NodeData& curData = G->getData(curSource);

    curData.numShortestPaths = 1;
    curData.currentDistance  = 1;

    SQ.push_back(curSource);

    // Do bfs while computing number of shortest paths (saved into sigma)
    // and successors of nodes;
    // Note this bfs makes it so source has distance of 1 instead of 0
    for (auto qq = SQ.begin(); qq != SQ.end(); ++qq) {
      GNode src = *qq;
      NodeData& srcData = G->getData(src);

      for (auto edge : G->edges(src, galois::MethodFlag::UNPROTECTED)) {
        GNode dest = G->getEdgeDst(edge);
        NodeData& destData = G->getData(dest);

        if (destData.currentDistance <= 0) {
          SQ.push_back(dest);
          destData.currentDistance = srcData.currentDistance + 1;
        }

        if (destData.currentDistance == srcData.currentDistance + 1) {
          destData.numShortestPaths += srcData.numShortestPaths;
          succ[src].push_back(dest);
        }
      }
    }

    // Back-propogate the dependency values (delta) along the BFS DAG
    // ignore the source (hence SQ.size > 1 and not SQ.empty)
    while (SQ.size() > 1) {
      GNode leaf = SQ.back();
      SQ.pop_back();

      NodeData& leafData = G->getData(leaf);
      auto& succ_list = succ[leaf];

      for (auto succ_iter = succ_list.begin(), succ_end = succ_list.end();
           succ_iter != succ_end; ++succ_iter) {
        NodeData& succData = G->getData(*succ_iter);
        leafData.dependency += (leafData.numShortestPaths / succData.numShortestPaths) *
                               (1.0 + succData.dependency);
      }
      leafData.bc += leafData.dependency;
    }

    // save result of this source's BC, reset all local values for next source
    for (int i = 0; i < NumNodes; ++i) {
      NodeData& nodeData        = G->getData(i);
      nodeData.dependency       = 0;
      nodeData.numShortestPaths = 0;
      nodeData.currentDistance  = 0;
    }
  }

  /**
   * Runs betweeness-centrality proper. Instead of a vector of sources,
   * it will operate on the first numSources sources.
   *
   * @param numSources Num sources to get BC contribution for
   */
  void runAll(unsigned numSources) {
    galois::StdForEach loop;
    // Each thread works on an individual source node
    loop(
        galois::iterate(0u, numSources),
        [&](const GNode& curSource) { doBC(curSource); },
        galois::loopname("Main"));
  }

  /**
   * Runs betweeness-centrality proper.
   *
   * @tparam Cont type of the data structure that holds the nodes to treat
   * as a source during betweeness-centrality.
   *
   * @param v Data structure that holds nodes to treat as a source during
   * betweeness-centrality
   */
  template <typename Cont>
  void run(const Cont& v) {
    galois::StdForEach loop;
    // Each thread works on an individual source node
    loop(
        galois::iterate(v),
        [&](const GNode& curSource) { doBC(curSource); },
        galois::loopname("Main"));
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
      NodeData& nodeData = G->getData(begin);
      double bc = nodeData.bc;

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

    printBCValues(0, NumNodes, outf, 9);

    outf.close();
  }

  //! sanity check of BC values
  void serialSanity(Graph& graph) {
    galois::GReduceMax<float> accumMax;
    galois::GReduceMin<float> accumMin;
    galois::GAccumulator<float> accumSum;
    accumMax.reset();
    accumMin.reset();
    accumSum.reset();

    // get max, min, sum of BC values using accumulators and reducers
    galois::do_all(
        galois::iterate(graph),
        [&](GNode n) {
          NodeData& nodeData = graph.getData(n);
          accumMax.update(nodeData.bc);
          accumMin.update(nodeData.bc);
          accumSum += nodeData.bc;
        },
        galois::no_stats(), galois::loopname("SerialBCSanity"));

    galois::gPrint("Max BC is ", accumMax.reduce(), "\n");
    galois::gPrint("Min BC is ", accumMin.reduce(), "\n");
    galois::gPrint("BC sum is ", accumSum.reduce(), "\n");
  }
};

/**
 * Functor that indicates if a node contains outgoing edges
 */
struct HasOutEdge {
  Graph* graph;
  HasOutEdge(Graph* g) : graph(g) {}

  bool operator()(const GNode& n) const {
    return graph->edge_begin(n) != graph->edge_end(n);
  }
};

void doSerialBC() {
  Graph g;
  galois::graphs::readGraph(g, inputFile);

  BCSerial bcSerial(g);

  size_t NumNodes = g.size();

  // preallocate pages for use in algorithm
  galois::reportPageAlloc("MeminfoPre");
  galois::preAlloc(galois::getActiveThreads() * NumNodes / 1650);
  galois::reportPageAlloc("MeminfoMid");

  // vector of sources to process; initialized if doing outSources
  std::vector<GNode> v;
  // preprocessing: find the nodes with out edges we will process and skip
  // over nodes with no out edges; only done if numOfSources isn't specified
  if (numOfSources == 0) {
    // find first node with out edges
    boost::filter_iterator<HasOutEdge, Graph::iterator> begin =
        boost::make_filter_iterator(HasOutEdge(&g), g.begin(), g.end());
    boost::filter_iterator<HasOutEdge, Graph::iterator> end =
        boost::make_filter_iterator(HasOutEdge(&g), g.end(), g.end());
    // adjustedEnd = last node we will process based on how many iterations
    // (i.e. sources) we want to do
    boost::filter_iterator<HasOutEdge, Graph::iterator> adjustedEnd =
        iterLimit ? galois::safe_advance(begin, end, (int)iterLimit) : end;

    size_t iterations = std::distance(begin, adjustedEnd);
    galois::gPrint("Num Nodes: ", NumNodes, " Start Node: ", startSource,
                   " Iterations: ", iterations, "\n");
    // vector of nodes we want to process
    v.insert(v.end(), begin, adjustedEnd);
  }

  //initial
  bcSerial.InitializeNodeData();
  // execute algorithm
  galois::StatTimer execTime("Timer_0");
  execTime.start();
  // either run a contiguous chunk of sources from beginning or run using
  // sources with outgoing edges only
  if (numOfSources > 0) {
    bcSerial.runAll(numOfSources);
  } else {
    bcSerial.run(v);
  }
  execTime.stop();

  bcSerial.printBCValues(0, std::min(10UL, NumNodes), std::cout, 6);
  bcSerial.serialSanity(g);
  if (output)
    bcSerial.printBCcertificate();


  galois::reportPageAlloc("MeminfoPost");
}

#endif // GALOIS_SERIALBC_H
