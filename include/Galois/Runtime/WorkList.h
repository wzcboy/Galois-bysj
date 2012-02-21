// Scalable Local worklists -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#ifndef __WORKLIST_H_
#define __WORKLIST_H_

#include <queue>
#include <stack>
#include <limits>
#include <map>
#include <algorithm>
#include <boost/utility.hpp>
#include <boost/optional.hpp>

#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/Threads.h"
//#include "Galois/Runtime/QueuingLock.h"
#include "Galois/Queue.h"

#include "mem.h"
#include "WorkListHelpers.h"

#ifndef WLCOMPILECHECK
#define WLCOMPILECHECK(name) //
#endif

namespace GaloisRuntime {
namespace WorkList {

// Worklists may not be copied.
// Worklists should be default instantiatable
// All classes (should) conform to:
template<typename T, bool concurrent>
class AbstractWorkList {
public:
  //! T is the value type of the WL
  typedef T value_type;

  //! change the concurrency flag
  template<bool newconcurrent>
  struct rethread {
    typedef AbstractWorkList<T, newconcurrent> WL;
  };

  //! change the type the worklist holds
  template<typename Tnew>
  struct retype {
    typedef AbstractWorkList<Tnew, concurrent> WL;
  };

  //! push a value onto the queue
  void push(value_type val);

  //! push many values onto the queue
  template<typename Iter>
  void push(Iter b, Iter e);

  //! push initial values onto the queue
  template<typename Iter>
  void push_initial(Iter b, Iter e);

  //! pop a value from the queue.
  boost::optional<value_type> pop();
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<typename T = int, bool concurrent = true>
class LIFO : private boost::noncopyable, private LL::PaddedLock<concurrent> {
  std::deque<T> wl;

  using LL::PaddedLock<concurrent>::lock;
  using LL::PaddedLock<concurrent>::try_lock;
  using LL::PaddedLock<concurrent>::unlock;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef LIFO<T, newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef LIFO<Tnew, concurrent> WL;
  };

  typedef T value_type;

  void push(value_type val) {
    lock();
    wl.push_back(val);
    unlock();
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    lock();
    while (b != e)
      wl.push_back(*b++);
    unlock();
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop()  {
    boost::optional<value_type> retval;
    lock();
    if (!wl.empty()) {
      retval = wl.back();
      wl.pop_back();
    }
    unlock();
    return retval;
  }
};
WLCOMPILECHECK(LIFO);

template<typename T = int, bool concurrent = true>
class FIFO : private boost::noncopyable, private LL::PaddedLock<concurrent>  {
  std::deque<T> wl;

  using LL::PaddedLock<concurrent>::lock;
  using LL::PaddedLock<concurrent>::try_lock;
  using LL::PaddedLock<concurrent>::unlock;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef FIFO<T, newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef FIFO<Tnew, concurrent> WL;
  };

  typedef T value_type;

  void push(value_type val) {
    lock();
    wl.push_back(val);
    unlock();
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    lock();
    while (b != e)
      wl.push_back(*b++);
    unlock();
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> retval;
    lock();
    if (!wl.empty()) {
      retval = wl.front();
      wl.pop_back();
    }
    unlock();
    return retval;
  }
};
WLCOMPILECHECK(FIFO);

//#define ASDF 1
template<class Indexer = DummyIndexer<int>, typename ContainerTy = FIFO<>, typename T = int, bool concurrent = true >
class OrderedByIntegerMetric : private boost::noncopyable {

  typedef typename ContainerTy::template rethread<concurrent>::WL CTy;

  struct perItem {
    CTy* current;
    unsigned int curVersion;
    unsigned int lastMasterVersion;
    std::map<unsigned int, CTy*> local;
#if ASDF
    std::map<unsigned int, bool> cache;
#endif
  };

  std::vector<std::pair<unsigned int, CTy*> > masterLog;
  LL::PaddedLock<concurrent> masterLock;
  unsigned int masterVersion;

  Indexer I;

  PerCPU<perItem> current;

  void updateLocal_i(perItem& p) {
    //ASSERT masterLock
    for (; p.lastMasterVersion < masterVersion; ++p.lastMasterVersion) {
      std::pair<unsigned int, CTy*> logEntry = masterLog[p.lastMasterVersion];
      p.local[logEntry.first] = logEntry.second;
      assert(logEntry.second);
#if ASDF
      p.cache[logEntry.first] = true;
#endif
    }
  }

  void updateLocal(perItem& p) {
    if (p.lastMasterVersion != masterVersion) {
      masterLock.lock();
      updateLocal_i(p);
      masterLock.unlock();
    }
  }

  CTy* updateLocalOrCreate(perItem& p, unsigned int i) {
    //Try local then try update then find again or else create and update the master log
    CTy*& lC = p.local[i];
    if (lC)
      return lC;
    masterLock.lock();
    updateLocal_i(p);
    if (!lC) {
      lC = new CTy();
      ++masterVersion;
      masterLog.push_back(std::make_pair(i, lC));
    }
    masterLock.unlock();
    return lC;
  }

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef OrderedByIntegerMetric<Indexer,ContainerTy,T,newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef OrderedByIntegerMetric<Indexer,typename ContainerTy::template retype<Tnew>::WL,Tnew,concurrent> WL;
  };

  typedef T value_type;

  OrderedByIntegerMetric(const Indexer& x = Indexer())
    :masterVersion(0), I(x)
  {
    for (unsigned int i = 0; i < current.size(); ++i) {
      current.get(i).current = NULL;
      current.get(i).lastMasterVersion = 0;
    }
  }

  ~OrderedByIntegerMetric() {
    for (typename std::vector<std::pair<unsigned int, CTy*> >::iterator ii = masterLog.begin(), ee = masterLog.end(); ii != ee; ++ii) {
      delete ii->second;
    }
  }

  void push(value_type val) {
    unsigned int index = I(val);
    perItem& p = current.get();
    //fastpath
    if (index == p.curVersion && p.current) {
      p.current->push(val);
      return;
    }
    //slow path
    CTy* lC = updateLocalOrCreate(p, index);
    //opportunistically move to higher priority work
#if ASDF
    if (index < p.curVersion) {
      p.curVersion = index;
      p.current = lC;
    }
    p.cache[index] = true;
#endif
    lC->push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop() {
    //Find a successful pop
    perItem& p = current.get();
    CTy*& C = p.current;
    boost::optional<value_type> retval;
    if (C && (retval = C->pop()))
      return retval;
    //Failed, find minimum bin
#if ASDF
    {
      //ltbb-style
      typename std::map<unsigned int, bool>::iterator ii = p.cache.begin(), ee = p.cache.end(), old;
      while (ii != ee) {
        p.curVersion = ii->first;
        C = p.local[ii->first];
        // why can C be null?
        if (C && (retval = C->pop()).first) {
          return retval;
        }
        old = ii;
        ++ii;
        p.cache.erase(old);
      }
    }
#endif

    updateLocal(p);
    for (typename std::map<unsigned int, CTy*>::iterator ii = p.local.begin(),
        ee = p.local.end(); ii != ee; ++ii) {
      p.curVersion = ii->first;
      C = ii->second;
      if ((retval = C->pop())) {
#if ASDF
        p.cache[ii->first] = true;
#endif
	return retval;
      } else {
#if ASDF
        p.cache.erase(ii->first);
#endif
      }
    }
    return boost::optional<value_type>();
  }
};
WLCOMPILECHECK(OrderedByIntegerMetric);

template<typename GlobalQueueTy = FIFO<>, typename LocalQueueTy = FIFO<>, typename T = int >
class LocalQueues : private boost::noncopyable {

  PerCPU<typename LocalQueueTy::template rethread<false>::WL> local;
  GlobalQueueTy global;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef LocalQueues<GlobalQueueTy, LocalQueueTy, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef LocalQueues<typename GlobalQueueTy::template retype<Tnew>::WL, typename LocalQueueTy::template retype<Tnew>::WL, Tnew> WL;
  };

  typedef T value_type;

  LocalQueues() {}

  void push(value_type val) {
    return local.get().push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    local.get().push(b,e);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    global.push_initial(b,e);
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> ret = local.get().pop();
    if (ret)
      return ret;
    return global.pop();
  }
};
WLCOMPILECHECK(LocalQueues);

template<typename ContainerTy = FIFO<>, typename T = int >
class LocalStealing : private boost::noncopyable {

  PerCPU<typename ContainerTy::template rethread<true>::WL> local;

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef LocalStealing<ContainerTy, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef LocalStealing<typename ContainerTy::template retype<Tnew>::WL, Tnew> WL;
  };

  typedef T value_type;
  
  LocalStealing() {}

  void push(value_type val) {
    local.get().push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    local.get().push(b,e);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    local.get().push_initial(b,e);
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> ret = local.get().pop();
    if (ret)
      return ret;
    return local.getNext(ThreadPool::getActiveThreads()).pop();
  }
};
WLCOMPILECHECK(LocalStealing);

template<typename ContainerTy = FIFO<>, typename T = int >
class LevelStealing : private boost::noncopyable {

  PerLevel<typename ContainerTy::template rethread<true>::WL> local;

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef LevelStealing<ContainerTy, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef LevelStealing<typename ContainerTy::template retype<Tnew>::WL, Tnew> WL;
  };

  typedef T value_type;
  
  LevelStealing() {}

  void push(value_type val) {
    local.get().push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    local.get().push(b,e);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    local.get().push_initial(b,e);
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> ret = local.get().pop();
    if (ret)
      return ret;

    int mp = LL::getMaxPackageForThread(ThreadPool::getActiveThreads() - 1);
    int id = local.myEffectiveID();
    for (unsigned i = 0; i < local.size(); ++i) {
      ++id;
      id %= local.size();
      if (id <= mp) {
	ret = local.get(id).pop();
	if (ret)
	  return ret;
      }
    }
    return ret;
  }
};
WLCOMPILECHECK(LevelStealing);

//This overly complex specialization avoids a pointer indirection for non-distributed WL when accessing PerLevel
template<bool d, typename TQ>
struct squeues;

template<typename TQ>
struct squeues<true,TQ> {
  PerLevel<TQ> queues;
  TQ& get(int i) { return queues.get(i); }
  TQ& get() { return queues.get(); }
  int myEffectiveID() { return queues.myEffectiveID(); }
  int size() { return queues.size(); }
};

template<typename TQ>
struct squeues<false,TQ> {
  TQ queue;
  TQ& get(int i) { return queue; }
  TQ& get() { return queue; }
  int myEffectiveID() { return 0; }
  int size() { return 0; }
};

template<typename T, template<typename, bool> class QT, bool distributed = false, bool isStack = false, int chunksize=64, bool concurrent=true>
class ChunkedMaster : private boost::noncopyable {
  class Chunk : public FixedSizeRing<T, chunksize, false>, public QT<Chunk, concurrent>::ListNode {};

  MM::FixedSizeAllocator heap;

  struct p {
    Chunk* cur;
    Chunk* next;
  };

  typedef QT<Chunk, concurrent> LevelItem;

  PerCPU<p> data;
  squeues<distributed, LevelItem> Q;

  Chunk* mkChunk() {
    return new (heap.allocate(sizeof(Chunk))) Chunk();
  }
  
  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
  }

  void pushChunk(Chunk* C)  {
    LevelItem& I = Q.get();
    I.push(C);
  }

  Chunk* popChunkByID(unsigned int i)  {
    LevelItem& I = Q.get(i);
    return I.pop();
  }

  Chunk* popChunk()  {
    int id = Q.myEffectiveID();
    Chunk* r = popChunkByID(id);
    if (r)
      return r;
    
    // int mp = LL::getMaxPackageForThread(ThreadPool::getActiveThreads() - 1);
    // for (int i = 0; i < Q.size(); ++i) {
    //   ++id;
    //   id %= Q.size();
    //   if (id <= mp) {
    // 	r = popChunkByID(id);
    // 	if (r)
    // 	  return r;
    //   }
    // }

    for (int i = id + 1; i < (int) Q.size(); ++i) {
      r = popChunkByID(i);
      if (r) 
	return r;
    }

    for (int i = 0; i < id; ++i) {
      r = popChunkByID(i);
      if (r)
	return r;
    }

    return 0;
  }

public:
  typedef T value_type;

  template<bool newconcurrent>
  struct rethread {
    typedef ChunkedMaster<T, QT, distributed, isStack, chunksize, newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef ChunkedMaster<Tnew, QT, distributed, isStack, chunksize, concurrent> WL;
  };

  ChunkedMaster() : heap(sizeof(Chunk)) {
    for (unsigned int i = 0; i < data.size(); ++i) {
      p& r = data.get(i);
      r.cur = 0;
      r.next = 0;
    }
  }

  void push(value_type val)  {
    p& n = data.get();
    if (n.next && n.next->push_back(val))
      return;
    if (n.next)
      pushChunk(n.next);
    n.next = mkChunk();
    bool worked = n.next->push_back(val);
    assert(worked);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop()  {
    p& n = data.get();
    boost::optional<value_type> retval;
    if (isStack) {
      if (n.next && (retval = n.next->pop_back()))
	return retval;
      if (n.next)
	delChunk(n.next);
      n.next = popChunk();
      if (n.next)
	return n.next->pop_back();
      return boost::optional<value_type>();
    } else {
      if (n.cur && (retval = n.cur->pop_front()))
	return retval;
      if (n.cur)
	delChunk(n.cur);
      n.cur = popChunk();
      if (!n.cur) {
	n.cur = n.next;
	n.next = 0;
      }
      if (n.cur)
	return n.cur->pop_front();
      return boost::optional<value_type>();
    }
  }
};

template<int chunksize=64, typename T = int, bool concurrent=true>
class ChunkedFIFO : public ChunkedMaster<T, ConExtLinkedQueue, false, false, chunksize, concurrent> {};
WLCOMPILECHECK(ChunkedFIFO);

template<int chunksize=64, typename T = int, bool concurrent=true>
class ChunkedLIFO : public ChunkedMaster<T, ConExtLinkedStack, false, true, chunksize, concurrent> {};
WLCOMPILECHECK(ChunkedLIFO);

template<int chunksize=64, typename T = int, bool concurrent=true>
class dChunkedFIFO : public ChunkedMaster<T, ConExtLinkedQueue, true, false, chunksize, concurrent> {};
WLCOMPILECHECK(dChunkedFIFO);

template<int chunksize=64, typename T = int, bool concurrent=true>
class dChunkedLIFO : public ChunkedMaster<T, ConExtLinkedStack, true, true, chunksize, concurrent> {};
WLCOMPILECHECK(dChunkedLIFO);

template<typename Partitioner = DummyPartitioner, typename T = int, typename ChildWLTy = dChunkedFIFO<>, bool concurrent=true>
class PartitionedWL : private boost::noncopyable {

  Partitioner P;
  PerCPU<ChildWLTy> Items;
  int active;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef PartitionedWL<T, Partitioner, ChildWLTy, newconcurrent> WL;
  };

  typedef T value_type;
  
  PartitionedWL(const Partitioner& p = Partitioner()) :P(p), active(ThreadPool::getActiveThreads()) {
    //std::cerr << active << "\n";
  }

  void push(value_type val)  {
    unsigned int index = P(val);
    //std::cerr << "[" << index << "," << index % active << "]\n";
    Items.get(index % active).push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop()  {
    return Items.get().pop();
  }
};

template<class Compare = std::less<int>, typename T = int>
class SkipListQueue : private boost::noncopyable {

  Galois::ConcurrentSkipListMap<T,int,Compare> wl;
  int magic;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef SkipListQueue<Compare, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef SkipListQueue<Compare, Tnew> WL;
  };

  typedef T value_type;

  void push(value_type val) {
    wl.putIfAbsent(val, &magic);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop() {
    return wl.pollFirstKey();
  }
};
WLCOMPILECHECK(SkipListQueue);

template<class Compare = std::less<int>, typename T = int>
class FCPairingHeapQueue : private boost::noncopyable {

  Galois::FCPairingHeap<T,Compare> wl;

public:

  template<bool newconcurrent>
  struct rethread {
    typedef FCPairingHeapQueue<Compare, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef FCPairingHeapQueue<Compare, Tnew> WL;
  };

  typedef T value_type;

  void push(value_type val) {
    wl.add(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop() {
    return wl.pollMin();
  }
};
WLCOMPILECHECK(FCPairingHeapQueue);

//End namespace
}
}

#endif
