/** Simplified executor for just bulk synchronous execution -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_EXECUTOR_BULKSYNCHRONOUS_H
#define GALOIS_RUNTIME_EXECUTOR_BULKSYNCHRONOUS_H

#include "Galois/Runtime/Executor_ForEach.h"
#include "Galois/Runtime/LoopStatistics.h"

namespace Galois {
namespace Runtime {
namespace BulkSynchronousImpl {

template<typename T, bool isLIFO, unsigned ChunkSize>
struct FixedSizeRingAdaptor: public Galois::FixedSizeRing<T,ChunkSize> {
  typedef typename FixedSizeRingAdaptor::reference reference;

  reference cur() { return isLIFO ? this->front() : this->back();  }

  template<typename U>
  void push(U&& val) {
    this->push_front(std::forward<U>(val));
  }

  void pop()  {
    if (isLIFO) this->pop_front();
    else this->pop_back();
  }
};

struct WID {
  unsigned tid;
  unsigned pid;
  WID(unsigned t): tid(t) {
    pid = LL::getLeaderForThread(tid);
  }
  WID() {
    tid = LL::getTID();
    pid = LL::getLeaderForThread(tid);
  }
};

template<typename T,template<typename,bool> class OuterTy, bool isLIFO,int ChunkSize>
class dChunkedMaster : private boost::noncopyable {
  class Chunk : public FixedSizeRingAdaptor<T,isLIFO,ChunkSize>, public OuterTy<Chunk,true>::ListNode {};

  MM::FixedSizeAllocator<Chunk> alloc;

  struct p {
    Chunk* next;
  };

  typedef OuterTy<Chunk, true> LevelItem;

  PerThreadStorage<p> data;
  PerPackageStorage<LevelItem> Q;

  Chunk* mkChunk() {
    Chunk* ptr = alloc.allocate(1);
    alloc.construct(ptr);
    return ptr;
  }
  
  void delChunk(Chunk* ptr) {
    alloc.destroy(ptr);
    alloc.deallocate(ptr, 1);
  }

  void pushChunk(const WID& id, Chunk* C)  {
    LevelItem& I = *Q.getLocal(id.pid);
    I.push(C);
  }

  Chunk* popChunkByID(unsigned int i)  {
    LevelItem* I = Q.getRemote(i);
    if (I)
      return I->pop();
    return 0;
  }

  Chunk* popChunk(const WID& id)  {
    Chunk* r = popChunkByID(id.pid);
    if (r)
      return r;
    
    for (unsigned int i = id.pid + 1; i < Q.size(); ++i) {
      r = popChunkByID(i);
      if (r) 
	return r;
    }

    for (unsigned int i = 0; i < id.pid; ++i) {
      r = popChunkByID(i);
      if (r)
	return r;
    }

    return 0;
  }

  void pushSP(const WID& id, p& n, const T& val);
  bool emptySP(const WID& id, p& n);
  void popSP(const WID& id, p& n);

public:
  typedef T value_type;

  dChunkedMaster() {
    for (unsigned int i = 0; i < data.size(); ++i) {
      p& r = *data.getRemote(i);
      r.next = 0;
    }
  }

  void push(const WID& id, const value_type& val)  {
    p& n = *data.getLocal(id.tid);
    if (n.next && !n.next->full()) {
      n.next->push(val);
      return;
    }
    pushSP(id, n, val);
  }

  unsigned currentChunkSize(const WID& id) {
    p& n = *data.getLocal(id.tid);
    if (n.next) {
      return n.next->size();
    }
    return 0;
  }

  template<typename Iter>
  void push(const WID& id, Iter b, Iter e) {
    while (b != e)
      push(id, *b++);
  }

  template<typename Iter>
  void push_initial(const WID& id, Iter b, Iter e) {
    push(id, b, e);
  }

  value_type& cur(const WID& id) {
    p& n = *data.getLocal(id.tid);
    return n.next->cur();
  }

  bool empty(const WID& id) {
    p& n = *data.getRemote(id.tid);
    if (n.next && !n.next->empty())
      return false;
    return emptySP(id, n);
  }

  bool sempty() {
    WID id;
    for (unsigned i = 0; i < data.size(); ++i) {
      id.tid = i;
      id.pid = LL::getLeaderForThread(i);
      if (!empty(id))
        return false;
    }
    return true;
  }

  void pop(const WID& id)  {
    p& n = *data.getLocal(id.tid);
    if (n.next && !n.next->empty()) {
      n.next->pop();
      return;
    }
    popSP(id, n);
  }
};

template<typename T,template<typename,bool> class OuterTy, bool isLIFO,int ChunkSize>
void dChunkedMaster<T,OuterTy,isLIFO,ChunkSize>::popSP(const WID& id, p& n) {
  while (true) {
    if (n.next && !n.next->empty()) {
      n.next->pop();
      return;
    }
    if (n.next)
      delChunk(n.next);
    n.next = popChunk(id);
    if (!n.next)
      return;
  }
}

template<typename T,template<typename,bool> class OuterTy, bool isLIFO,int ChunkSize>
bool dChunkedMaster<T,OuterTy,isLIFO,ChunkSize>::emptySP(const WID& id, p& n) {
  while (true) {
    if (n.next && !n.next->empty())
      return false;
    if (n.next)
      delChunk(n.next);
    n.next = popChunk(id);
    if (!n.next)
      return true;
  }
}

template<typename T,template<typename,bool> class OuterTy, bool isLIFO,int ChunkSize>
void dChunkedMaster<T,OuterTy,isLIFO,ChunkSize>::pushSP(const WID& id, p& n, const T& val) {
  if (n.next)
    pushChunk(id, n.next);
  n.next = mkChunk();
  n.next->push(val);
}

// TODO: Switch to thread-local worklists
template<typename T,int ChunkSize>
class Worklist: public dChunkedMaster<T, WorkList::ConExtLinkedQueue, true, ChunkSize> { };

template<class T, class FunctionTy, class ArgsTy>
class Executor {
  typedef T value_type;
  typedef Worklist<value_type,256> WLTy;

  static const bool needsStats = !exists_by_supertype<does_not_need_stats_tag, ArgsTy>::value;
  static const bool needsPush = !exists_by_supertype<does_not_need_push_tag, ArgsTy>::value;
  static const bool needsAborts = !exists_by_supertype<does_not_need_aborts_tag, ArgsTy>::value;
  static const bool needsPia = exists_by_supertype<needs_per_iter_alloc_tag, ArgsTy>::value;
  static const bool needsBreak = exists_by_supertype<needs_parallel_break_tag, ArgsTy>::value;

  struct ThreadLocalData {
    Galois::Runtime::UserContextAccess<value_type> facing;
    SimpleRuntimeContext ctx;
    LoopStatistics<needsStats> stat;
    ThreadLocalData(const char* ln): stat(ln) { }
  };

  WLTy wls[2];
  FunctionTy function;
  const char* loopname;
  Galois::Runtime::Barrier& barrier;
  LL::CacheLineStorage<volatile long> done;

  bool empty(WLTy* wl) {
    return wl->sempty();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void abortIteration(ThreadLocalData& tld, const WID& wid, WLTy* cur, WLTy* next) {
    tld.ctx.cancelIteration();
    tld.stat.inc_conflicts();
    if (needsPush) {
      tld.facing.resetPushBuffer();
    }
    value_type& val = cur->cur(wid);
    next->push(wid, val);
    cur->pop(wid);
  }

  void processWithAborts(ThreadLocalData& tld, const WID& wid, WLTy* cur, WLTy* next) {
    int result = 0;
#ifdef GALOIS_USE_LONGJMP
    if ((result = setjmp(hackjmp)) == 0) {
#else
    try {
#endif
      process(tld, wid, cur, next);
#ifdef GALOIS_USE_LONGJMP
    } else { clearConflictLock(); }
#else
    } catch (const ConflictFlag& flag) { clearConflictLock(); result = flag; }
#endif
    clearReleasable(); 
    switch (result) {
    case 0: break;
    case Galois::Runtime::CONFLICT:
      abortIteration(tld, wid, cur, next);
      break;
    case Galois::Runtime::BREAK:
    default:
      abort();
    }
  }

  void process(ThreadLocalData& tld, const WID& wid, WLTy* cur, WLTy* next) {
    int cs = std::max(cur->currentChunkSize(wid), 1U);
    for (int i = 0; i < cs; ++i) {
      value_type& val = cur->cur(wid);
      tld.stat.inc_iterations();
      function(val, tld.facing.data());
      if (needsPush) {
        next->push(wid,
            tld.facing.getPushBuffer().begin(),
            tld.facing.getPushBuffer().end());
        tld.facing.resetPushBuffer();
      }
      if (needsAborts)
        tld.ctx.commitIteration();
      cur->pop(wid);
    }
  }

  void go() {
    ThreadLocalData tld(loopname);
    setThreadContext(&tld.ctx);
    unsigned tid = LL::getTID();
    WID wid;

    WLTy* cur = &wls[0];
    WLTy* next = &wls[1];

    while (true) {
      while (!cur->empty(wid)) {
        if (needsAborts) {
          processWithAborts(tld, wid, cur, next);
        } else {
          process(tld, wid, cur, next);
        }
        if (needsPia)
          tld.facing.resetAlloc();
      }

      std::swap(next, cur);

      barrier.wait();

      if (tid == 0) {
        if (empty(cur))
          done.get() = true;
      }
      
      barrier.wait();

      if (done.get())
        break;
    }

    setThreadContext(0);
  }

public:
  static_assert(!needsBreak, "not supported by this executor");
  
  Executor(const FunctionTy& f, const ArgsTy& args):
    function(f), 
    loopname(get_by_supertype<loopname_tag>(args).value),
    barrier(getSystemBarrier()) { }

  template<typename RangeTy>
  void init(const RangeTy& range) { }
  
  template<typename RangeTy>
  void initThread(const RangeTy& range) {
    wls[0].push_initial(WID(), range.local_begin(), range.local_end());
  }

  void operator()() {
    go();
  }
};

}
}

namespace WorkList {

template<typename T=int>
struct BulkSynchronousInline {
  template<bool _concurrent>
  struct rethread { typedef BulkSynchronousInline<T> type; };

  template<typename _T>
  struct retype { typedef BulkSynchronousInline<_T> type; };

  typedef T value_type;
};

}

namespace Runtime {

template<class T, class FunctionTy, class ArgsTy>
struct ForEachExecutor<WorkList::BulkSynchronousInline<T>, FunctionTy, ArgsTy>:
  public BulkSynchronousImpl::Executor<T, FunctionTy, ArgsTy> 
{
  typedef BulkSynchronousImpl::Executor<T, FunctionTy, ArgsTy> SuperTy;
  ForEachExecutor(const FunctionTy& f, const ArgsTy& args): SuperTy(f, args) { }
};

}

}
#endif