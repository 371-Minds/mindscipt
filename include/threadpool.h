#ifndef BN_THREADPOOL_H
#define BN_THREADPOOL_H

// Persistent pthread thread pool for parallel dispatch.
// Replaces OpenMP fork/join with ~2us condvar dispatch.

typedef void (*bn_tp_fn)(void *ctx, int start, int end);

typedef struct {
    bn_tp_fn fn;    // range function: called with [start, end)
    void *ctx;   // opaque context pointer
    int   n;     // iteration count
} BnTPTask;

typedef struct BnThreadPool BnThreadPool;

// Create a thread pool with n_workers background threads.
// Main thread participates as thread 0 (not counted in n_workers).
BnThreadPool *bn_tp_create(int n_workers);

// Destroy the thread pool, joining all worker threads.
void bn_tp_free(BnThreadPool *pool);

// Dispatch tasks to the pool. Blocks until all tasks complete.
// If pool is NULL, runs serially on the calling thread.
// Multi-task dispatch (2-3 tasks) uses a single wake/wait cycle.
void bn_tp_dispatch(BnThreadPool *pool, const BnTPTask *tasks, int n_tasks);

// Returns total thread count (n_workers + 1 for main thread).
int bn_tp_num_threads(const BnThreadPool *pool);

#endif // BN_THREADPOOL_H
