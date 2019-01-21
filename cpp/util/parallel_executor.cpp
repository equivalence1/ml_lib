#include <parallel_executor.h>

//tiny wrapper for torch::thread_pool

#include <ATen/core/thread_pool.h>
#include <thread>
#include "singleton.h"
#include "parallel_executor.h"

namespace {
    struct ParallelExecutor {
        ParallelExecutor()
            : pool_(std::thread::hardware_concurrency()) {
        }

        c10::ThreadPool pool_;
    };
}


inline ParallelExecutor& Executor() {
    return Singleton<ParallelExecutor>();
}

void parallelFor(int64_t from, int64_t to, std::function<void(int64_t)> func) {
    //implement me
    assert(false);

}
