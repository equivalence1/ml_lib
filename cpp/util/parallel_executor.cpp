#include "parallel_executor.h"
#include "singleton.h"

//tiny wrapper for torch::thread_pool



ThreadPool::ThreadPool()
    : pool_(8) {

}
