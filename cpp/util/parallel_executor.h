#pragma once

#include "singleton.h"
#include <cstdint>
#include <functional>

#include <ATen/core/thread_pool.h>
#include <thread>

class ThreadPool {
public:
    ThreadPool();

    template <class Task>
    void enqueue(Task&& task) {
        pool_.run(std::forward<Task>(task));
    }

    void waitComplete() {
        pool_.waitWorkComplete();
    }
private:

    c10::ThreadPool pool_;
};


inline ThreadPool& GlobalThreadPool() {
    return Singleton<ThreadPool>();
}




