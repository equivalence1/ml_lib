#pragma once

#include <mutex>
#include <condition_variable>

class Semaphore {
public:
    Semaphore();

    void acquire();
    void release();

    ~Semaphore() = default;

private:
    unsigned int cnt_;
    std::condition_variable cv_;
    std::mutex mtx_;
};

class SemaphoreReleaseGuard {
public:
    SemaphoreReleaseGuard(Semaphore& sema, unsigned int toRelease);

    ~SemaphoreReleaseGuard();

private:
    Semaphore& sema_;
    unsigned int toRelease_;
};

class SemaphoreAcquireGuard {
public:
    SemaphoreAcquireGuard(Semaphore& sema, unsigned int toAcquire);

    ~SemaphoreAcquireGuard();

private:
    Semaphore& sema_;
    unsigned int toAcquire_;
};
