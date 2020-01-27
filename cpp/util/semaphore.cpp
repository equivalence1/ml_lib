#include "semaphore.h"

Semaphore::Semaphore() : cnt_(0) {

}

void Semaphore::acquire() {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [&](){return cnt_ >= 1;});
    cnt_ -= 1;
}

void Semaphore::release() {
    {
        std::unique_lock<std::mutex> lk(mtx_);
        cnt_ += 1;
    }
    cv_.notify_one();
}

SemaphoreReleaseGuard::SemaphoreReleaseGuard(Semaphore& sema, unsigned int toRelease)
        : sema_(sema)
        , toRelease_(toRelease) {

}

SemaphoreReleaseGuard::~SemaphoreReleaseGuard() {
    for (unsigned int i = 0; i < toRelease_; ++i) {
        sema_.release();
    }
}

SemaphoreAcquireGuard::SemaphoreAcquireGuard(Semaphore& sema, unsigned int toAcquire)
        : sema_(sema)
        , toAcquire_(toAcquire) {

}

SemaphoreAcquireGuard::~SemaphoreAcquireGuard() {
    for (unsigned int i = 0; i < toAcquire_; ++i) {
        sema_.acquire();
    }
}

