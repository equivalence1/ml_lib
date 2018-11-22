#pragma once

#include <vector>
#include <cassert>
#include <cstdint>

//TODO: could be specialized for vectors, matrix
//here just vector
template <class T>
struct Batch {
    std::vector<T> Batch;

    T& operator[](int64_t idx) {
        assert(idx >= 0);//TODO
        return Batch[idx];
    }

    const T& operator[](int64_t idx) const {
        assert(idx >= 0);//TODO
        return Batch[idx];
    }

    int64_t size() const {
        return Batch.size();
    }
};
