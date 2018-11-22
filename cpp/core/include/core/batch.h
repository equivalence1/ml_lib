#pragma once

#include <vector>
#include <cassert>
#include <cstdint>

//TODO: could be specialized for vectors, matrix
//here just vector
template <class T>
struct Batch {
    std::vector<T> batch;

    T& operator[](int64_t idx) {
        assert(idx >= 0);//TODO
        return batch[idx];
    }

    const T& operator[](int64_t idx) const {
        assert(idx >= 0);//TODO
        return batch[idx];
    }

    int64_t size() const {
        return batch.size();
    }
};
