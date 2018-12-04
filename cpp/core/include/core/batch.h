#pragma once

#include <vector>
#include <cassert>
#include <cstdint>
#include <cassert>
//TODO: could be specialized for vectors, matrix
//here just vector

template <class T>
struct Batch {
    std::vector<T> batch_;

    Batch(std::initializer_list<T> list)
    : batch_(list) {

    }

    T operator[](int64_t idx) {
        assert(idx >= 0);//TODO
        return batch_[idx];
    }

    int64_t size() const {
        return batch_.size();
    }
};
