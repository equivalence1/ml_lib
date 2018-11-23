#pragma once

#include <cstdint>
#include <cassert>

template <class T, class Impl>
class VecOps {
public:
    VecOps() {
    }

    VecOps(VecOps&& other) = default;
    VecOps(const VecOps& other) = default;

    void set(int64_t idx, double val) {
        assert(idx < implSize());
        implData()[idx] = val;
    }

    double get(int64_t idx) const {
        assert((idx) < implSize());
        return implData()[idx];
    }

    int64_t dim() const {
        return implSize();
    }

private:

    T* implData() {
        return static_cast<Impl*>(this)->data();
    }

    const T* implData() const {
        return static_cast<const Impl*>(this)->data();
    }

    int64_t implSize() const {
        return static_cast<const Impl*>(this)->size();
    }
};
