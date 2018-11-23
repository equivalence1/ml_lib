#pragma once

#include "fwd.h"
#include <cstdint>
#include <variant>
#include <memory>

class Vec {
public:

    Vec(Vec& other) = default;

    Vec(const Vec& other) = default;

    Vec(Vec&& other) = default;

    void set(int64_t index, double value);

    double get(int64_t index) const;

    double operator()(int64_t index) const {
        return get(index);
    }

    int64_t dim() const;

//    Vec slice(int64_t from, int64_t to);
//    Vec slice(int64_t from, int64_t to) const;

    using Data =  std::variant<ArrayVecPtr,
                               VecRefPtr,
                               #if defined(CUDA)
                               CudaVecPtr,
                               #endif
                               SingleElemVecPtr>;

    Data& data() {
        return data_;
    }
    const Data& data() const {
        return data_;
    }
protected:
    template <class T>
    using Ptr = std::shared_ptr<T>;

    template <class T>
    explicit Vec(Ptr<T>&& data)
        : data_(std::move(data)) {

    }

private:
    Data data_;

    friend class VecFactory;
};
