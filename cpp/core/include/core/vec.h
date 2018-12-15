#pragma once

#include "object.h"

#include <cassert>
#include <cstdint>
#include <variant>
#include <memory>
#include <utility>
#include <functional>

class Vec {
public:
    explicit Vec(int64_t dim);

    Vec(Vec&& other) = default;
    Vec(Vec& other) = default;

    Vec(const Vec& other)
        : impl_(other.impl_)
        , immutable_(true) {
    }




    void set(int64_t index, double value);

    Vec slice(int64_t from, int64_t size);
    Vec slice(int64_t from, int64_t size) const;

    double get(int64_t index) const;

    double operator()(int64_t index) const {
        return get(index);
    }

    int64_t dim() const;

    const AnyVec* anyVec() const {
        return impl_.get();
    }

    AnyVec* anyVec() {
        assert(!immutable_);
        return impl_.get();
    }



protected:

    template <class T>
    explicit Vec(std::shared_ptr<T>&& ptr, bool immutable=true)
        : impl_(std::static_pointer_cast<AnyVec>(ptr))
        , immutable_(false) {

    }

    std::shared_ptr<AnyVec> impl_;
    bool immutable_ = true;

    friend class VecFactory;
};


