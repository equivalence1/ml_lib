#pragma once

#include "object.h"
#include <cstdint>

class Vec  {
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

protected:
    explicit Vec(ObjectPtr<Object> data)
    : data_(data) {

    }

    Object* data() {
        return data_.get();
    }

    const Object* data() const {
        return data_.get();
    }
private:
    ObjectPtr<Object> data_;
    friend class VecFactory;
};
