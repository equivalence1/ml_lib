#pragma once

#include <math/vec_data.h>
#include <cstdint>

class Vec : public DataHolder<VecDataImpl> {
public:
    Vec(Vec& other) = default;
    Vec(const Vec& other) = default;
    Vec(Vec&& other) = default;

    void set(int64_t index, double value);

    double get(int64_t index) const

    int64_t dim() const;

    Vec slice(int64_t from, int64_t to);
    Vec slice(int64_t from, int64_t to) const;

protected:
    Vec(VecImplPtr data)
    : DataHolder(data) {

    }

    friend class VecFactory;
};
