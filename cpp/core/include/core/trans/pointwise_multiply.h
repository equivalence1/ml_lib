#pragma once

#include <core/trans.h>

class PointwiseMultiply: public MapStub<PointwiseMultiply> {
public:
    explicit PointwiseMultiply(const Vec& param)
    : MapStub<PointwiseMultiply>(param.dim())
    , param_(param) {

    }

    Vec trans(const Vec& x, Vec to) const;

private:
    Vec param_;
};
