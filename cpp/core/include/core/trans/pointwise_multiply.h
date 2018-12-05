#pragma once

#include <core/trans.h>

class PointwiseMultiply: public MapStub<PointwiseMultiply> {
public:
    explicit PointwiseMultiply(Vec param)
    : MapStub<PointwiseMultiply>(param.dim())
    , param_(param) {

    }

    VecRef trans(ConstVecRef x, VecRef to) const;

private:
    Vec param_;
};
