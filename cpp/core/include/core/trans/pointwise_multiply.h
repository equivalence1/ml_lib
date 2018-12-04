#pragma once

#include <core/trans.h>

class PointwiseMultiply: public MapStub<PointwiseMultiply> {
public:
    explicit PointwiseMultiply(ConstVec param)
    : MapStub<PointwiseMultiply>(param.dim())
    , param_(param) {

    }

    VecRef trans(ConstVecRef x, VecRef to) const;

private:
    ConstVec param_;
};
