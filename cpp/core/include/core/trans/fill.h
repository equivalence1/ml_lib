#pragma once

#include <core/vec.h>
#include <core/trans.h>
#include <utility>

class FillConst : public TransC1Stub<FillConst> {
public:
    FillConst(
        double value,
        int64_t xdim,
        int64_t ydim)
        : TransC1Stub<FillConst>(xdim, ydim)
          , value_(value) {

    }

    VecRef trans(ConstVecRef, VecRef to) const;

    FillConst gradient() const;
private:
    double value_ = 0;
};

class FillVec : public TransC1Stub<FillVec> {
public:
    FillVec(
        ConstVecRef params,
        int64_t xdim)
        : TransC1Stub<FillVec>(xdim, params.dim())
          , params_(params) {

    }

    VecRef trans(ConstVecRef, VecRef to) const;

    FillConst gradient() const;

private:
    ConstVec params_;
};
