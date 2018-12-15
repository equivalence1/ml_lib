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

    Vec trans(const Vec&, Vec to) const final;

    Trans gradient() const final;

    Vec gradientRowTo(const Vec&, Vec to, int64_t) const final {
        assert(false);
        return to;
    }
private:
    double value_ = 0;
};

class FillVec : public TransC1Stub<FillVec> {
public:
    FillVec(
        const Vec& params,
        int64_t xdim)
        : TransC1Stub<FillVec>(xdim, params.dim())
          , params_(params) {

    }

    Vec trans(const Vec&, Vec to) const final;

    Trans gradient() const final;

    Vec gradientRowTo(const Vec&, Vec to, int64_t) const final {
        assert(false);
        return to;
    }

private:
    Vec params_;
};
