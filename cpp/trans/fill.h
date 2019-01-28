#pragma once

#include <core/vec.h>
#include <core/trans.h>
#include <utility>
#include <core/vec_factory.h>

class FillConst : public Stub<TransC1, FillConst> {
public:
    FillConst(
        double value,
        int64_t xdim,
        int64_t ydim)
        : Stub<TransC1, FillConst>(xdim, ydim)
          , value_(value) {

    }

    Vec trans(const Vec&, Vec to) const final;

    std::unique_ptr<Trans> gradient() const final;

    Vec gradientRowTo(const Vec&, Vec to, int64_t) const final {
        assert(false);
        return to;
    }
private:
    double value_ = 0;
};

class FillVec : public Stub<TransC1, FillVec> {
public:
    FillVec(
        const Vec& params,
        int64_t xdim)
        : Stub<TransC1, FillVec>(xdim, params.dim())
          , params_(params) {

    }

    Vec trans(const Vec&, Vec to) const final;

    std::unique_ptr<Trans> gradient() const final;

    Vec gradientRowTo(const Vec&, Vec to, int64_t) const final;

private:
    Vec params_;
};
