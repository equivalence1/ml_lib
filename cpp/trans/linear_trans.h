#pragma once

#include <core/vec.h>
#include <core/trans.h>
#include <utility>
#include <core/vec_factory.h>

class LinearTrans : public Stub<TransC1, LinearTrans> {
public:
    LinearTrans(
        const Mx& mx)
        : Stub<TransC1, LinearTrans>(mx.xdim(), mx.ydim())
          , mx_(mx) {

    }

    Vec trans(const Vec&, Vec to) const final;

    std::unique_ptr<Trans> gradient() const final;

    Vec gradientRowTo(const Vec&, Vec to, int64_t) const final;

private:
    Mx mx_;
};
