#pragma once

#include "identity.h"
#include <core/trans.h>
#include <core/vec.h>

class AddVecTrans: public MapC1Stub<AddVecTrans> {
public:
    AddVecTrans(const Vec& b)
    : MapC1Stub<AddVecTrans>(b.dim())
    , b_(b) {

    }

    Vec trans(const Vec& x, Vec to) const final;

    Trans gradient() const final;

    Vec gradientRowTo(const Vec&, Vec to, int64_t) const final {
        assert(false);
        return to;
    }

private:
    Vec b_;
};
