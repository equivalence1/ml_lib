#pragma once

#include "fill.h"
#include <core/vec.h>
#include <core/trans.h>
#include <utility>

class IdentityMap : public MapC1Stub<IdentityMap> {
public:
    IdentityMap(int64_t dim)
        : MapC1Stub<IdentityMap>(dim) {

    }

    VecRef trans(ConstVecRef x, VecRef to) const;

    Trans gradient() const;

    VecRef gradientRowTo(ConstVecRef, VecRef to, int64_t) const final {
        assert(false);
        return to;
    }
};

