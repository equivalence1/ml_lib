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

    FillConst gradient() const;
};

