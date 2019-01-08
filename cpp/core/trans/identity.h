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

    Vec trans(const Vec& x, Vec to) const;

    Vec gradientRowTo(const Vec&, Vec to, int64_t) const final;
};

