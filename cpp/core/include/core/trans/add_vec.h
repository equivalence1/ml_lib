#pragma once

#include "identity.h"
#include <core/trans.h>
#include <core/vec.h>

class AddVecTrans: public MapC1Stub<AddVecTrans> {
public:
    AddVecTrans(VecRef b)
    : MapC1Stub<AddVecTrans>(b.dim())
    , b_(b) {

    }

    VecRef trans(ConstVecRef x, VecRef to) const;

    IdentityMap gradient() const;

private:
    ConstVec b_;
};
