#pragma once

#include <core/trans.h>

class Pow : public MapC1Stub<Pow> {
public:
    Pow(double k,
        int64_t dim,
        double alpha = 1
        )
        : MapC1Stub<Pow>(dim)
        , k_(k)
        , alpha_(alpha) {

    }

    VecRef trans(ConstVecRef x, VecRef to) const;

    Pow gradient() const;

private:
    double k_;
    double alpha_;
};
