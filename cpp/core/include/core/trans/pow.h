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

    Vec trans(const Vec& x, Vec to) const final;

    Trans gradient() const final;

    Vec gradientRowTo(const Vec&, Vec to, int64_t) const final {
        assert(false);
        return to;
    }
private:
    double k_;
    double alpha_;
};
