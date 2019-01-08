#include <core/funcs/linear.h>

#include <core/trans/fill.h>

#include <memory>
#include <cassert>


Trans Linear::gradient() const {
    return FillVec(param_, param_.dim());
}

DoubleRef Linear::valueTo(const Vec& x, DoubleRef to) const {
    assert(dim() == x.dim());
    to = bias_;

    for (int64_t i = 0; i < x.dim(); ++i) {
        to += x(i) * param_(i);
    }
    return to;
}
