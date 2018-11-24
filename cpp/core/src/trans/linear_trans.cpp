#include <core/trans/linear_trans.h>

#include <core/vec_tools/fill.h>
#include <core/vec_tools/transform.h>

#include <cassert>

int64_t LinearTrans::xdim() const {
    return param_.dim();
}

int64_t LinearTrans::ydim() const {
    return param_.dim();
}

const Trans& LinearTrans::trans(const Vec& x, Vec& to) const {
    VecTools::copyTo(x, to);
    VecTools::mul(to, param_);
    return *this;
}

ObjectPtr<Trans> LinearTrans::gradient() const {
    assert(false);
}
