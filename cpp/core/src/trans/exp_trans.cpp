#include <core/trans/exp_trans.h>

#include <core/vec_tools/ops.h>

#include <cassert>

int64_t ExpTrans::xdim() const {
    return dim_;
}

int64_t ExpTrans::ydim() const {
    return dim_;
}

const Trans& ExpTrans::trans(const Vec& x, Vec& to) const {
    VecTools::exp(exp_, x, to);
    return *this;
}

ObjectPtr<Trans> ExpTrans::gradient() const {
    assert(false);
}
