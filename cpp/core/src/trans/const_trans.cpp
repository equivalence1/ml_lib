#include <core/trans/const_trans.h>

#include <core/vec_factory.h>

#include <memory>
#include <cassert>

int64_t ConstTrans::xdim() const {
    return params_.dim();
}

int64_t ConstTrans::ydim() const {
    return params_.dim();
}

const Trans& ConstTrans::trans(const Vec& x, Vec& to) const {
    to = params_;
    return *this;
}

ObjectPtr<Trans> ConstTrans::gradient() const {
    // TODO
    assert(false);
}
