#include <core/trans/offset_trans.h>

#include <core/vec_tools/fill.h>
#include <core/vec_tools/transform.h>
#include <core/vec_factory.h>
#include <core/trans/const_trans.h>

#include <memory>

int64_t OffsetTrans::xdim() const {
    return b_.dim();
}

int64_t OffsetTrans::ydim() const {
    return b_.dim();
}

const Trans& OffsetTrans::trans(const Vec &x, Vec &to) const {
    VecTools::copyTo(x, to);
    VecTools::subtract(to, b_);
    return *this;
}

ObjectPtr<Trans> OffsetTrans::gradient() const {
    Vec x = VecFactory::create(VecType::Cpu, xdim());
    VecTools::fill(1, x);
    return std::make_shared<ConstTrans>(x);
}
