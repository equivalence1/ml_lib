#include <core/trans/const_trans.h>

#include <core/vec_factory.h>

#include <memory>

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
    // TODO this actually should be a matrix
    auto size = xdim() * ydim();
    auto v = VecFactory::create(VecType::Cpu, size);
    return std::make_shared<ConstTrans>(ConstTrans(v));
}
