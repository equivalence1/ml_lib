#include <core/trans/compose_trans.h>

#include <core/vec_factory.h>

#include <memory>

int64_t ComposeTrans::xdim() const {
    return right_->xdim();
}

int64_t ComposeTrans::ydim() const {
    return left_->ydim();
}

const Trans& ComposeTrans::trans(const Vec& x, Vec& to) const {
    Vec tmp = VecFactory::create(VecType::Cpu, to.dim());
    right_->trans(x, tmp);
    left_->trans(tmp, to);
    return *this;
}

ObjectPtr<Trans> ComposeTrans::gradient() const {
    auto l_prime = std::make_shared<ComposeTrans>(left_->gradient(), right_);
    auto r_prime = right_->gradient();
    return std::make_shared<ComposeTrans>(l_prime, r_prime);
}
