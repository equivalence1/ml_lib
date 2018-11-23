#include <core/funcs/linear.h>

#include <core/trans/const_trans.h>

#include <memory>
#include <cassert>

int64_t Linear::xdim() const {
    assert(param_.dim());
    return param_.dim() - 1;
}

double Linear::value(const Vec& x) const {
    assert(xdim() == x.dim());
    double result = param_(0);

    for (int64_t i = 0; i < x.dim(); ++i) {
        result += x(i) * param_(i + 1);
    }
    return result;
}

ObjectPtr<Trans> Linear::gradient() const {
    return std::make_shared<ConstTrans>(ConstTrans(param_));
}
