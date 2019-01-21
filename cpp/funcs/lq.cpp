#include <funcs/lq.h>

#include <vec_tools/distance.h>
#include <vec_tools/fill.h>
#include <vec_tools/transform.h>

DoubleRef Lq::valueTo(const Vec& x, DoubleRef to) const {
    to = VecTools::distanceLq(q_, x, b_);
    return to;
}

Vec Lq::LqGrad::trans(const Vec& x, Vec to) const {
    // p * sign(x - b) * | x - b | ^ {p - 1}
    VecTools::copyTo(x, to);
    VecTools::subtract(to, b_);

    Vec sign(x.dim());
    VecTools::sign(to, sign);
    VecTools::abs(to);
    VecTools::pow(q_ - 1, to);
    VecTools::mul(q_, to);
    VecTools::mul(sign, to);
    return to;
}
