#include <core/trans/pow.h>
#include <core/vec_tools/transform.h>

#include <cassert>



// alpha x^k
Vec Pow::trans(const Vec& x, Vec to) const {
    VecTools::pow(k_, x, to);
    to *= alpha_;
    return to;
}

Trans Pow::gradient() const {
    return Pow(k_ - 1, xdim(), alpha_ * k_);
}

