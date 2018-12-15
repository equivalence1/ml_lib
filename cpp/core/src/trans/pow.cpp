#include <core/trans/pow.h>
#include <core/vec_tools/fill.h>

#include <cassert>



// alpha x^k
Vec Pow::trans(const Vec& x, Vec to) const {
    VecTools::mul(alpha_, VecTools::pow(k_, x, to));
    return to;
}

Trans Pow::gradient() const {
    return Pow(k_ -1, xdim(), alpha_ * k_);
}

