#include <core/trans/pow.h>
#include <core/vec_tools/fill.h>

#include <cassert>



// alpha x^k
VecRef Pow::trans(ConstVecRef x, VecRef to) const {
    VecTools::mul(alpha_, VecTools::pow(k_, x, to));
    return to;
}

Pow Pow::gradient() const {
    return Pow(k_ -1, xdim(), alpha_ * k_);
}

