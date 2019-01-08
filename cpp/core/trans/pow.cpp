#include "pow.h"
#include <core/vec_tools/transform.h>

#include <core/vec_tools/fill.h>

// alpha x^k
Vec Pow::trans(const Vec& x, Vec to) const {
    VecTools::pow(k_, x, to);
    to *= alpha_;
    return to;
}


Vec Pow::gradientRowTo(const Vec& x, Vec to, int64_t row) const {
    VecTools::fill(0, to);
    auto diagElem = to.slice(row, 1);
    VecTools::pow(k_ - 1, x, diagElem);
    diagElem *= alpha_ * k_;
    return to;
}

