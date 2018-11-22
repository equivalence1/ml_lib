#include <core/vec_tools/ops.h>

#include <cassert>

double VecTools::dotProduct(const Vec& left, const Vec& right) {
    assert(left.dim() == right.dim());
    double val = 0;

    for (int64_t i = 0; i < left.dim(); ++i) {
        val += left(i) * right(i);
    }
    return val;
}
Vec& VecTools::fill(double alpha, Vec& x) {
    for (int64_t i = 0; i < x.dim(); ++i) {
        x.set(i, alpha);
    }
    return x;
}

Vec& VecTools::makeSequence(double from, double step, Vec& x) {
    double cursor = from;
    for (int64_t i = 0; i < x.dim(); ++i) {
        x.set(i, cursor);
        cursor += step;
    }
    return x;
}
