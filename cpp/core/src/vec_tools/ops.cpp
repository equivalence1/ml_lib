#include "vec_impls.h"

#include "fill.cuh"

#include <core/vec_tools/fill.h>

#include <cmath>
#include <cassert>
#include <iostream>

using namespace Impl;

double VecTools::dotProduct(ConstVecRef left, ConstVecRef right) {
    assert(left.dim() == right.dim());
    double val = 0;

    for (int64_t i = 0; i < left.dim(); ++i) {
        val += left(i) * right(i);
    }
    return val;
}

VecRef VecTools::fill(double alpha, VecRef x) {
    return std::visit([&](auto&& impl) -> VecRef {
        using T = std::decay_t<decltype(impl)>;

        #if defined(CUDA)
        if constexpr (std::is_same_v<T, CudaVec*>) {
            const float val = alpha;
            Cuda::Kernel::FillBuffer<float>(impl->data(), val, impl->dim(), 0);
            return x;
        }
        #endif

        for (int64_t i = 0; i < x.dim(); ++i) {
            x.set(i, alpha);
        }
        return x;
    }, DynamicDispatch(x.anyVec()));
}

VecRef VecTools::makeSequence(double from, double step, VecRef x) {
    double cursor = from;
    for (int64_t i = 0; i < x.dim(); ++i) {
        x.set(i, cursor);
        cursor += step;
    }
    return x;
}

VecRef VecTools::subtract(VecRef x, ConstVecRef y) {
    assert(x.dim() == y.dim());
    for (auto i = 0; i < x.dim(); i++) {
        x.set(i, x(i) - y(i));
    }
    return x;
}

VecRef VecTools::pow(double p, ConstVecRef from, VecRef to) {
    for (auto i = 0; i < from.dim(); i++) {
        to.set(i, std::pow(from(i), p));
    }
    return to;
}

VecRef VecTools::mul(ConstVecRef x, VecRef y) {
    assert(x.dim() == y.dim());
    for (auto i = 0; i < x.dim(); i++) {
        y.set(i, x(i) * y(i));
    }
    return y;
}

VecRef VecTools::mul(double alpha, VecRef x) {
    for (auto i = 0; i < x.dim(); i++) {
        x.set(i, alpha * x(i));
    }
    return x;
}
VecRef VecTools::pow(double p, VecRef x) {
    for (auto i = 0; i < x.dim(); i++) {
        x.set(i, std::pow(x(i), p));
    }
    return x;
}

template <class T>
T sgn(const T& val) {
    return val > 0 ? 1 : val < 0 ? -1 : 0;
}

VecRef VecTools::sign(ConstVecRef x, VecRef to) {
    assert(x.dim() == to.dim());

    for (int64_t i = 0; i < x.dim(); ++i) {
        to.set(i, sgn(x.get(i)));
    }
    return to;
}

VecRef VecTools::abs(VecRef x) {
    for (int64_t i = 0; i < x.dim(); ++i) {
        x.set(i, std::abs(x.get(i)));
    }
    return x;
}
