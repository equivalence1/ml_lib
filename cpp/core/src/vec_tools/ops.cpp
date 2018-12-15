#include "vec_impls.h"

#include "fill.cuh"

#include <core/vec_tools/fill.h>

#include <cmath>
#include <cassert>
#include <iostream>

using namespace Impl;

double VecTools::dotProduct(const Vec& left, const Vec& right) {
    assert(left.dim() == right.dim());
    double val = 0;

    for (int64_t i = 0; i < left.dim(); ++i) {
        val += left(i) * right(i);
    }
    return val;
}

Vec VecTools::fill(double alpha, Vec x) {
    return std::visit([&](auto&& impl) -> Vec {
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

Vec VecTools::makeSequence(double from, double step, Vec x) {
    double cursor = from;
    for (int64_t i = 0; i < x.dim(); ++i) {
        x.set(i, cursor);
        cursor += step;
    }
    return x;
}

Vec VecTools::subtract(Vec x, const Vec& y) {
    assert(x.dim() == y.dim());
    for (auto i = 0; i < x.dim(); i++) {
        x.set(i, x(i) - y(i));
    }
    return x;
}

Vec VecTools::pow(double p, const Vec& from, Vec to) {
    for (auto i = 0; i < from.dim(); i++) {
        to.set(i, std::pow(from(i), p));
    }
    return to;
}

Vec VecTools::mul(const Vec& x, Vec y) {
    assert(x.dim() == y.dim());
    for (auto i = 0; i < x.dim(); i++) {
        y.set(i, x(i) * y(i));
    }
    return y;
}

Vec VecTools::mul(double alpha, Vec x) {
    for (auto i = 0; i < x.dim(); i++) {
        x.set(i, alpha * x(i));
    }
    return x;
}
Vec VecTools::pow(double p, Vec x) {
    for (auto i = 0; i < x.dim(); i++) {
        x.set(i, std::pow(x(i), p));
    }
    return x;
}

template <class T>
T sgn(const T& val) {
    return val > 0 ? 1 : val < 0 ? -1 : 0;
}

Vec VecTools::sign(const Vec& x, Vec to) {
    assert(x.dim() == to.dim());

    for (int64_t i = 0; i < x.dim(); ++i) {
        to.set(i, sgn(x.get(i)));
    }
    return to;
}

Vec VecTools::abs(Vec x) {
    for (int64_t i = 0; i < x.dim(); ++i) {
        x.set(i, std::abs(x.get(i)));
    }
    return x;
}
