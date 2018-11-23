#include <core/vec_tools/fill.h>
#include <vec_impls.h>
#include <fill.cuh>

#include <cmath>
#include <cassert>
#include <iostream>

namespace {

    struct FillVec {

        FillVec(Vec& vec, double val)
        : vec_(vec)
        , val_(val) {

        }

        template <class Impl>
        void operator()(Impl&) {
            for (int64_t i = 0; i < vec_.dim(); ++i) {
                vec_.set(i, val_);
            }
        }

        #if defined(CUDA)
        void operator()(CudaVecPtr& impl) {
            //don't do this in production, just for test
            const float val = val_;
            Cuda::Kernel::FillBuffer<float>(impl->data(), val, impl->dim(), 0);
        }
        #endif

        Vec& vec_;
        double val_;
    };
}

double VecTools::dotProduct(const Vec& left, const Vec& right) {
    assert(left.dim() == right.dim());
    double val = 0;

    for (int64_t i = 0; i < left.dim(); ++i) {
        val += left(i) * right(i);
    }
    return val;
}


Vec& VecTools::fill(double alpha, Vec& x) {
    std::visit(FillVec(x, alpha), x.data());
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

Vec& VecTools::subtract(Vec& x, const Vec& y) {
    assert(x.dim() == y.dim());
    for (auto i = 0; i < x.dim(); i++) {
        x.set(i, x(i) - y(i));
    }
    return x;
}

Vec& VecTools::exp(double p, const Vec& from, Vec& to) {
    for (auto i = 0; i < from.dim(); i++) {
        to.set(i, std::pow(from(i), p));
    }
    return to;
}

Vec& VecTools::mul(Vec& x, const Vec& y) {
    for (auto i = 0; i < x.dim(); i++) {
        x.set(i, x(i) * y(i));
    }
    return x;
}
