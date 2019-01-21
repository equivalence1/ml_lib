#pragma once

#include <core/vec.h>
#include <core/func.h>

#include <functional>
#include <core/vec_factory.h>

/**
 * || x - b ||_p^p
 *
 * grad: p  || x - b||^(p -1)
 */




class Lq : public FuncC1Stub<Lq> {
public:

    class LqGrad : public TransStub<LqGrad> {
    public:
        LqGrad(
            double p,
            const Vec& b)
            : TransStub<LqGrad>(b.dim(), b.dim())
              , q_(p)
              , b_(b) {

        }

        Vec trans(const Vec& x, Vec to) const;

    private:
        double q_;
        Vec b_;
    };

public:
    Lq(const Lq& other) = default;

    Lq(
        double p,
        const Vec& b)
        : FuncC1Stub<Lq>(b.dim())
          , q_(p)
          , b_(b) {

    }

    DoubleRef valueTo(const Vec& x, DoubleRef to) const;

    std::unique_ptr<Trans> gradient() const {
        return LqGrad(q_, b_);
    }
private:
    double q_;
    Vec b_;
};
