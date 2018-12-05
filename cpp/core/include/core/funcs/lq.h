#pragma once

#include <core/vec.h>
#include <core/func.h>

#include <functional>

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
            ConstVecRef b)
            : TransStub<LqGrad>(b.dim(), b.dim())
              , q_(p)
              , b_(b) {

        }

        VecRef trans(ConstVecRef x, VecRef to) const;

    private:
        double q_;
        Vec b_;
    };

public:
    Lq(const Lq& other) = default;

    Lq(double p,
       ConstVecRef b)
        : FuncC1Stub<Lq>(b.dim())
          , q_(p)
          , b_(b) {

    }

    DoubleRef valueTo(ConstVecRef x, DoubleRef to) const;

    Trans gradient() const {
        return LqGrad(q_, b_);
    }
private:
    double q_;
    Vec b_;
};
