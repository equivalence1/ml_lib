#pragma once

#include <core/vec.h>
#include <core/optimizer.h>

class GradientDescent: public Optimizer {
public:
    GradientDescent(double eps, uint64_t iter_lim):
            eps_(eps), iter_lim_(iter_lim) {

    }

    GradientDescent(double eps): GradientDescent(eps, 100000) {

    }

    GradientDescent(uint64_t iter_lim): GradientDescent(0.01, iter_lim) {

    }

    Vec optimize(const FuncC1& f, Vec cursor) const override;

private:
    double eps_;
    uint64_t iter_lim_;
};
