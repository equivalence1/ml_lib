#pragma once

#include <data/dataset.h>
#include <core/vec.h>
#include <core/func.h>
#include <util/array_ref.h>
#include <core/buffer.h>
#include <vec_tools/stats.h>

class Accuracy : public Stub<Func, Accuracy> {
public:

    Accuracy(const Vec& target, double decisionBorder = 0)
    : Stub<Func, Accuracy>(target.dim())
    , target_(target)
    , decisionBorder_(decisionBorder) {

    }

    Accuracy(const Vec& target, double targetBorder, double decisionBorder)
    : Stub<Func, Accuracy>(target.dim())
    , target_(target > targetBorder)
    , decisionBorder_(decisionBorder) {

    }

    DoubleRef valueTo(const Vec& x, DoubleRef to) const {
        auto bintarget = target_.arrayRef();
        auto ref = x.arrayRef();

        double s = 0;
        for (int64_t i = 0; i < ref.size(); ++i) {
           s += bintarget[i];
        }
        to = VecTools::sum(eq((x > decisionBorder_), target_)) / x.dim();
        return to;
    }

private:
    Vec target_;
    double decisionBorder_;
};


