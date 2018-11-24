#pragma once

#include "vec.h"
#include "vec_factory.h"
#include "func.h"

class Optimizer {
    virtual Vec optimize(const Func& f, const Vec& x0) const = 0;

    virtual Vec optimize(const Func& f) const {
        return optimize(f, VecFactory::create(VecType::Cpu, f.xdim()));
    }
};
