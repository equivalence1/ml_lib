#pragma once

#include "vec.h"
#include "vec_factory.h"
#include "func.h"
#include "batch.h"

class BatchOptimizer {
public:
    virtual Vec optimize(Batch<FuncC1> f, Vec x0) const = 0;

    virtual ~BatchOptimizer() = default;

};
