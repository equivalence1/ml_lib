#pragma once

#include <core/batch_optimizer.h>

class StochasticGrad : public BatchOptimizer {
public :
    StochasticGrad(int64_t vecSize, double learningRate = 0.001)
        :
        learningRate(learningRate) {

    }
    Vec optimize(Batch<FuncC1> f, Vec cursor) const override;
private:
    double learningRate;
};
