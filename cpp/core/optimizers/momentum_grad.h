#pragma once

#include <core/batch_optimizer.h>

class MomentumGrad: public BatchOptimizer {
public :
    MomentumGrad(int64_t vecSize, double learningRate=0.001, double momentum=0.9);
    Vec optimize(Batch<FuncC1> f, Vec cursor) const override;
    void reset();
private:
    double learningRate;
    double momentum;
    mutable Vec gradBuf;
};