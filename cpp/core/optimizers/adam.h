#pragma once

#include <core/batch_optimizer.h>

class Adam: public BatchOptimizer {
public :
    Adam(int64_t vecSize, double learningRate=0.001, double betta1=0.9, double betta2=0.999, double eps=0.00000001);
    Vec optimize(Batch<FuncC1> f, Vec cursor) const override;
    void reset();
private:
    double betta1;
    double betta2;
    mutable double betta1Pow;
    mutable double betta2Pow;
    double learningRate;
    double eps;
    Vec vD;
    Vec sD;
};