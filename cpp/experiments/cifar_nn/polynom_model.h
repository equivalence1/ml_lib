#pragma once

#include "model.h"
#include <models/polynom/polynom_gpu.h>
#include <models/polynom/polynom_autograd.h>

class PolynomModel : public experiments::Model {
public:

    explicit PolynomModel(PolynomPtr polynom)
        : polynom_(polynom) {}

    PolynomModel() {

    }

    torch::Tensor forward(torch::Tensor x) override;

    void reset(PolynomPtr polynom) {
        polynom_ = polynom;
        polynomCuda_ = nullptr;
    }

    void setLambda(double lambda) {
        polynom_->Lambda_ = lambda;
    }
private:
    PolynomPtr polynom_;
    PolynomCudaPtr polynomCuda_;

};
