#pragma once

#include "polynom.h"
#include "polynom_gpu.h"
#include <torch/torch.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/edge.h>

#include <cassert>
#include <iostream>
#include <util/parallel_executor.h>


class PolynomBackward : public torch::autograd::Function {
public:
    PolynomBackward(torch::Tensor samplesBatch,
                    PolynomPtr polynom,
                    torch::autograd::edge_list&& nextEdges)
            : torch::autograd::Function(std::move(nextEdges))
            , samplesBatch_(std::move(samplesBatch))
            , polynom_(polynom) {

    }

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override;

private:
    torch::Tensor samplesBatch_;
    PolynomPtr polynom_;
};

class PolynomForward : public torch::autograd::Function {
public:

    PolynomForward(PolynomPtr polynom)
        : polynom_(std::move(polynom)){

    }

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override;
private:

    PolynomPtr polynom_;
};


class PolynomBackwardCuda : public torch::autograd::Function {
public:
    PolynomBackwardCuda(torch::Tensor samplesBatch,
                    PolynomCudaPtr polynom,
                    torch::autograd::edge_list&& nextEdges)
        : torch::autograd::Function(std::move(nextEdges))
          , samplesBatch_(std::move(samplesBatch))
          , polynom_(polynom) {

    }

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override;

private:
    torch::Tensor samplesBatch_;
    PolynomCudaPtr polynom_;
};

class PolynomForwardCuda : public torch::autograd::Function {
public:

    explicit PolynomForwardCuda(PolynomCudaPtr polynom)
        : polynom_(std::move(polynom)){

    }

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override;
private:

    PolynomCudaPtr polynom_;
};


