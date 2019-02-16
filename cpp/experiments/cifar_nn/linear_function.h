#pragma once

#include <torch/torch.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/edge.h>

#include <eigen3/Eigen/Core>
#include <Eigen/LU>

#include <iostream>

class LinearFunctionBackward : public torch::autograd::Function {
public:
    LinearFunctionBackward(torch::Tensor x, torch::Tensor w,
                           torch::autograd::edge_list&& next_edges)
            : torch::autograd::Function(std::move(next_edges))
            , x_(std::move(x))
            , w_(std::move(w)) {

    }

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override {
        return {torch::mm(inputs[0], w_.t()), torch::mm(x_.t(), inputs[0])};
    }

private:
    torch::Tensor x_;
    torch::Tensor w_;
};

#include <cassert>
class LinearFunction : public torch::autograd::Function {
public:
    LinearFunction() = default;

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override {
        torch::autograd::Variable x = inputs[0];
        torch::autograd::Variable w = inputs[1];

        namespace E = Eigen;

        E::Map<E::Matrix<float, E::Dynamic, E::Dynamic, E::RowMajor>> x_((float *)x.data_ptr(), x.size(0), x.size(1));

        E::Map<E::Matrix<float, E::Dynamic, E::Dynamic, E::RowMajor>> w_((float *)w.data_ptr(), w.size(0), w.size(1));

        auto resultEigen = x_ * w_;
        auto resultRaw = new float[resultEigen.rows() * resultEigen.cols()];

        E::Map<E::Matrix<float, E::Dynamic, E::Dynamic, E::RowMajor>> resultMap(resultRaw, resultEigen.rows(), resultEigen.cols());
        resultMap = resultEigen;

        torch::autograd::Variable res;

        res = torch::from_blob(resultRaw,
                               {resultEigen.rows(), resultEigen.cols()},
                               {(long)(resultEigen.cols()), 1},
                               torch::kFloat32);

        auto grad_fn = std::make_shared<LinearFunctionBackward>(x, w, torch::autograd::collect_next_edges(inputs));
        torch::autograd::create_gradient_edge(res, grad_fn);

        return {res};
    }
};
