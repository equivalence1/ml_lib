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
        auto r1 = torch::mm(inputs[0], w_);
        auto r2 = torch::mm(x_.t(), inputs[0]).t();
        auto r3 = inputs[0];
        return {r1, r2, r3};
    }

private:
    torch::Tensor x_;
    torch::Tensor w_;
};

namespace E = Eigen;

class LinearFunction : public torch::autograd::Function {
public:
    LinearFunction() = default;

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override {
        torch::autograd::Variable x = inputs[0];
        torch::autograd::Variable w = inputs[1];
        torch::autograd::Variable b = inputs[2];

        E::Map<E::Matrix<float, E::Dynamic, E::Dynamic, E::RowMajor>> x_((float*)x.data_ptr(), x.size(0), x.size(1));
        E::Map<E::Matrix<float, E::Dynamic, E::Dynamic, E::RowMajor>> w_((float*)w.data_ptr(), w.size(0), w.size(1));
        E::Map<E::RowVectorXf, E::RowMajor> b_((float*)b.data_ptr(), 1, b.size(0));

        // x_ should be transposed, as it has shape [batch_size, n_features_in]
        // the result is transposed to match shape [batch_size, n_features_out]
        auto resultEigen = (w_ * x_.transpose()).transpose();
        torch::autograd::Variable resultTorch = torch::zeros({resultEigen.rows(), resultEigen.cols()}, torch::kFloat32);

        E::Map<E::Matrix<float, E::Dynamic, E::Dynamic, E::RowMajor>> resultMap((float*)resultTorch.data_ptr(), resultEigen.rows(), resultEigen.cols());
        resultMap = resultEigen;
        resultMap.rowwise() += b_;

        auto grad_fn = std::make_shared<LinearFunctionBackward>(x, w, torch::autograd::collect_next_edges(inputs));
        torch::autograd::create_gradient_edge(resultTorch, grad_fn);

        return {resultTorch};
    }
};
