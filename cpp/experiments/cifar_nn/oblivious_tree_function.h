#pragma once

#include <torch/torch.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/edge.h>

#include <core/vec.h>
#include <models/oblivious_tree.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include <iostream>
#include <cassert>
#include <vec_tools/transform.h>
#include <vec_tools/fill.h>


class ObliviousTreeFunctionBackward : public torch::autograd::Function {
public:
    ObliviousTreeFunctionBackward(torch::Tensor x, ModelPtr& tree,
                           torch::autograd::edge_list&& next_edges)
            : torch::autograd::Function(std::move(next_edges))
            , x_(std::move(x))
            , tree_(tree) {

    }

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override {
        auto sz = x_.sizes();
        torch::Tensor grads = torch::zeros({sz[0], sz[1]}, torch::kFloat32);
        auto backGrads = inputs[0];

        parallelFor(0, grads.sizes()[0], [&](int64_t i) {
            Vec elem = Vec(x_[i]);
            Vec grad_c(sz[1]);
            VecTools::fill(0, grad_c);
            tree_->grad(elem, grad_c);
            auto gradRef = grad_c.arrayRef();
            for (int j = 0; j < grad_c.size(); j++) {
                grads[i][j] = backGrads[i] * gradRef[j];
            }
        });
        return {grads};
    }

private:
    torch::Tensor x_;
    ModelPtr tree_;
};

class ObliviousTreeFunction : public torch::autograd::Function {
public:

    ObliviousTreeFunction(ModelPtr& tree)
        : tree_(tree){

    }

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override {
        torch::autograd::Variable x = inputs[0];

        auto sz = x.sizes();
        torch::autograd::Variable res = torch::zeros({sz[0]}, torch::kFloat32);

        parallelFor(0, sz[0], [&](int64_t i) {
            Vec elem = Vec(x[i]);
            double rs = tree_->value(elem);
            res[i] = rs;
        });

        auto grad_fn = std::make_shared<ObliviousTreeFunctionBackward>(x, tree_, torch::autograd::collect_next_edges(inputs));
        torch::autograd::create_gradient_edge(res, grad_fn);

        return {res};
    }

    ModelPtr tree_;
};
