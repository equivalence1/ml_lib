#pragma once

#include <torch/torch.h>
#include <memory>

class Loss {
public:
    virtual ~Loss() = default;

    virtual torch::Tensor value(const torch::Tensor& outputs, const torch::Tensor& targets) const = 0;

//    virtual void gradients(const Model& model, const torch::Tensor& ds, const torch::Tensor& targets) const {
//        auto outputs = model.apply(ds);
//        auto inputGradients = innerModel_.inputGradinets(outputs, targets, innerLoss_);
//        auto loss = 0.5 * (outputs - inputGradients) ^ 2;
//        loss.backward();
//    }

};

using LossPtr = std::shared_ptr<Loss>;

class ZeroLoss final : public Loss {
public:
    ZeroLoss() = default;

    torch::Tensor value(const torch::Tensor &outputs, const torch::Tensor &targets) const override;
};
