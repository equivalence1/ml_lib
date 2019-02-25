#pragma once

#include "em_like_train.h"
#include "simple_conv_net.h"
#include "linear_model.h"
#include "cross_entropy_loss.h"

class LinearTrainer : public EMLikeTrainer {
public:
    LinearTrainer(uint32_t it_global,
            uint32_t it_repr,
            uint32_t it_decision) : EMLikeTrainer() {
        representationsModel = std::make_shared<SimpleConvNet>();
        representationOptimizer_ = std::make_shared<DefaultSGDOptimizer>(it_repr);

        decisionModel = std::make_shared<LinearModel>(16 * 5 * 5, 10);
        decisionFuncOptimizer_ = std::make_shared<DefaultSGDOptimizer>(it_decision);

        initializer_ = std::make_shared<NoopInitializer>();
        iterations_ = it_global;
    }

    ModelPtr getTrainedModel(TensorPairDataset& ds) {
        LossPtr loss = std::make_shared<CrossEntropyLoss>();
        return EMLikeTrainer::getTrainedModel(ds, loss);
    }

    LossPtr makeRepresentationLoss(ModelPtr trans, LossPtr loss) const override {
        class ReprLoss : public Loss {
        public:
            ReprLoss(ModelPtr model, LossPtr loss)
                    : model_(std::move(model))
                    , loss_(std::move(loss)) {

            }

            torch::Tensor value(const torch::Tensor& outputs, const torch::Tensor& targets) const override {
                return loss_->value(model_->forward(outputs), targets);
            }

        private:
            ModelPtr model_;
            LossPtr loss_;
        };

        return std::make_shared<ReprLoss>(trans, loss);
    }

};
