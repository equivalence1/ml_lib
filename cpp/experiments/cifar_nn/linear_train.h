#pragma once

#include "em_like_train.h"
#include "lenet.h"
#include "linear_model.h"
#include "cross_entropy_loss.h"

class LeNetLinearTrainer : public EMLikeTrainer {
public:
    LeNetLinearTrainer(uint32_t it_global,
            uint32_t it_repr,
            uint32_t it_decision) : EMLikeTrainer() {
        representationsModel = std::make_shared<LeNetConv>();
        torch::optim::SGDOptions reprOptimOptions(0.001);
        representationOptimizer_ = std::make_shared<DefaultSGDOptimizer>(it_repr, reprOptimOptions);

        decisionModel = std::make_shared<LinearModel>(16 * 5 * 5, 10);

        torch::optim::SGDOptions decisionOptimOptions(0.1);
        decisionFuncOptimizer_ = std::make_shared<DefaultSGDOptimizer>(it_decision, decisionOptimOptions);

        initializer_ = std::make_shared<NoopInitializer>();
        iterations_ = it_global;
    }

    experiments::ModelPtr getTrainedModel(TensorPairDataset& ds) {
        LossPtr loss = std::make_shared<CrossEntropyLoss>();
        return EMLikeTrainer::getTrainedModel(ds, loss);
    }

    LossPtr makeRepresentationLoss(experiments::ModelPtr trans, LossPtr loss) const override {
        class ReprLoss : public Loss {
        public:
            ReprLoss(experiments::ModelPtr model, LossPtr loss)
                    : model_(std::move(model))
                    , loss_(std::move(loss)) {

            }

            torch::Tensor value(const torch::Tensor& outputs, const torch::Tensor& targets) const override {
                return loss_->value(model_->forward(outputs), targets);
            }

        private:
            experiments::ModelPtr model_;
            LossPtr loss_;
        };

        return std::make_shared<ReprLoss>(trans, loss);
    }

};
