#pragma once

#include "model.h"
#include "tensor_pair_dataset.h"
#include "loss.h"
#include "optimizer.h"
#include "initializer.h"

#include <torch/torch.h>

class EMLikeTrainer {
public:
    EMLikeTrainer(OptimizerPtr representationOptimizer,
                  OptimizerPtr decisionFuncOptimizer,
                  InitializerPtr initializer
                  )
        : representationOptimizer_(std::move(representationOptimizer))
        , decisionFuncOptimizer_(std::move(decisionFuncOptimizer))
        , initializer_(std::move(initializer)) {

    }

    void train(TensorPairDataset& ds, LossPtr& loss) {
        initializer_->init(ds, loss, &representationsModel, &decisionModel);

        for (uint32_t i = 0; i < iterations_; ++i) {
            std::cout << "EM iteration: " << i << std::endl;

            for (auto& param : representationsModel->parameters()) {
                param.set_requires_grad(false);
            }
            for (auto& param : decisionModel->parameters()) {
                param.set_requires_grad(true);
            }

            torch::Tensor lastLayer = representationsModel->forward(ds.data());

            auto targets = ds.targets();
            decisionFuncOptimizer_->train(lastLayer, targets, loss, decisionModel);

            for (auto& param : representationsModel->parameters()) {
                param.set_requires_grad(true);
            }
            for (auto& param : decisionModel->parameters()) {
                param.set_requires_grad(false);
            }

            LossPtr representationLoss = makeRepresentationLoss(decisionModel, loss);
            representationOptimizer_->train(ds, representationLoss, representationsModel);
        }
    }

    ModelPtr getTrainedModel(TensorPairDataset& ds, LossPtr& loss) {
        train(ds, loss);
        return std::make_shared<CompositionalModel>(representationsModel, decisionModel);
    }

    virtual LossPtr makeRepresentationLoss(ModelPtr trans, LossPtr loss) const = 0;

protected:
    EMLikeTrainer() = default;

    OptimizerPtr representationOptimizer_;
    OptimizerPtr decisionFuncOptimizer_;

    ModelPtr representationsModel;
    ModelPtr decisionModel;

    InitializerPtr initializer_;

    uint32_t iterations_ = 5;
};
