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

    void train(const TensorPairDataset& ds, const Loss& loss) const {
        ModelPtr representations;
        ModelPtr decisionTrans;
        initializer_->init(ds, loss, &representations, &decisionTrans);

        for (uint32_t i = 0; i < iterations_; ++i) {
            torch::Tensor lastLayer = representations->forward(ds.data());
            decisionFuncOptimizer_->train(lastLayer, ds.targets(), loss, decisionTrans);

            LossPtr representationLoss = makeRepresentationLoss(decisionTrans, loss);
            representationOptimizer_->train(ds, *representationLoss, representations);
        }
    }

    virtual LossPtr makeRepresentationLoss(ModelPtr trans, const Loss& loss) const = 0;

private:
    OptimizerPtr representationOptimizer_;
    OptimizerPtr decisionFuncOptimizer_;
    InitializerPtr initializer_;
    uint32_t iterations_ = 10;
};
