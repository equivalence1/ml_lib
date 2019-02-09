#pragma once
#include "model.h"
#include "tensor_pair_dataset.h"


class Loss {
public:
    ~Loss() {
    }

    virtual torch::Tensor value(const torch::Tensor& outputs, const torch::Tensor& targets) const = 0;

//    virtual void gradients(const Model& model, const torch::Tensor& ds, const torch::Tensor& targets) const {
//        auto outputs = model.apply(ds);
//        auto inputGradients = innerModel_.inputGradinets(outputs, targets, innerLoss_);
//        auto loss = 0.5 * (outputs - inputGradients) ^ 2;
//        loss.backward();
//    }

};

class Optimizer {
public:
    ~Optimizer() {

    }

    virtual void train(const TensorPairDataset& ds, const Loss& loss, Model* model) const = 0;
    virtual void train(const torch::Tensor& ds, const torch::Tensor& target, const Loss& loss, Model* model) const = 0;
};

using ModelPtr =  std::shared_ptr<Model>;

class Initializer {
public:

    virtual void init(const TensorPairDataset& ds,  const Loss& loss, std::shared_ptr<Model>* representation, std::shared_ptr<Model>* decisionFunc)  = 0;
};


class EMLikeTrainer {
public:

    EMLikeTrainer(std::shared_ptr<Optimizer> representationOptimizer,
                  std::shared_ptr<Optimizer> decisionFuncOptimizer,
                  std::shared_ptr<Initializer> initializer
                  )
        : representationOptimizer_(std::move(representationOptimizer))
        , decisionFuncOptimizer_(std::move(decisionFuncOptimizer))
        , initializer_(std::move(initializer)) {

    }


    void train(const TensorPairDataset& ds, const Loss& loss) const {

        std::shared_ptr<Model> representations;
        std::shared_ptr<Model> decisionTrans;
        initializer_->init(ds, loss, &representations, &decisionTrans);

        for (uint32_t i = 0; i < iterations_; ++i) {
            torch::Tensor lastLayer = representations->applyToDs(ds.data());
            decisionFuncOptimizer_->train(lastLayer, ds.targets(), loss, decisionTrans.get());

            std::shared_ptr<Loss> representationLoss = makeRepresentationLoss(decisionTrans, loss);
            representationOptimizer_->train(ds, *representationLoss, representations.get());
        }
    }

    virtual std::shared_ptr<Loss> makeRepresentationLoss(std::shared_ptr<Model> trans, const Loss& loss) const = 0;

private:
    std::shared_ptr<Optimizer> representationOptimizer_;
    std::shared_ptr<Optimizer> decisionFuncOptimizer_;
    std::shared_ptr<Initializer> initializer_;
    uint32_t iterations_ = 10;
};
