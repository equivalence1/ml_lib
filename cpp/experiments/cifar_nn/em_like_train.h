#pragma once

#include "model.h"
#include "tensor_pair_dataset.h"
#include "loss.h"
#include "optimizer.h"
#include "initializer.h"

#include <torch/torch.h>

#include <vector>

template <typename TransformType>
class EMLikeTrainer {
public:
    virtual void train(TensorPairDataset& ds, const LossPtr& loss) {
        initializer_->init(ds, loss, &representationsModel, &decisionModel);

        for (uint32_t i = 0; i < iterations_; ++i) {
            std::cout << "EM iteration: " << i << std::endl;

            for (auto& param : representationsModel->parameters()) {
                param.set_requires_grad(false);
            }
            for (auto& param : decisionModel->parameters()) {
                param.set_requires_grad(true);
            }

            std::cout << "    getting representations" << std::endl;

            auto mds = ds.map(reprTransform_);
            auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(100));
            auto device = representationsModel->parameters().data()->device();
            std::vector<torch::Tensor> reprList;
            std::vector<torch::Tensor> targetsList;

            for (auto& batch : *dloader) {
                auto res = representationsModel->forward(batch.data.to(device));
                auto target = batch.target.to(device);
                reprList.push_back(res);
                targetsList.push_back(target);
            }

            auto repr = torch::cat(reprList, 0);
            auto targets = torch::cat(targetsList, 0);

            std::cout << "    optimizing decision model" << std::endl;

            auto decisionFuncOptimizer = getDecisionOptimizer(decisionModel);
            decisionFuncOptimizer->train(repr, targets, loss, decisionModel);

            for (auto& param : representationsModel->parameters()) {
                param.set_requires_grad(true);
            }
            for (auto& param : decisionModel->parameters()) {
                param.set_requires_grad(false);
            }

            std::cout << "    optimizing representation model" << std::endl;

            LossPtr representationLoss = makeRepresentationLoss(decisionModel, loss);

            auto representationOptimizer = getReprOptimizer(representationsModel);
            representationOptimizer->train(ds, representationLoss, representationsModel);

            fireListeners(i);
        }
    }

    virtual experiments::ModelPtr getTrainedModel(TensorPairDataset& ds, const LossPtr& loss) {
        train(ds, loss);
        return std::make_shared<CompositionalModel>(representationsModel, decisionModel);
    }

    virtual LossPtr makeRepresentationLoss(experiments::ModelPtr model, LossPtr loss) const {
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

        return std::make_shared<ReprLoss>(model, loss);
    }

    using IterationListener = std::function<void(uint32_t, experiments::ModelPtr)>;

    virtual void registerGlobalIterationListener(IterationListener listener) {
        listeners_.push_back(std::move(listener));
    }

protected:
    EMLikeTrainer(TransformType reprTransform, uint32_t iterations)
            : reprTransform_(reprTransform)
            , iterations_(iterations) {

    }

    virtual experiments::OptimizerPtr getReprOptimizer(const experiments::ModelPtr& reprModel) {
        return {nullptr};
    }

    virtual experiments::OptimizerPtr getDecisionOptimizer(const experiments::ModelPtr& decisionModel) {
        return {nullptr};
    }

private:
    void fireListeners(uint32_t iteration) {
        std::cout << std::endl;

        auto model = std::make_shared<CompositionalModel>(representationsModel, decisionModel);
        model->eval();
        for (auto& listener : listeners_) {
            listener(iteration, model);
        }
        model->train();

        std::cout << std::endl;
    }

protected:
    experiments::ModelPtr representationsModel;
    experiments::ModelPtr decisionModel;

    TransformType reprTransform_;

    InitializerPtr initializer_;

    uint32_t iterations_;

    std::vector<IterationListener> listeners_;
};
