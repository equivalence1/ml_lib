#pragma once

#include "em_like_train.h"
#include "simple_conv_net.h"
#include "linear_model.h"
#include "bce_sigm.h"
#include "oblivious_tree_model.h"

#include <data/grid_builder.h>
#include <models/oblivious_tree.h>
#include <methods/boosting.h>
#include <methods/greedy_oblivious_tree.h>
#include <methods/boosting_weak_target_factory.h>
#include <targets/cross_entropy.h>
#include <metrics/accuracy.h>


class ObliviousTreeTrainer : public EMLikeTrainer<torch::data::transforms::Stack<>> {
public:
    ObliviousTreeTrainer() : EMLikeTrainer(torch::data::transforms::Stack<>(), 0) {
        representationsModel_ = std::make_shared<SimpleConvNet>();

        torch::optim::SGDOptions reprOptimOptions(0.0002);

        // TODO optimizer

//        representationOptimizer_ = std::make_shared<DefaultSGDOptimizer>(6, reprOptimOptions);
    
        initializer_ = std::make_shared<NoopInitializer>();
    }

    void train(TensorPairDataset& ds, const LossPtr& loss) override {
        initializer_->init(ds, loss, &representationsModel_, &decisionModel_);

        for (auto& param : representationsModel_->parameters()) {
            param = torch::randn(param.sizes());
        }
        iterations_ = 20;
        for (uint32_t i = 0; i < iterations_ + 1; ++i) {
            std::cout << "EM iteration: " << i << std::endl;
            for (auto& param : representationsModel_->parameters()) {
                param.set_requires_grad(false);
            }
            torch::Tensor lastLayer = representationsModel_->forward(ds.data());
            int64_t lsSz = lastLayer.sizes()[0];
            Vec nwDs = Vec(lastLayer.view({-1}));
            Mx dt(nwDs, lastLayer.sizes()[0], lastLayer.sizes()[1]);

            DataSet dsLst(dt, Vec(torch::_cast_Float(ds.targets())));

            assert(dsLst.sample(lsSz - 1).size() != 0);

            BinarizationConfig config;
            config.bordersCount_ = 32;//32;
            auto grid = buildGrid(dsLst, config);
            BoostingConfig boostingConfig;
            boostingConfig.iterations_ = 100;
            boostingConfig.step_ = 0.8;
            Boosting boosting(boostingConfig, std::make_unique<GradientBoostingWeakTargetFactory>(), std::make_unique<GreedyObliviousTree>(grid, 6));//6));


            auto metricsCalcer = std::make_shared<BoostingMetricsCalcer>(dsLst);
            metricsCalcer->addMetric(CrossEntropy(dsLst), "CrossEntropy");
            metricsCalcer->addMetric(Accuracy(dsLst.target(), 0.1, 0), "Accuracy");
            boosting.addListener(metricsCalcer);
            CrossEntropy target(dsLst, 0.1);
            decisionModel_ = std::make_shared<ObliviousTreeModel>(boosting.fit(dsLst, target));

            if (i == iterations_) break;

            for (auto& param : representationsModel_->parameters()) {
                param.set_requires_grad(true);
            }

            LossPtr representationLoss = makeRepresentationLoss(decisionModel_, loss);

            // TODO train

//            representationOptimizer_->train_adam(ds, representationLoss, representationsModel_);
        }
    }

    experiments::ModelPtr getTrainedModel(TensorPairDataset& ds) {
        LossPtr loss = std::make_shared<BCESigmoidLoss>();
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
