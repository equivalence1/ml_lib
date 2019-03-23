#pragma once

#include "em_like_train.h"
#include "simple_conv_net.h"
#include "linear_model.h"
#include "cross_entropy_loss.h"
#include "oblivious_tree_model.h"

#include <data/grid_builder.h>
#include <models/oblivious_tree.h>
#include <models/ensemble.h>
#include <methods/boosting.h>
#include <methods/greedy_oblivious_tree.h>
#include <methods/boosting_weak_target_factory.h>
#include <targets/cross_entropy.h>
#include <metrics/accuracy.h>


class TreeTrainer : public EMLikeTrainer {
public:
    TreeTrainer() : EMLikeTrainer() {
    }

    void train(TensorPairDataset& ds, LossPtr& loss) override {
            Vec nwDs = Vec(ds.data().reshape({10000*3072}));
            std::cout << ds.data().sizes() << std::endl;
            Mx dt(nwDs, ds.data().sizes()[0], ds.data().sizes()[1]);
            std::cout << dt.xdim() << " " << dt.ydim() << std::endl;
            DataSet dsLst(dt, Vec(torch::_cast_Float(ds.targets())));
            BinarizationConfig config;
            config.bordersCount_ = 32;
            auto grid = buildGrid(dsLst, config);
            BoostingConfig boostingConfig;
            boostingConfig.step_=0.8;
            boostingConfig.iterations_=100;
            Boosting boosting(boostingConfig, std::make_unique<GradientBoostingWeakTargetFactory>(), std::make_unique<GreedyObliviousTree>(grid, 6));
            auto metricsCalcer = std::make_shared<BoostingMetricsCalcer>(dsLst);
            metricsCalcer->addMetric(CrossEntropy(dsLst), "CrossEntropy");
            metricsCalcer->addMetric(Accuracy(dsLst.target(), 0.1, 0), "Accuracy");
            boosting.addListener(metricsCalcer);
            CrossEntropy target(dsLst, 0.1);
            decisionModel = std::make_shared<ObliviousTreeModel>(boosting.fit(dsLst, target));
//        }
    }

    experiments::ModelPtr getTrainedModel(TensorPairDataset& ds) {
        LossPtr loss = std::make_shared<CrossEntropyLoss>();
        train(ds, loss);
        auto res = decisionModel;
        return res;
    }

    LossPtr makeRepresentationLoss(experiments::ModelPtr trans, LossPtr loss) const override {
        class ReprLoss : public Loss {
        public:
            ReprLoss(experiments::ModelPtr model, LossPtr loss)
                    : model_(std::move(model))
                    , loss_(std::move(loss)) {

            }

            torch::Tensor value(const torch::Tensor& outputs, const torch::Tensor& targets) const override {
                auto res_mod = model_->forward(outputs);
                auto rs = loss_->value(res_mod, targets);
                return rs;
            }

        private:
            experiments::ModelPtr model_;
            LossPtr loss_;
        };

        return std::make_shared<ReprLoss>(trans, loss);
    }

};
