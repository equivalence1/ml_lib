#include "common.h"

#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/em_like_train.h>
#include <experiments/core/params.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

// CommonEm

struct CommonEmOptions {
    CommonEmOptions(const std::vector<int>& counts) {
        globalIterationsCount = counts[0];
        representationsIterations = counts[1];
        decisionIterations = counts[2];
    }

    uint32_t globalIterationsCount = 0;
    uint32_t representationsIterations;
    uint32_t decisionIterations;
};

class CommonEm : public EMLikeTrainer<decltype(getDefaultCifar10TrainTransform())> {
public:
    // TODO a lot of bad code here. Left it here for compatibility reasons, need to revisit
    // (or maybe even remove and just use catboost_em)
    CommonEm(experiments::ConvModelPtr model,
            const json& params)
            : EMLikeTrainer(getDefaultCifar10TrainTransform(),
                    CommonEmOptions(params[NIterationsKey]).globalIterationsCount,
                    std::move(model))
            , opts_(params[NIterationsKey]) {
        convParams_[DeviceKey] = params[DeviceKey];
        convParams_[ModelCheckpointFileKey] =
                "conv_" + (std::string)params[ModelCheckpointFileKey];
        convParams_[BatchSizeKey] = params[BatchSizeKey];
        convParams_[ReportsPerEpochKey] = params[ReportsPerEpochKey];

        decisionParams_[DeviceKey] = params[DeviceKey];
        decisionParams_[ModelCheckpointFileKey] =
                "decision_" + (std::string)params[ModelCheckpointFileKey];
        decisionParams_[BatchSizeKey] = params[BatchSizeKey];
        decisionParams_[ReportsPerEpochKey] = params[ReportsPerEpochKey];
    }

protected:
    experiments::OptimizerPtr getReprOptimizer(const experiments::ModelPtr& reprModel) override {
        auto transform = getDefaultCifar10TrainTransform();
        using TransT = decltype(transform);

        experiments::OptimizerArgs<TransT> args(transform, opts_.representationsIterations);

        torch::optim::AdamOptions opt(0.0005);
//        opt.weight_decay_ = 5e-4;
        auto optim = std::make_shared<torch::optim::Adam>(reprModel->parameters(), opt);
        args.torchOptim_ = optim;

        auto lr = &(optim->options.learning_rate_);
        args.lrPtrGetter_ = [=]() { return lr; };

        const auto batchSize= 256;
        auto dloaderOptions = torch::data::DataLoaderOptions(batchSize);
        args.dloaderOptions_ = std::move(dloaderOptions);

        auto optimizer = std::make_shared<experiments::DefaultOptimizer<TransT>>(args);
        attachDefaultListeners(optimizer, convParams_);
        return optimizer;
    }

    experiments::OptimizerPtr getDecisionOptimizer(const experiments::ModelPtr& decisionModel) override {
        auto transform = torch::data::transforms::Stack<>();
        using TransT = decltype(transform);

        experiments::OptimizerArgs<TransT> args(transform, opts_.decisionIterations);

        torch::optim::AdamOptions opt(0.0005);
//        opt.weight_decay_ = 5e-4;
        auto optim = std::make_shared<torch::optim::Adam>(decisionModel->parameters(), opt);
        args.torchOptim_ = optim;

        auto lr = &(optim->options.learning_rate_);
        args.lrPtrGetter_ = [=]() { return lr; };

        const auto batchSize= 256;
        auto dloaderOptions = torch::data::DataLoaderOptions(batchSize);
        args.dloaderOptions_ = std::move(dloaderOptions);

        auto optimizer = std::make_shared<experiments::DefaultOptimizer<TransT>>(args);
        attachDefaultListeners(optimizer, decisionParams_);
        return optimizer;
    }

private:
    CommonEmOptions opts_;
    json convParams_;
    json decisionParams_;

};


// ExactLinearEm & co

class LinearClassifier : public experiments::Model {
public:
    LinearClassifier(int in, int out) {
        linear_ = register_module("linear", torch::nn::Linear(in, out));
    }

    torch::Tensor forward(torch::Tensor x) override {
        x = x.view({x.size(0), -1});
        x = experiments::correctDevice(x, *this);
        return linear_->forward(x);
    }

    void setParameters(torch::Tensor weight, torch::Tensor bias) {
        weight = weight.to(linear_->weight.device());
        linear_->weight = std::move(weight);
        bias = bias.to(linear_->bias.device());
        linear_->bias = std::move(bias);
    }

private:
    torch::nn::Linear linear_{nullptr};

};

class ExactLinearOptimizer : public experiments::Optimizer {
public:
    ExactLinearOptimizer() = default;

    void train(TensorPairDataset& ds, LossPtr loss, experiments::ModelPtr model) const override {
        auto x = ds.data();
        x = x.view({x.size(0), -1});
        auto y = ds.targets();
        y = y.view({-1, 1});
        auto yOnehot = torch::zeros({y.size(0), 10}, torch::kFloat32);
        yOnehot.scatter_(1, y, 1);

        auto b = torch::ones({x.size(0), 1}, torch::kFloat32);
        x = torch::cat({x, b}, 1);
        auto solution = linearLeastSquares(x, yOnehot);
        auto weight = solution.slice(1, 0, solution.size(1) - 1, 1);
        auto bias = solution.slice(1, solution.size(1) - 1, solution.size(1), 1);
        bias = bias.view({-1});

        auto m = std::dynamic_pointer_cast<LinearClassifier>(model);
        m->setParameters(weight, bias);
    }

    ~ExactLinearOptimizer() override = default;

private:
    torch::Tensor linearLeastSquares(const torch::Tensor& x, const torch::Tensor& y) const {
        auto tmp = (torch::mm(x.t(), x)).inverse();
        tmp = torch::mm(tmp, x.t());
        tmp = torch::mm(tmp, y);
        return tmp.t();
    }

};

class ExactLinearEm : public CommonEm {
public:
    ExactLinearEm(experiments::ModelPtr reprModel,
                  experiments::ClassifierPtr decisionModel,
                  const json& params)
            : CommonEm(std::make_shared<experiments::ConvModel>(reprModel, decisionModel), params) {

    }

protected:
    experiments::OptimizerPtr getDecisionOptimizer(const experiments::ModelPtr& decisionModel) override {
        return std::make_shared<ExactLinearOptimizer>();
    }

};
