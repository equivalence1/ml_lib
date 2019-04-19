#include "common.h"
#include <cifar_nn/cifar10_reader.hpp>
#include <cifar_nn/optimizer.h>
#include <cifar_nn/cross_entropy_loss.h>
#include <cifar_nn/em_like_train.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

struct CommonEmOptions {
    uint32_t globalItCnt;
    uint32_t reprEpochs;
    uint32_t decisionEpochs;
};

class CommonEm : public EMLikeTrainer<decltype(getDefaultCifar10TrainTransform())> {
public:
    using ConvModelPtr = std::shared_ptr<experiments::ConvModel>;

    CommonEm(CommonEmOptions opts, ConvModelPtr model, torch::DeviceType device)
            : EMLikeTrainer(getDefaultCifar10TrainTransform(), opts.globalItCnt)
            , opts_(opts)
            , model_(std::move(model))
            , device_(device) {

        initializer_ = std::make_shared<NoopInitializer>();

        representationsModel = model_->conv();
        decisionModel = model_->classifier();
    }

    experiments::ModelPtr getTrainedModel(TensorPairDataset& ds, const LossPtr& loss) override {
        train(ds, loss);
        return model_;
    }

protected:
    experiments::OptimizerPtr getReprOptimizer(const experiments::ModelPtr& reprModel) override {
        auto transform = getDefaultCifar10TrainTransform();
        using TransT = decltype(transform);

        experiments::OptimizerArgs<TransT> args(transform, opts_.reprEpochs, device_);

        torch::optim::AdamOptions opt(0.001);
//        opt.weight_decay_ = 5e-4;
        auto optim = std::make_shared<torch::optim::Adam>(reprModel->parameters(), opt);
        args.torchOptim_ = optim;

        auto lr = &(optim->options.learning_rate_);
        args.lrPtrGetter_ = [=]() { return lr; };

        auto dloaderOptions = torch::data::DataLoaderOptions(32);
        args.dloaderOptions_ = std::move(dloaderOptions);

        auto optimizer = std::make_shared<experiments::DefaultOptimizer<TransT>>(args);
        attachDefaultListeners(optimizer, 50000 / 32 / 10, "lenet_em_conv_checkpoint.pt");
        return optimizer;
    }

    experiments::OptimizerPtr getDecisionOptimizer(const experiments::ModelPtr& decisionModel) override {
        auto transform = torch::data::transforms::Stack<>();
        using TransT = decltype(transform);

        experiments::OptimizerArgs<TransT> args(transform, opts_.decisionEpochs, device_);

        torch::optim::AdamOptions opt(0.001);
//        opt.weight_decay_ = 5e-4;
        auto optim = std::make_shared<torch::optim::Adam>(decisionModel->parameters(), opt);
        args.torchOptim_ = optim;

        auto lr = &(optim->options.learning_rate_);
        args.lrPtrGetter_ = [=]() { return lr; };

        auto dloaderOptions = torch::data::DataLoaderOptions(32);
        args.dloaderOptions_ = std::move(dloaderOptions);

        auto optimizer = std::make_shared<experiments::DefaultOptimizer<TransT>>(args);
        attachDefaultListeners(optimizer, 50000 / 32 / 10, "lenet_em_classifier_checkpoint.pt");
        return optimizer;
    }

private:
    CommonEmOptions opts_;
    ConvModelPtr model_;
    torch::DeviceType device_;

};
