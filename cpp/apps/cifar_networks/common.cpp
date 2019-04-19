#include "common.h"

#include <cifar_nn/transform.h>
#include <cifar_nn/model.h>
#include <cifar_nn/tensor_pair_dataset.h>

#include <torch/torch.h>

#include <vector>
#include <memory>
#include <string>

TransformType getDefaultCifar10TrainTransform() {
    // transforms are similar to https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L31

    auto normTransformTrain = std::make_shared<torch::data::transforms::Normalize<>>(
            torch::ArrayRef<double>({0.4914, 0.4822, 0.4465}),
            torch::ArrayRef<double>({0.2023, 0.1994, 0.2010}));
//    auto cropTransformTrain = std::make_shared<experiments::RandomCrop>(
//            std::vector<int>({32, 32}),
//            std::vector<int>({4, 4}));
    auto flipTransformTrain = std::make_shared<experiments::RandomHorizontalFlip>(0.5);
    auto stackTransformTrain = std::make_shared<torch::data::transforms::Stack<>>();

    auto transformFunc = [=](std::vector<torch::data::Example<>>&& batch){
        batch = normTransformTrain->apply_batch(batch);
        batch = flipTransformTrain->apply_batch(batch);
//        batch = cropTransformTrain->apply_batch(batch);
        return stackTransformTrain->apply_batch(batch);
    };

    TransformType transform(transformFunc);
    return transform;
}

TransformType getDefaultCifar10TestTransform() {
    // transforms are similar to https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L38

    auto normTransformTrain = std::make_shared<torch::data::transforms::Normalize<>>(
        std::vector<double>({0.4914, 0.4822, 0.4465}),
        std::vector<double>({0.2023, 0.1994, 0.2010}));
    auto stackTransformTrain = std::make_shared<torch::data::transforms::Stack<>>();

    auto transformFunc = [=](std::vector<torch::data::Example<>>&& batch){
        batch = normTransformTrain->apply_batch(batch);
        return stackTransformTrain->apply_batch(batch);
    };

    TransformType transform(transformFunc);
    return transform;
}

OptimizerType<TransformType> getDefaultCifar10Optimizer(int epochs, const experiments::ModelPtr& model,
        torch::DeviceType device) {
    experiments::OptimizerArgs<TransformType> args(getDefaultCifar10TrainTransform(),
            epochs, device);

    args.epochs_ = epochs;

    args.dloaderOptions_ = torch::data::DataLoaderOptions(128);

    torch::optim::AdamOptions opt(0.001);
//    opt.momentum_ = 0.9;
//    opt.weight_decay_ = 5e-4;
    auto optim = std::make_shared<torch::optim::Adam>(model->parameters(), opt);
    args.torchOptim_ = optim;

    auto learningRatePtr = &(optim->options.learning_rate_);
    args.lrPtrGetter_ = [=](){return learningRatePtr;};

    auto optimizer = std::make_shared<experiments::DefaultOptimizer<TransformType>>(args);
    return optimizer;
}

void attachDefaultListeners(const experiments::OptimizerPtr& optimizer,
                            int nBatchesReport, std::string savePath) {
    // report 10 times per epoch
    auto brListener = std::make_shared<experiments::BatchReportOptimizerListener>(nBatchesReport);
    optimizer->registerListener(brListener);

    auto epochReportOptimizerListener = std::make_shared<experiments::EpochReportOptimizerListener>();
    optimizer->registerListener(epochReportOptimizerListener);

    // see https://github.com/kuangliu/pytorch-cifar/blob/master/README.md#learning-rate-adjustment
//    auto lrDecayListener = std::make_shared<experiments::LrDecayOptimizerListener>(10,
//                                                                                   std::vector<int>({150, 250, 350}));
//    optimizer->registerListener(lrDecayListener);

//    auto msListener = std::make_shared<experiments::ModelSaveOptimizerListener>(1, savePath);
//    optimizer->registerListener(msListener);
}
