#include "common.h"

#include <experiments/core/transform.h>
#include <experiments/core/model.h>
#include <experiments/core/tensor_pair_dataset.h>

#include <experiments/datasets/cifar10/cifar10_reader.h>
#include <experiments/datasets/mnist/mnist_reader.h>
#include <experiments/datasets/svhn/svhn_reader.h>

#include <torch/torch.h>

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>

TransformType getDefaultCifar10TrainTransform() {
    // transforms are similar to https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L31
    // We do normalization when we load dataset
//    auto normTransformTrain = std::make_shared<torch::data::transforms::Normalize<>>(
//            torch::ArrayRef<double>({0.4914, 0.4822, 0.4465}),
//            torch::ArrayRef<double>({0.2023, 0.1994, 0.2010}));
//    auto cropTransformTrain = std::make_shared<experiments::RandomCrop>(
//            std::vector<int>({32, 32}),
//            std::vector<int>({4, 4}));
    auto flipTransformTrain = std::make_shared<experiments::RandomHorizontalFlip>(0.5);
    auto stackTransformTrain = std::make_shared<torch::data::transforms::Stack<>>();

    auto transformFunc = [=](std::vector<torch::data::Example<>>&& batch){
//        batch = normTransformTrain->apply_batch(batch);
        batch = flipTransformTrain->apply_batch(batch);
//        batch = cropTransformTrain->apply_batch(batch);
        return stackTransformTrain->apply_batch(batch);
    };

    TransformType transform(transformFunc);
    return transform;
}

TransformType getDefaultCifar10TestTransform() {
    // transforms are similar to https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L38
    // We do normalization when we load dataset
//    auto normTransformTrain = std::make_shared<torch::data::transforms::Normalize<>>(
//        std::vector<double>({0.4914, 0.4822, 0.4465}),
//        std::vector<double>({0.2023, 0.1994, 0.2010}));
    auto stackTransformTrain = std::make_shared<torch::data::transforms::Stack<>>();

    auto transformFunc = [=](std::vector<torch::data::Example<>>&& batch){
//        batch = normTransformTrain->apply_batch(batch);
        return stackTransformTrain->apply_batch(batch);
    };

    TransformType transform(transformFunc);
    return transform;
}


TransformType getCifar10TrainFinalCatboostTransform() {
  // transforms are similar to https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L38
  // We do normalization when we load dataset
//  auto normTransformTrain = std::make_shared<torch::data::transforms::Normalize<>>(
//      std::vector<double>({0.4914, 0.4822, 0.4465}),
//      std::vector<double>({0.2023, 0.1994, 0.2010}));
  auto stackTransformTrain = std::make_shared<torch::data::transforms::Stack<>>();

  auto transformFunc = [=](std::vector<torch::data::Example<>>&& batch){
//    batch = normTransformTrain->apply_batch(batch);
    const int batchSize = batch.size();
    for (int i = 0; i < batchSize; ++i) {
      auto example = batch[i];
      torch::data::Example flippedExample = {example.data.flip(2), example.target};
      batch.push_back(flippedExample);
    }
    return stackTransformTrain->apply_batch(batch);
  };

  TransformType transform(transformFunc);
  return transform;
}


OptimizerType<TransformType> getDefaultOptimizer(const experiments::ModelPtr& model,
        const json& params) {
    using namespace experiments;

    double step = params[ParamKeys::StepSizeKey];
    int epochs = params[ParamKeys::NIterationsKey];
    int batchSize = params[ParamKeys::BatchSizeKey];
    auto device = getDevice(params[ParamKeys::DeviceKey]);

    experiments::OptimizerArgs<TransformType> args(getDefaultCifar10TrainTransform(),
            epochs, device);

    args.epochs_ = epochs;
    args.dloaderOptions_ = torch::data::DataLoaderOptions(batchSize);

//    torch::optim::AdamOptions opt(0.1);
    torch::optim::SGDOptions opt(step);
    opt.momentum_ = 0.9;
    opt.weight_decay_ = 5e-4;
    auto optim = std::make_shared<torch::optim::SGD>(model->parameters(), opt);
    args.torchOptim_ = optim;

    auto learningRatePtr = &(optim->options.learning_rate_);
    args.lrPtrGetter_ = [=](){return learningRatePtr;};

    auto optimizer = std::make_shared<experiments::DefaultOptimizer<TransformType>>(args);
    return optimizer;
}

void attachDefaultListeners(const experiments::OptimizerPtr& optimizer,
                            const json& params,int reduction, std::vector<int> threshold ) {
    using namespace experiments;

    // TODO for now just hardcoded Cifar-10 ds size
    int nBatchesReport = 50000 / (int)params[ParamKeys::BatchSizeKey] / (int)params[ParamKeys::ReportsPerEpochKey];

    auto brListener = std::make_shared<experiments::BatchReportOptimizerListener>(nBatchesReport);
    optimizer->registerListener(brListener);

    auto epochReportOptimizerListener = std::make_shared<experiments::EpochReportOptimizerListener>();
    optimizer->registerListener(epochReportOptimizerListener);

    // see https://github.com/kuangliu/pytorch-cifar/blob/master/README.md#learning-rate-adjustment
    auto lrDecayListener = std::make_shared<experiments::LrDecayOptimizerListener>(reduction, threshold
                                                                                   );
//    auto lrDecayListener = std::make_shared<experiments::LrDecayOptimizerListener>(10,
//                                                                                   std::vector<int>({50, 100, 150, 200}));
    optimizer->registerListener(lrDecayListener);

//    auto msListener = std::make_shared<experiments::ModelSaveOptimizerListener>(1, savePath);
//    optimizer->registerListener(msListener);
}

torch::DeviceType getDevice(const std::string& deviceType) {
    if (deviceType == "GPU") {
        return torch::kCUDA;
    } else {
        return torch::kCPU;
    }
}

std::pair<TensorPairDataset, TensorPairDataset> readDataset(const std::string& dataset) {
    if (dataset == "cifar-10") {
        const std::string& path = "../../../../resources/cifar10/cifar-10-batches-bin";
        return experiments::cifar10::read_dataset(path);
    } else if (dataset == "mnist") {
        const std::string& path = "../../../../resources/mnist";
        return experiments::mnist::read_dataset(path);
    } else if (dataset == "svhn") {
        const std::string& path = "../../../../resources/svhn";
        return experiments::svhn::read_dataset(path);
    } else {
        throw std::runtime_error("Unsupported dataset");
    }
}
