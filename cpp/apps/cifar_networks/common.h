#pragma once

#include <cifar_nn/optimizer.h>
#include <cifar_nn/model.h>
#include <cifar_nn/tensor_pair_dataset.h>

#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

using TransformType = torch::data::transforms::BatchLambda<std::vector<torch::data::Example<>>, torch::data::Example<>>;

TransformType getDefaultCifar10TrainTransform();

TransformType getDefaultCifar10TestTransform();

template <typename T>
using OptimizerType = std::shared_ptr<experiments::DefaultOptimizer<T>>;

OptimizerType<TransformType> getDefaultCifar10Optimizer(int epochs, const experiments::ModelPtr& model,
        torch::DeviceType device);

template <typename T>
void attachDefaultListeners(const OptimizerType<T>& optimizer,
        int nBatchesReport, std::string savePath) {
    // report 10 times per epoch
    auto brListener = std::make_shared<experiments::BatchReportOptimizerListener>(nBatchesReport);
    optimizer->registerListener(brListener);

    auto epochReportOptimizerListener = std::make_shared<experiments::EpochReportOptimizerListener>();
    optimizer->registerListener(epochReportOptimizerListener);

    // see https://github.com/kuangliu/pytorch-cifar/blob/master/README.md#learning-rate-adjustment
    auto lrDecayListener = std::make_shared<experiments::LrDecayOptimizerListener>(10,
                                                                                   std::vector<int>({150, 250, 350}));
    optimizer->registerListener(lrDecayListener);

    auto msListener = std::make_shared<experiments::ModelSaveOptimizerListener>(1, savePath);
    optimizer->registerListener(msListener);
}

template <typename T>
float evalModelTestAccEval(TensorPairDataset& ds,
        const experiments::ModelPtr& model,
        torch::DeviceType device,
        const T& transform) {
    model->eval();

    auto mds = ds.map(transform);
    auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(100));

    int rightAnswersCnt = 0;

    for (auto& batch : *dloader) {
        auto data = batch.data;
        data = data.to(device);
        torch::Tensor target = batch.target;

        torch::Tensor prediction = model->forward(data);
        prediction = prediction.to(torch::kCPU);

        for (int i = 0; i < target.size(0); ++i) {
            if (target[i].item<long>() == torch::argmax(prediction[i]).item<long>()) {
                rightAnswersCnt++;
            }
        }
    }

    return rightAnswersCnt * 100.0f / ds.size().value();
}
