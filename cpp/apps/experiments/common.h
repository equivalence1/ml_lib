#pragma once

#include <experiments/core/optimizer.h>
#include <experiments/core/model.h>
#include <experiments/core/tensor_pair_dataset.h>
#include <util/json.h>

#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>
#include <utility>

using TransformType = torch::data::transforms::BatchLambda<std::vector<torch::data::Example<>>, torch::data::Example<>>;

TransformType getDefaultCifar10TrainTransform();

TransformType getDefaultCifar10TestTransform();

TransformType getCifar10TrainFinalCatboostTransform();

template <typename T>
using OptimizerType = std::shared_ptr<experiments::DefaultOptimizer<T>>;

OptimizerType<TransformType> getDefaultOptimizer(const experiments::ModelPtr &model,
                                                 const json& params);

void attachDefaultListeners(const experiments::OptimizerPtr& optimizer,
        const json& params, int reduction=20, std::vector<int> threshold = std::vector<int>({30, 50, 100}));

template <typename T>
float evalModelTestAccEval(TensorPairDataset& ds,
        const experiments::ModelPtr& model,
        const T& transform) {
    model->eval();

    auto mds = ds.map(transform);
    auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(100));

    int rightAnswersCnt = 0;

    for (auto& batch : *dloader) {
        auto data = batch.data;
        torch::Tensor target = batch.target;

        torch::Tensor prediction = model->forward(data);
        prediction = prediction.to(torch::kCPU);

        for (int i = 0; i < target.size(0); ++i) {
            if (target[i].item<int64_t>() == torch::argmax(prediction[i]).item<int64_t>()) {
                rightAnswersCnt++;
            }
        }
    }

    return rightAnswersCnt * 100.0f / ds.size().value();
}

std::pair<TensorPairDataset, TensorPairDataset> readDataset(const json& params);

std::string getParamsFolder(int argc, const char* argv[]);
