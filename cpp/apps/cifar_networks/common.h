#pragma once

#include <cifar_nn/optimizer.h>
#include <cifar_nn/model.h>
#include <cifar_nn/tensor_pair_dataset.h>

#include <torch/torch.h>

#include <memory>

using TransformType = torch::data::transforms::BatchLambda<std::vector<torch::data::Example<>>, torch::data::Example<>>;

TransformType getDefaultCifar10TrainTransform();

TransformType getDefaultCifar10TestTransform();

using OptimizerType = std::shared_ptr<experiments::DefaultOptimizer<TransformType>>;

OptimizerType getDefaultCifar10Optimizer(int epochs, const experiments::ModelPtr& model,
        torch::DeviceType device);

float evalModelTestAccEval(TensorPairDataset& testDs, experiments::ModelPtr model);
