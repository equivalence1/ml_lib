#include "model.h"

#include "networks/lenet.h"
#include "experiments/core/networks/vgg.h"
#include "networks/resnet.h"
#include "networks/mobile_net_v2.h"

#include <algorithm>
#include <stdexcept>

namespace experiments {

MLP::MLP(const std::vector<int>& sizes) {
    for (int i = 0; i < sizes.size() - 1; i++) {
        int in = sizes[i];
        int out = sizes[i + 1];
        auto layer = register_module("layer_" + std::to_string(i),
                torch::nn::Linear(in, out));
        layers_.emplace_back(std::move(layer));
    }
}

torch::Tensor MLP::forward(torch::Tensor x) {
    for (auto& l : layers_) {
        x = l->forward(x);
    }
    return x;
}

LinearCifarClassifier::LinearCifarClassifier(int dim) {
    fc1_ = register_module("fc1_", torch::nn::Linear(dim, 10));
}

torch::Tensor LinearCifarClassifier::forward(torch::Tensor x) {
    return fc1_->forward(x.view({x.size(0), -1}));
}

SigmoidLinearCifarClassifier::SigmoidLinearCifarClassifier(int dim) {
    fc1_ = register_module("fc1_", torch::nn::Linear(dim, 10));
}

torch::Tensor SigmoidLinearCifarClassifier::forward(torch::Tensor x) {
    return fc1_->forward(torch::sigmoid(x.view({x.size(0), -1})));
}

torch::Tensor experiments::ConvModel::forward(torch::Tensor x) {
    x = conv()->forward(x);
    return classifier()->forward(x);
}

void experiments::ConvModel::train(bool on) {
    conv()->train(on);
    classifier()->train(on);
}

torch::Tensor experiments::Classifier::forward(torch::Tensor x) {
    x = x.view({x.size(0), -1});
    auto result = classifier_->forward(x);
    if (baseline_) {
        result *= classifierScale_;
        result += baseline_->forward(x);
    }
    return result;
}

ZeroClassifier::ZeroClassifier(int numClasses)
        : numClasses_(numClasses) {

}

torch::Tensor ZeroClassifier::forward(torch::Tensor x) {
    return torch::zeros({x.size(0), numClasses_}, torch::kFloat32).to(x.device());
}

Bias::Bias(int dim) {
    bias_ = register_parameter("bias_", torch::zeros({dim}, torch::kFloat32));
}

torch::Tensor Bias::forward(torch::Tensor x) {
    torch::TensorOptions options;
    options = options.device(x.device());
    options = options.dtype(torch::kFloat32);
    torch::Tensor result = torch::zeros({x.size(0), bias_.size(0)}, options);
    result += bias_;
    return result;
}

// Utils

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params) {
    std::string modelName = params[ParamKeys::ModelArchKey];

    if (modelName == "LeNet") {
        return lenet::createConvLayers(inputShape, params);
    } else if (modelName == "VGG") {
        return vgg::createConvLayers(inputShape, params);
    } else if (modelName == "ResNet") {
        return resnet::createConvLayers(inputShape, params);
    } else if (modelName == "MobileNetV2") {
        return mobile_net_v2::createConvLayers(inputShape, params);
    }

    std::string errMsg("Unsupported model architecture");
    throw std::runtime_error(errMsg + " " + modelName);
}

ClassifierPtr createClassifier(int numClasses, const json& params) {
    ModelPtr mainClassifier{nullptr};
    ModelPtr baselineClassifier{nullptr};

    if (params.count(ParamKeys::ClassifierMainKey)) {
        throw std::runtime_error("TODO");
    } else {
        mainClassifier = std::make_shared<ZeroClassifier>(numClasses);
    }

    if (params.count(ParamKeys::ClassifierBaselineKey)) {
        const json& baselineParams = params[ParamKeys::ClassifierBaselineKey];
        std::string archType = baselineParams[ParamKeys::ModelArchKey];

        if (archType == "MLP") {
            std::vector<int> sizes = baselineParams[ParamKeys::DimsKey];
            baselineClassifier = std::make_shared<MLP>(sizes);
        } else {
            std::string errMsg("Unsupported baseline classifier type");
            throw std::runtime_error(errMsg + " " + archType);
        }
    }

    if (baselineClassifier.get() != nullptr) {
        return std::make_shared<Classifier>(mainClassifier, baselineClassifier);
    } else {
        return std::make_shared<Classifier>(mainClassifier);
    }
}

}
