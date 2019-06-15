#include "vgg.h"

#include <experiments/core/params.h>

#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

namespace experiments::vgg {

static torch::nn::ConvOptions<2> buildConvOptions(int inChannels, int outChannels, int kernelSize) {
    auto convOptions = torch::nn::ConvOptions<2>(inChannels, outChannels, kernelSize);
    convOptions.padding(kernelSize / 2);
    return convOptions;
}

// Vgg16Conv

Vgg16Conv::Vgg16Conv() {
    static int layersLayout[] = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};

    int poolings = 0;
    int id = 0;
    int inChannels = 3;
    const int kKernelSize = 3;

    for (auto outChannels : layersLayout) {
        if (outChannels == -1) {
            std::function<torch::Tensor(torch::Tensor)> layer = [=](torch::Tensor x) {
                return torch::max_pool2d(x, 2, 2);
            };
            layers_.push_back(layer);

            poolings++;
            id = 0;
        } else {
            auto options = buildConvOptions(inChannels, outChannels, kKernelSize);
            std::string convName = "conv_" + std::to_string(poolings) + std::to_string(id);
            auto conv = register_module(convName, torch::nn::Conv2d(options));

            std::string bnName = "batch_norm_" + std::to_string(poolings) + std::to_string(id);
            auto batchNorm = register_module(bnName, torch::nn::BatchNorm(outChannels));

            std::function<torch::Tensor(torch::Tensor)> layer = [conv, batchNorm](torch::Tensor x){
                return torch::relu(batchNorm->forward(conv->forward(x)));
            };
            layers_.push_back(layer);

            inChannels = outChannels;
            id++;
        }
    }

    layers_.emplace_back([](torch::Tensor x) {
        return torch::avg_pool2d(x, 1, 1);
    });
}

torch::Tensor Vgg16Conv::forward(torch::Tensor x) {
    for (const auto &l : layers_) {
        x = l(x);
    }
    return x;
}

// Utils

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params) {
    std::string archVersion = params[ModelArchVersionKey];

    if (archVersion == "16") {
        return std::make_shared<Vgg16Conv>();
    }

    std::string errMsg("Unsupported VGG Architecture version VGG-");
    throw std::runtime_error(errMsg + " " + archVersion);
}

}
