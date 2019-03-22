#include "vgg.h"

#include <memory>

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
            std::function<torch::Tensor(torch::Tensor)> layer = [=](torch::Tensor x){
                return torch::max_pool2d(x, 2, 2);
            };
            layers_.push_back(layer);

            poolings++;
            id = 0;
        } else {
            auto options = buildConvOptions(inChannels, outChannels, kKernelSize);
            string convName = "conv_" + std::to_string(poolings) + std::to_string(id);
            auto conv = register_module(convName, torch::nn::Conv2d(options));

            string bnName = "batch_norm_" + std::to_string(poolings) + std::to_string(id);
            auto batchNorm = register_module(bnName, torch::nn::BatchNorm(outChannels));

            std::function<torch::Tensor(torch::Tensor)> layer = [conv, batchNorm](torch::Tensor x){
                return torch::relu(batchNorm->forward(conv->forward(x)));
            };
            layers_.push_back(layer);

            inChannels = outChannels;
            id++;
        }
    }

    layers_.emplace_back([](torch::Tensor x){
        return torch::avg_pool2d(x, 1, 1);
    });
}

torch::Tensor Vgg16Conv::forward(torch::Tensor x) {
    for (const auto& l : layers_) {
        x = l(x);
    }
    return x;
}

// VggClassifier

VggClassifier::VggClassifier() {
    fc1_ = register_module("fc1_", torch::nn::Linear(512, 10));
}

torch::Tensor VggClassifier::forward(torch::Tensor x) {
    return fc1_->forward(x.view({x.size(0), -1}));
}

// Vgg

Vgg::Vgg(VggConfiguration cfg) {
    if (cfg == VggConfiguration::Vgg16) {
        conv_ = std::make_shared<Vgg16Conv>();
    } else {
        throw "Unsupported configuration";
    }
}

torch::Tensor Vgg::forward(torch::Tensor x) {
    x = conv_->forward(x);
    return classifier_.forward(x);
}
