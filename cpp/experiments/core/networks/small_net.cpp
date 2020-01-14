#include "small_net.h"

namespace experiments::small_net {

static torch::nn::ConvOptions<2> buildConvOptions(int inChannels, int outChannels, int kernelSize) {
    auto convOptions = torch::nn::ConvOptions<2>(inChannels, outChannels, kernelSize);
    convOptions.padding(kernelSize / 2);
    return convOptions;
}

SmallNetConv::SmallNetConv() {
    conv1_ = register_module("conv1_", torch::nn::Conv2d(buildConvOptions( 3,  8, 3)));
    conv2_ = register_module("conv2_", torch::nn::Conv2d(buildConvOptions( 8, 16, 3)));
    conv3_ = register_module("conv3_", torch::nn::Conv2d(buildConvOptions(16, 32, 3)));
    conv4_ = register_module("conv4_", torch::nn::Conv2d(buildConvOptions(32, 32, 3)));
    bn_ = register_module("bn_", torch::nn::BatchNorm(32));
//    l_ = register_module("l_", torch::nn::Linear(512, 512));
}

torch::Tensor SmallNetConv::forward(torch::Tensor x) {
    x = correctDevice(x, *this);
    // 3 * 32 * 32
    x = torch::relu(conv1_->forward(x));
    // 8 * 32 * 32s
    x = torch::max_pool2d(x, 2, 2);
    // 8 * 16 * 16
    x = torch::relu(conv2_->forward(x));
    // 16 * 16 * 16
    x = torch::max_pool2d(x, 2, 2);
    // 16 * 8 * 8
    x = torch::relu(conv3_->forward(x));
    // 32 * 8 * 8
    x = torch::max_pool2d(x, 2, 2);
    // 32 * 4 * 4
    x = conv4_->forward(x);
//    if (lastBias_.dim() != 0) {
//        x = x.view({x.size(0), -1});
//        x += lastBias_;
//    }
//    x = x.view({x.size(0), -1});
//    x = l_->forward(x);
//    x = bn_->forward(x);
    return x;
}

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params) {
    return std::make_shared<SmallNetConv>();
}

}
