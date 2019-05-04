#include "lenet.h"
using namespace experiments;
// LeNetConv

LeNetConv::LeNetConv() {
    conv1_ = register_module("conv1_", torch::nn::Conv2d(3, 6, 5));
    conv2_ = register_module("conv2_", torch::nn::Conv2d(6, 16, 5));
}

torch::Tensor LeNetConv::forward(torch::Tensor x) {
    x = conv1_->forward(x);
    x = torch::max_pool2d(torch::relu(x), 2, 2);
    x = conv2_->forward(x);
    x = torch::max_pool2d(torch::relu(x), 2, 2);
    return x;
}

// LeNetClassifier

LeNetClassifier::LeNetClassifier() {
    layerNorm_ = register_module("layerNorm_", std::make_shared<LayerNorm>(16 * 5 * 5));
    fc1_ = register_module("fc1_", torch::nn::Linear(16 * 5 * 5, 120));
    fc2_ = register_module("fc2_", torch::nn::Linear(120, 84));
    fc3_ = register_module("fc3_", torch::nn::Linear(84, 10));
}

torch::Tensor LeNetClassifier::forward(torch::Tensor x) {
    x = x.view({x.size(0), -1});

    x = torch::relu(fc1_->forward(x));
    x = torch::relu(fc2_->forward(x));
    x = fc3_->forward(x);
    return x;
}

// LeNet


experiments::ModelPtr LeNet::conv() {
    return conv_;
}

experiments::ClassifierPtr LeNet::classifier() {
    return classifier_;
}

LeNet::LeNet(ClassifierPtr classifier) {
    conv_ = register_module("conv_", std::make_shared<LeNetConv>());
    classifier_ = register_module("classifier_", classifier);
}
