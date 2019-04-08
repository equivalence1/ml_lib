#include "common.h"
#include <cifar_nn/lenet.h>
#include <cifar_nn/cifar10_reader.hpp>
#include <cifar_nn/optimizer.h>
#include <cifar_nn/cross_entropy_loss.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

int main(int argc, char* argv[]) {
    auto device = torch::kCPU;
    if (argc > 1 && std::string(argv[1]) == std::string("CUDA")
            && torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA device for training" << std::endl;
    } else {
        std::cout << "Using CPU device for training" << std::endl;
    }

    auto lenet = std::make_shared<LeNet>();
    lenet->to(device);

    const std::string& path = "../../../../python/resources/cifar10/cifar-10-batches-bin";
    auto dataset = cifar::read_dataset(path);

    auto transform = torch::data::transforms::Stack<>();
    experiments::OptimizerArgs<decltype(transform)> args(transform, 2, device);

    auto dloaderOptions = torch::data::DataLoaderOptions(4);
    args.dloaderOptions_ = std::move(dloaderOptions);

    torch::optim::SGDOptions opt(0.001);
    opt.momentum_ = 0.9;
    auto optim = std::make_shared<torch::optim::SGD>(lenet->parameters(), opt);
    args.torchOptim_ = optim;

    args.lrPtrGetter_ = [&](){return &optim->options.learning_rate_;};

    auto optimizer = std::make_shared<experiments::DefaultOptimizer<decltype(args.transform_)>>(args);
    auto loss = std::make_shared<CrossEntropyLoss>();

    experiments::DefaultOptimizerListener optimizerListener(2000);
    optimizer->registerBatchListener(&optimizerListener);
    optimizer->train(dataset.first, loss, lenet);

    // TODO evaluate on CPU, otherwise I get OOM error. Need to investigate it.
    lenet->to(torch::kCPU);
    lenet->eval();
    auto testResModel = lenet->forward(dataset.second.data());
    auto testResReal = dataset.second.targets();
    int rightAnswersCnt = 0;

    for (int i = 0; i < testResModel.size(0); ++i) {
        if (torch::argmax(testResModel[i]).item<float>() == testResReal[i].item<float>()) {
            rightAnswersCnt++;
        }
    }

    std::cout << "LeNet test accuracy: " << std::setprecision(2)
            << rightAnswersCnt * 100.0f / testResReal.size(0) << "%" << std::endl;
}
