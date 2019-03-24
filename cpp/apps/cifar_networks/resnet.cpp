#include <cifar_nn/resnet.h>
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

    auto resnet = std::make_shared<ResNet>(ResNetConfiguration::ResNet16);
    resnet->to(device);

    const std::string& path = "../../../../python/resources/cifar10/cifar-10-batches-bin";
    auto dataset = cifar::read_dataset(path);
    dataset.first = dataset.first.to(device);

    auto optim = std::make_shared<DefaultSGDOptimizer>(10);
    auto loss = std::make_shared<CrossEntropyLoss>();

    optim->train(dataset.first, loss, resnet);

    // TODO evaluate on CPU, otherwise I get OOM error. Need to investigate it.
    resnet->to(torch::kCPU);
    resnet->eval();
    auto testResModel = resnet->forward(dataset.second.data());
    auto testResReal = dataset.second.targets();
    int rightAnswersCnt = 0;

    for (int i = 0; i < testResModel.size(0); ++i) {
        if (torch::argmax(testResModel[i]).item<float>() == testResReal[i].item<float>()) {
            rightAnswersCnt++;
        }
    }

    std::cout << "ResNet test accuracy: " << std::setprecision(2)
              << rightAnswersCnt * 100.0f / testResReal.size(0) << "%" << std::endl;
}
