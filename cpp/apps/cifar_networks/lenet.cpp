#include <cifar_nn/lenet.h>
#include <cifar_nn/cifar10_reader.hpp>
#include <cifar_nn/optimizer.h>
#include <cifar_nn/cross_entropy_loss.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

int main() {
    auto lenet = std::make_shared<LeNet>();

    const std::string& path = "../../../../python/resources/cifar10/cifar-10-batches-bin";
    auto dataset = cifar::read_dataset(path);

    auto optim = std::make_shared<DefaultSGDOptimizer>(2);
    auto loss = std::make_shared<CrossEntropyLoss>();

    optim->train(dataset.first, loss, lenet);

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
