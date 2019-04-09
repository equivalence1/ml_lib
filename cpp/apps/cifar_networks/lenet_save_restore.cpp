#include "common.h"
#include <cifar_nn/lenet.h>
#include <cifar_nn/cifar10_reader.hpp>
#include <cifar_nn/optimizer.h>
#include <cifar_nn/cross_entropy_loss.h>
#include <cifar_nn/model.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>
#include <cassert>


int main(int argc, char* argv[]) {
    auto lenet = std::make_shared<LeNet>();
    auto lenet2 = std::make_shared<LeNet>();

    torch::Tensor fakeBatch = torch::randn({10, 3, 32, 32}, torch::kFloat32);

    auto resBeforeSave = lenet->forward(fakeBatch);

    std::cout << "saving net" << std::endl;
    torch::save(lenet, "lenet_save_restore_test.pt");
    std::cout << "restoring net" << std::endl;
    torch::load(lenet2, "lenet_save_restore_test.pt");
    std::cout << "done" << std::endl;

    auto resAfterLoad = lenet2->forward(fakeBatch);

    std::cout << resBeforeSave << std::endl;
    std::cout << resAfterLoad << std::endl;

    assert(resBeforeSave.equal(resAfterLoad));
}
