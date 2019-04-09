#include <cifar_nn/resnet.h>
#include <cifar_nn/model.h>

#include <torch/torch.h>

#include <memory>
#include <iostream>
#include <cassert>


int main(int argc, char* argv[]) {
    auto resnet = std::make_shared<ResNet>(ResNetConfiguration::ResNet16);
    auto resnet2 = std::make_shared<ResNet>(ResNetConfiguration::ResNet16);

    torch::Tensor fakeBatch = torch::randn({10, 3, 32, 32}, torch::kFloat32);

    auto resBeforeSave = resnet->forward(fakeBatch);
    auto resBeforeSave2 = resnet2->forward(fakeBatch);

    assert(!resBeforeSave.equal(resBeforeSave2));

    std::cout << "saving net" << std::endl;
    torch::save(resnet, "resnet_save_restore_test.pt");
    std::cout << "restoring net" << std::endl;
    torch::load(resnet2, "resnet_save_restore_test.pt");
    std::cout << "done" << std::endl;

    auto resAfterLoad = resnet2->forward(fakeBatch);

    std::cout << resBeforeSave << std::endl;
    std::cout << resAfterLoad << std::endl;

    assert(resBeforeSave.equal(resAfterLoad));
}
