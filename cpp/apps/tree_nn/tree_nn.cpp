#include <cifar_nn/lenet.h>
#include <cifar_nn/cifar10_reader.hpp>
#include <cifar_nn/optimizer.h>
#include <experiments/cifar_nn/tree_train.h>
#include <experiments/cifar_nn/oblivious_tree_train.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

int main() {
//    auto lenet = std::make_shared<LeNet>();

    const std::string& path = "../../../../python/resources/cifar10/cifar-10-batches-bin";
    auto dataset = cifar::read_dataset(path);

    int pic_size = 3 * 32 * 32;

    auto* train_x_t = static_cast<float*>(dataset.first.data().reshape({-1}).data_ptr());
    auto* train_y_t = static_cast<long*>(dataset.first.targets().data_ptr());
    auto* test_x_t = static_cast<float*>(dataset.second.data().reshape({-1}).data_ptr());
    auto* test_y_t = static_cast<long*>(dataset.second.targets().data_ptr());

    torch::Tensor train_x = torch::zeros({10000 * 3*32*32});
    torch::Tensor train_y = torch::zeros({10000});


    auto* data_x_ptr = static_cast<float*>(train_x.data_ptr());
    auto* data_y_ptr = static_cast<float*>(train_y.data_ptr());

    int j = 0;
    for (int i = 0; i < 50000; i++) {
        if (train_y_t[i] == 0 || train_y_t[i] == 1) {
            int add = j * 3072;
            for (int k = 0; k < 3*32*32; k++) {
                data_x_ptr[add + k] = train_x_t[i * pic_size + k];
            }
            data_y_ptr[j] = train_y_t[i];
            j++;
        }
    }

    torch::Tensor test_x = torch::zeros({2000 * 3*32*32});
    torch::Tensor test_y = torch::zeros({2000});

    auto* test_x_ptr = static_cast<float*>(test_x.data_ptr());
    auto* test_y_ptr = static_cast<float*>(test_y.data_ptr());

    j = 0;
    for (int i = 0; i < 10000; i++) {
        if (test_y_t[i] == 0 || test_y_t[i] == 1) {
            int add = j * 3072;
            for (int k = 0; k < 3*32*32; k++) {
                test_x_ptr[add + k] = test_x_t[i * pic_size + k];
            }
            test_y_ptr[j] = test_y_t[i];
            j++;
        }
    }
//
//    auto tr_x = dataset.first.data();
//    auto tr_y = dataset.first.targets();


//    auto train_x = dataset.first.data();
//    auto train_y =dataset.first.targets();
//    auto test_x = dataset.second.data();
//    auto test_y =dataset.second.targets();
//
//    auto train_mask_y = torch::__and__(train_y == 0, train_y == 1);
//    auto train_mask_x = train_mask_y.reshape({-1, 1, 1, 1});
//    auto test_mask_y = torch::__and__(test_y == 0, test_y == 1);
//    auto test_mask_x = test_mask_y.reshape({-1, 1, 1, 1});

//    auto train_mask_y = torch::__and__(tr_y == 0, tr_y == 1);
//    auto train_mask_x = train_mask_y.reshape(auto train_mask_x = train_mask_y.reshape({-1, 1, 1, 1});{-1, 1, 1, 1});//torch::ones(train_x.sizes());


//    train_x = train_x.masked_select(train_mask_x);
//    train_y = train_y.masked_select(train_mask_y);
//    test_x = test_x.masked_select(test_mask_x);
//    test_y = test_y.masked_select(test_mask_y);



//////////TreeTrainer////////
//    train_x = train_x.reshape({10000, 3*32*32});
//    test_x = test_x.reshape({2000, 3*32*32});
/////////////////////////////

    train_x = train_x.reshape({10000, 3, 32, 32});
    test_x = test_x.reshape({2000, 3, 32, 32});

    TensorPairDataset ds(train_x, train_y);

//    TreeTrainer tr;
    ObliviousTreeTrainer tr;

    auto start = std::chrono::system_clock::now();
    auto mod = tr.getTrainedModel(ds);
    std::cout << "training time: " <<  std::chrono::duration<double>(std::chrono::system_clock::now() - start).count() << "sec" << std::endl;
    start = std::chrono::system_clock::now();
    auto res = mod->forward(test_x);
    std::cout << "forward time: " <<  std::chrono::duration<double>(std::chrono::system_clock::now() - start).count() << "sec" << std::endl;

    std::cout << "accuracy: " << (2000.0 - ((torch::_cast_Float(res.gt(0)) + test_y).eq(1)).sum().item().toFloat())/20.0 << "%" << std::endl;
}
