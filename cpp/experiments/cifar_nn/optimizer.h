#pragma once

#include "tensor_pair_dataset.h"
#include "loss.h"
#include "model.h"

#include <torch/torch.h>
#include <memory>

class Optimizer {
public:
    virtual ~Optimizer() = default;

    // TODO consts!!!

    virtual void train(TensorPairDataset& ds, LossPtr loss, ModelPtr model) const = 0;
    virtual void train(torch::Tensor& x, torch::Tensor& y, LossPtr loss, ModelPtr model) const = 0;
};

using OptimizerPtr = std::shared_ptr<Optimizer>;

class DefaultSGDOptimizer : public Optimizer {
public:
    // TODO move options here.
    explicit DefaultSGDOptimizer(int epochs): epochs_(epochs) {

    }

    void train(TensorPairDataset& ds, LossPtr loss, ModelPtr model) const override {
        auto mds = ds.map(torch::data::transforms::Stack<>());
        auto dloader = torch::data::make_data_loader(mds, 64);

        torch::optim::SGDOptions options(0.001);
        options.momentum_ = 0.9;
        torch::optim::SGD optimizer(model->parameters(), options);

        for (int epoch = 0; epoch < epochs_; epoch++) {
            int batch_index = 0;
            for (auto& batch : *dloader) {
                optimizer.zero_grad();
                auto prediction = model->forward(batch.data);
                auto lossVal = loss->value(prediction, batch.target);
                lossVal.backward();
                optimizer.step();

                if (++batch_index % 100 == 0) {
                    std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                              << " | Loss: " << lossVal.item<float>() << std::endl;
                    // Serialize your model periodically as a checkpoint.
//                            torch::save(this, "net.pt");
                }
            }
        }
    }

    void train(torch::Tensor& x, torch::Tensor& y, LossPtr loss, ModelPtr model) const override {
        TensorPairDataset ds(x, y);
        this->train(ds, loss, model);
    }

private:
    int epochs_;
};
