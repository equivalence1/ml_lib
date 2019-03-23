#pragma once

#include "tensor_pair_dataset.h"
#include "loss.h"
#include "model.h"

#include <torch/torch.h>
#include <memory>
namespace experiments {
    class Optimizer {
    public:
        virtual ~Optimizer() = default;

        // TODO consts!!!

        virtual void train(TensorPairDataset &ds, LossPtr loss, ModelPtr model) const = 0;

        virtual void train(torch::Tensor &x, torch::Tensor &y, LossPtr loss, ModelPtr model) const = 0;

        virtual void train_adam(TensorPairDataset &ds, LossPtr loss, ModelPtr model) {}
    };

    using OptimizerPtr = std::shared_ptr<Optimizer>;
}
class DefaultSGDOptimizer : public experiments::Optimizer {
public:
    // TODO move options here.
    explicit DefaultSGDOptimizer(int epochs, torch::optim::SGDOptions options = createDefaultOptions())
            : epochs_(epochs), options_(options) {

    }

    static torch::optim::SGDOptions createDefaultOptions() {
        torch::optim::SGDOptions options(0.001);
        options.momentum_ = 0.9;
        return options;
    }

    void train(TensorPairDataset &ds, LossPtr loss, experiments::ModelPtr model) const override {
        auto mds = ds.map(torch::data::transforms::Stack<>());
        auto dloader = torch::data::make_data_loader(mds, 4);

        torch::optim::SGD optimizer(model->parameters(), options_);

        const int kBatchesReport = 2000;

        for (int epoch = 0; epoch < epochs_; epoch++) {
            int batch_index = 0;
            float runningLoss = 0;
            for (auto &batch : *dloader) {
                optimizer.zero_grad();
                auto prediction = model->forward(batch.data);
                auto lossVal = loss->value(prediction, batch.target);
                lossVal.backward();
                optimizer.step();

                runningLoss += lossVal.item<float>();
                if (++batch_index % kBatchesReport == kBatchesReport - 1) {
                    std::cout << "[" << epoch + 1 << ", " << (batch_index + 1)
                              << "] loss: " << (runningLoss / kBatchesReport) << std::endl;
                    runningLoss = 0;
                    // Serialize your model periodically as a checkpoint.
//                            torch::save(this, "net.pt");
                }
            }
        }
    }

    void train_adam(TensorPairDataset &ds, LossPtr loss, experiments::ModelPtr model) {
        auto mds = ds.map(torch::data::transforms::Stack<>());
        auto dloader = torch::data::make_data_loader(mds, 64);

        torch::optim::AdamOptions opt(options_.learning_rate());
        torch::optim::Adam optimizer(model->parameters(), opt.beta1(0.5));

        const int kBatchesReport = 120;

        for (int epoch = 0; epoch < epochs_; epoch++) {
            int batch_index = 0;
            float runningLoss = 0;
            for (auto &batch : *dloader) {
                optimizer.zero_grad();
                auto prediction = model->forward(batch.data);
                auto lossVal = loss->value(prediction, batch.target);
                lossVal.backward();
                optimizer.step();

                runningLoss += lossVal.item<float>();
                if (++batch_index % kBatchesReport == kBatchesReport - 1) {
                    std::cout << "[" << epoch + 1 << ", " << (batch_index + 1)
                              << "] loss: " << (runningLoss / kBatchesReport) << std::endl;
                    runningLoss = 0;
                    // Serialize your model periodically as a checkpoint.
//                            torch::save(this, "net.pt");
                }
            }
        }
    }

    void train(torch::Tensor &x, torch::Tensor &y, LossPtr loss, experiments::ModelPtr model) const override {
        TensorPairDataset ds(x, y);
        this->train(ds, loss, model);
    }

private:
    int epochs_;
    torch::optim::SGDOptions options_;
};
