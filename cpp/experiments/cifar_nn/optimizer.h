#pragma once

#include "tensor_pair_dataset.h"
#include "loss.h"
#include "model.h"

#include <torch/torch.h>

#include <memory>
#include <functional>
#include <vector>
#include <string>

namespace experiments {

// OptimizerListener

class OptimizerBatchListener {
public:
    OptimizerBatchListener() = default;

    virtual void batchReset() = 0;

    virtual void onBatch(int batchId, float batchLoss) = 0;

    virtual ~OptimizerBatchListener() = default;
};

class OptimizerEpochListener {
public:
    OptimizerEpochListener() = default;

    virtual void epochReset() = 0;

    virtual void onEpoch(int epoch, double* lr, ModelPtr model) = 0;

    virtual ~OptimizerEpochListener() = default;
};

class DefaultOptimizerListener : public OptimizerBatchListener, OptimizerEpochListener {
public:
    explicit DefaultOptimizerListener(int nBatchesReport,
            int nEpochsSave = -1,
            std::string savePath = "")
            : OptimizerBatchListener()
            , OptimizerEpochListener()
            , nBatchesReport_(nBatchesReport)
            , nEpochsSave_(nEpochsSave)
            , savePath_(std::move(savePath)) {
        epoch_ = 0;
        runningLoss_ = 0.;
    }

    void batchReset() override {
        runningLoss_ = 0.;
    }

    void onBatch(int batchId, float batchLoss) override {
        runningLoss_ += batchLoss;
        if ((batchId + 1) % nBatchesReport_ != 0) {
            return;
        }
        std::cout << "[" << epoch_ + 1 << ", " << (batchId + 1)
                  << "] loss: " << (runningLoss_ / nBatchesReport_) << std::endl;
        runningLoss_ = 0;
    }

    void epochReset() override {
        epochStartTime_ = std::chrono::high_resolution_clock::now();
    }

    void onEpoch(int epoch, double* lr, ModelPtr model) override {
        std::cout << "=== end of epoch #" << (epoch + 1) << ", lr = " << std::setprecision(3) << (*lr);
        if (nEpochsSave_ > 0 && (epoch + 1) % nEpochsSave_ == 0) {
            std::cout << ", saving model to '" << savePath_ << "'";
            torch::save(model, savePath_);
        }
        auto epochElapsedTime = std::chrono::high_resolution_clock::now() - epochStartTime_;
        auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(epochElapsedTime);
        std::cout << " elapsed time since last epoch: " << elapsedTimeMs.count() << std::endl;

        epoch_ = epoch;
    }

private:
    int nBatchesReport_;
    int epoch_;
    float runningLoss_;
    int nEpochsSave_;
    std::chrono::high_resolution_clock::time_point epochStartTime_;
    std::string savePath_;

};

// Optimizer

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void train(TensorPairDataset& ds, LossPtr loss, ModelPtr model) const = 0;

    void train(torch::Tensor& x, torch::Tensor& y, LossPtr loss, ModelPtr model) const {
        TensorPairDataset ds(x, y);
        this->train(ds, std::move(loss), std::move(model));
    }

    virtual void registerBatchListener(OptimizerBatchListener* listener) = 0;

    virtual void registerEpochListener(OptimizerEpochListener* listener) = 0;
};

using OptimizerPtr = std::shared_ptr<Optimizer>;

// OptimizerArgs

template <typename TransformType>
struct OptimizerArgs {
    std::shared_ptr<torch::optim::Optimizer> torchOptim_{nullptr};

    torch::data::DataLoaderOptions dloaderOptions_;
    TransformType transform_;

    std::function<double*(void)> lrPtrGetter_;

    int epochs_ = -1;
};

// DefaultOptimizer

template <typename TransformType>
class DefaultOptimizer : public Optimizer {
public:
    explicit DefaultOptimizer(const OptimizerArgs<TransformType>& args)
            : args_(std::move(args)) {

    }

    void registerBatchListener(OptimizerBatchListener* listener) override {
        batchListeners_.push_back(listener);
    }

    void registerEpochListener(OptimizerEpochListener* listener) override {
        epochListeners_.push_back(listener);
    }

    void train(TensorPairDataset& ds, LossPtr loss, ModelPtr model) const override {
        auto mds = ds.map(args_.transform_);
        auto dloader = torch::data::make_data_loader(mds, args_.dloaderOptions_);

        for (int epoch = 0; epoch < args_.epochs_; epoch++) {
            this->fireEpochResetListeners();
            this->fireBatchResetListeners();
            int batchId = 0;
            for (auto& batch : *dloader) {
                args_.torchOptim_->zero_grad();
                auto prediction = model->forward(batch.data);
                torch::Tensor lossVal = loss->value(prediction, batch.target);
                lossVal.backward();
                args_.torchOptim_->step();
                this->fireOnBatchListeners(batchId, lossVal.item<float>());
                batchId++;
            }
            double* lr = args_.lrPtrGetter_();
            this->fireOnEpochListeners(epoch, lr, model);
        }
    }

    ~DefaultOptimizer() override = default;

private:
    void fireEpochResetListeners() const {
        for (auto* listener : epochListeners_) {
            listener->epochReset();
        }
    }

    void fireOnEpochListeners(int epoch, double* lr, ModelPtr model) const {
        for (auto* listener : epochListeners_) {
            listener->onEpoch(epoch, lr, model);
        }
    }

    void fireBatchResetListeners() const {
        for (auto* listener : batchListeners_) {
            listener->batchReset();
        }
    }

    void fireOnBatchListeners(int batchId, float batchLoss) const {
        for (auto* listener : batchListeners_) {
            listener->onBatch(batchId, batchLoss);
        }
    }

private:
    OptimizerArgs<TransformType> args_;
    std::vector<OptimizerBatchListener*> batchListeners_;
    std::vector<OptimizerEpochListener*> epochListeners_;

};


}
