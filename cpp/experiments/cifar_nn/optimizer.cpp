#include "optimizer.h"
#include "tensor_pair_dataset.h"
#include "loss.h"
#include "model.h"

#include <torch/torch.h>

#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

namespace experiments {

// BatchReportOptimizerListener

BatchReportOptimizerListener::BatchReportOptimizerListener(int nBatchesReport)
        : OptimizerBatchListener()
        , nBatchesReport_(nBatchesReport) {
    runningLoss_ = 0.;
}

void BatchReportOptimizerListener::batchReset() {
    runningLoss_ = 0.;
}

void BatchReportOptimizerListener::onBatch(int epoch, int batchId, float batchLoss) {
    runningLoss_ += batchLoss;
    if ((batchId + 1) % nBatchesReport_ != 0) {
        return;
    }
    std::cout << "[" << epoch << ", " << batchId << "] "
              << "loss: " << (runningLoss_ / nBatchesReport_) << std::endl;
    runningLoss_ = 0;
}

// EpochReportOptimizerListener

EpochReportOptimizerListener::EpochReportOptimizerListener()
        : OptimizerEpochListener() {

}

void EpochReportOptimizerListener::epochReset() {
    epochStartTime_ = std::chrono::high_resolution_clock::now();
}

void EpochReportOptimizerListener::onEpoch(int epoch, double *lr, experiments::ModelPtr model) {
    std::cout << "End of epoch #" << epoch << ", lr = " << std::setprecision(3) << (*lr);

    auto epochElapsedTime = std::chrono::high_resolution_clock::now() - epochStartTime_;
    auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(epochElapsedTime);
    std::cout << ", elapsed time since last epoch: " << elapsedTimeMs.count() << std::endl;
}

// LrDecayOptimizerListener

LrDecayOptimizerListener::LrDecayOptimizerListener(double lrDecay, std::vector<int> decayEpochs)
        : OptimizerEpochListener()
        , lrDecay_(lrDecay)
        , decayEpochs_(std::move(decayEpochs)) {

}

void LrDecayOptimizerListener::epochReset() {

}

void LrDecayOptimizerListener::onEpoch(int epoch, double *lr, experiments::ModelPtr model) {
    if (std::find(decayEpochs_.begin(), decayEpochs_.end(), epoch) != decayEpochs_.end()) {
        std::cout << "Decaying lr: (" << (*lr) << " -> " << (*lr / lrDecay_) << ")" << std::endl;
        *lr /= lrDecay_;
    }
}

// ModelSaveOptimizerListener

ModelSaveOptimizerListener::ModelSaveOptimizerListener(int nEpochsSave, std::string path)
        : OptimizerEpochListener()
        , nEpochsSave_(nEpochsSave)
        , path_(std::move(path)) {

}

void ModelSaveOptimizerListener::epochReset() {

}

void ModelSaveOptimizerListener::onEpoch(int epoch, double *lr, experiments::ModelPtr model) {
    if (epoch % nEpochsSave_ == 0) {
        std::cout << "Saving model to '" << path_ << "'" << std::endl;
        torch::save(model, path_);
    }
}

}
