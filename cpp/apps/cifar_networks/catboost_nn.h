#include "common.h"
#include <cifar_nn/cifar10_reader.h>
#include <cifar_nn/optimizer.h>
#include <cifar_nn/cross_entropy_loss.h>
#include <cifar_nn/em_like_train.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

struct CatBoostNNConfig {
    uint32_t globalIterationsCount = 500;
    uint32_t representationsIterations = 3;
    double dropOut_ = 0;

    int batchSize = 256;
    double lambda_ = 1.0;
    std::string catboostParamsFile = "catboost_params.json";
    std::string catboostInitParamsFile = "catboost_params.json";
    std::string catboostFinalParamsFile = "catboost_final_params.json";

    double adamStep = 0.001;

};

class CatBoostNN : public EMLikeTrainer<decltype(getDefaultCifar10TrainTransform())> {
public:
    using ConvModelPtr = std::shared_ptr<experiments::ConvModel>;

    CatBoostNN(const CatBoostNNConfig& opts,
        ConvModelPtr model,
        torch::DeviceType device)
            : EMLikeTrainer(getDefaultCifar10TrainTransform(), opts.globalIterationsCount)
            , opts_(opts)
            , model_(std::move(model))
            , device_(device) {

        initializer_ = std::make_shared<NoopInitializer>();

        representationsModel_ = model_->conv();
        decisionModel_ = model_->classifier();
    }

    template <class Ds>
    TensorPairDataset applyConvLayers(const Ds& ds) {
        representationsModel_->eval();

        auto dloader = torch::data::make_data_loader(ds, torch::data::DataLoaderOptions(256));
        auto device = representationsModel_->parameters().data()->device();
        std::vector<torch::Tensor> reprList;
        std::vector<torch::Tensor> targetsList;

        for (auto& batch : *dloader) {
            auto res = representationsModel_->forward(batch.data.to(device));
            auto target = batch.target.to(device);
            reprList.push_back(res);
            targetsList.push_back(target);
        }

        auto repr = torch::cat(reprList, 0);
        auto targets = torch::cat(targetsList, 0);
        return TensorPairDataset(repr, targets);
    }

    void setLambda(double lambda);
    experiments::ModelPtr trainFinalDecision(const TensorPairDataset& learn, const TensorPairDataset& test);

    void train(TensorPairDataset& ds, const LossPtr& loss) override;

    experiments::ModelPtr getTrainedModel(TensorPairDataset& ds, const LossPtr& loss) override;

protected:
    void trainDecision(TensorPairDataset& ds, const LossPtr& loss);
    void trainRepr(TensorPairDataset& ds, const LossPtr& loss);
protected:
    experiments::OptimizerPtr getReprOptimizer(const experiments::ModelPtr& reprModel) override;

    experiments::OptimizerPtr getDecisionOptimizer(const experiments::ModelPtr& decisionModel) override;
private:
    const CatBoostNNConfig& opts_;
    ConvModelPtr model_;
    torch::DeviceType device_;
    int64_t seed_ = 0;
    bool Init_ = true;
};
