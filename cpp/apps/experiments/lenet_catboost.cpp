#include "common.h"
#include "common_em.h"
#include "catboost_nn.h"

#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/em_like_train.h>
#include <experiments/core/polynom_model.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

int main(int argc, char* argv[]) {
    using namespace experiments;

    // Init model

    CatBoostNNConfig catBoostNnConfig;
    catBoostNnConfig.batchSize = 128;
    catBoostNnConfig.lambda_ = 1;
    catBoostNnConfig.sgdStep_ = 0.1;
    catBoostNnConfig.representationsIterations = 10;
    catBoostNnConfig.globalIterationsCount = 1000;

    catBoostNnConfig.catboostParamsFile = "../../../../cpp/apps/cifar_networks/lenet_params/catboost_params_gpu.json";
    catBoostNnConfig.catboostInitParamsFile = "../../../../cpp/apps/cifar_networks/lenet_params/catboost_params_init.json";
    catBoostNnConfig.catboostFinalParamsFile = "../../../../cpp/apps/cifar_networks/lenet_params/catboost_params_final.json";

    json params = {
            {ParamKeys::ModelKey, {
                    {ParamKeys::ConvKey, {
                            {ParamKeys::ModelArchKey, "LeNet"},
//                          {ParamKeys::ModelArchVersionKey, 16},
                    }},
                    {ParamKeys::ClassifierKey, {
                            {ParamKeys::ClassifierMainKey, {
                                    {ParamKeys::ModelArchKey, "Polynom"},
                                    {ParamKeys::LambdaKey, catBoostNnConfig.lambda_},
                            }},
                            {ParamKeys::ClassifierBaselineKey, {
                                    {ParamKeys::ModelArchKey, "MLP"},
                                    {ParamKeys::DimsKey, {16 * 5 * 5, 10}},
                            }},
                    }},
            }},
            {ParamKeys::DeviceKey, "GPU"},
            {ParamKeys::DatasetKey, "cifar-10"},
            {ParamKeys::ModelCheckpointFileKey, "lenet_catboost_checkpoint.pt"},
            {ParamKeys::BatchSizeKey, catBoostNnConfig.batchSize},
            {ParamKeys::ReportsPerEpochKey, 10},
            {ParamKeys::NIterationsKey,
                    {catBoostNnConfig.globalIterationsCount,
                     catBoostNnConfig.representationsIterations,
                     1}
            },
            {ParamKeys::StepSizeKey, catBoostNnConfig.sgdStep_},
    };

    auto device = getDevice(params[ParamKeys::DeviceKey]);

    const json& convParams = params[ParamKeys::ModelKey][ParamKeys::ConvKey];
    const json& classParams = params[ParamKeys::ModelKey][ParamKeys::ClassifierKey];

    auto conv = createConvLayers({}, convParams);
    auto classifier = createClassifier(10, classParams);

    auto model = std::make_shared<ConvModel>(conv, classifier);
    model->to(device);

    torch::setNumThreads(16);

    CatBoostNN nnTrainer(catBoostNnConfig,
                         model,
                         device,
                         makeClassifier<LinearCifarClassifier>(16 * 5 * 5));

    // Read dataset

    auto dataset = readDataset(params[ParamKeys::DatasetKey]);

    // Attach Listener

    auto mds = dataset.second.map(getDefaultCifar10TestTransform());
    nnTrainer.registerGlobalIterationListener([&](uint32_t epoch, ModelPtr model) {
        AccuracyCalcer<decltype(mds)>(device, catBoostNnConfig, mds, nnTrainer)(epoch, model);
    });


    nnTrainer.registerGlobalIterationListener([&](uint32_t epoch, ModelPtr model) {
        if (epoch % 2 != 0) {
            std::cout << "--------===============CATBOOST learn + test start ====================---------------  "
                      << std::endl;
            auto learn = nnTrainer.applyConvLayers(dataset.first.map(getCifar10TrainFinalCatboostTransform()));
            auto test = nnTrainer.applyConvLayers(dataset.second.map(getDefaultCifar10TestTransform()));
            nnTrainer.trainFinalDecision(learn, test);
            std::cout << "--------===============CATBOOST learn + test finish ====================---------------  "
                      << std::endl;
        }
    });
    // Train

    auto loss = std::make_shared<CrossEntropyLoss>();
    nnTrainer.train(dataset.first, loss);

    // Eval model

    auto acc = evalModelTestAccEval(dataset.second,
                                    model,
                                    device,
                                    getDefaultCifar10TestTransform());

    std::cout << "Test accuracy: " << std::setprecision(2)
              << acc << "%" << std::endl;
}
