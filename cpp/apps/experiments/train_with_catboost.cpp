#include "common.h"
#include "common_em.h"
#include "catboost_nn.h"

#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/em_like_train.h>
#include <experiments/core/polynom_model.h>
#include <experiments/core/params.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>
#include <set>

int main(int argc, const char* argv[]) {
    using namespace experiments;

    // Init model

    auto paramsFolder = getParamsFolder(argc, argv);
    auto params = readJson(paramsFolder + "train_with_caboost_params.json");

    auto device = getDevice(params[DeviceKey]);

    const json& convParams = params[ModelKey][ConvKey];
    const json& classParams = params[ModelKey][ClassifierKey];

    auto conv = createConvLayers({}, convParams);
    auto classifier = createClassifier(2, classParams);

    auto model = std::make_shared<ConvModel>(conv, classifier);
    model->to(device);

    torch::setNumThreads(16);

    CatBoostNN nnTrainer(params,
                         model,
                         device);
//                         createClassifier(2, classParams));

    // Read dataset

    auto dataset = readDataset(params[DatasetKey]);

    // Attach Listener

    auto mds = dataset.second.map(getDefaultCifar10TestTransform());
    nnTrainer.registerGlobalIterationListener([&](uint32_t epoch, ModelPtr model) {
        // TODO params might have changed
        AccuracyCalcer<decltype(mds)>(device, params, mds, nnTrainer)(epoch, model);
    });


    nnTrainer.registerGlobalIterationListener([&](uint32_t epoch, ModelPtr model) {
        if (epoch % 2 != 0) {
            std::cout << "--------===============CATBOOST learn + test start ====================---------------  "
                      << std::endl;
            std::cout << "skipping trainFinalDecision" << std::endl;
//            auto learn = nnTrainer.applyConvLayers(dataset.first.map(getCifar10TrainFinalCatboostTransform()));
//            auto test = nnTrainer.applyConvLayers(dataset.second.map(getDefaultCifar10TestTransform()));
//            nnTrainer.trainFinalDecision(learn, test);
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
