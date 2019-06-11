#include "common.h"
#include "common_em.h"

#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/em_like_train.h>
#include <experiments/core/transform.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

int main(int argc, const char* argv[]) {
    using namespace experiments;

    // Init model

    auto paramsFolder = getParamsFolder(argc, argv);
    auto params = readJson(paramsFolder + "train_default_em_params.json");

    auto device = getDevice(params[ParamKeys::DeviceKey]);
    int batchSize = params[ParamKeys::BatchSizeKey];

    const json& convParams = params[ParamKeys::ModelKey][ParamKeys::ConvKey];
    const json& classParams = params[ParamKeys::ModelKey][ParamKeys::ClassifierKey];

    auto conv = createConvLayers({}, convParams);
    auto classifier = createClassifier(10, classParams);

    auto model = std::make_shared<ConvModel>(conv, classifier);
    model->to(device);

    // Read dataset
    auto dataset = readDataset(params[ParamKeys::DatasetKey]);

    // Init trainer

    std::vector<int> iterations(params[ParamKeys::NIterationsKey]);
    CommonEm emTrainer(model, params);

    // Attach Listeners

    auto mds = dataset.second.map(getDefaultCifar10TestTransform());
    emTrainer.registerGlobalIterationListener([&](uint32_t epoch, ModelPtr model) {
        model->eval();

        auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(batchSize));
        int rightAnswersCnt = 0;

        for (auto& batch : *dloader) {
            auto data = batch.data;
            data = data.to(device);
            torch::Tensor target = batch.target;

            torch::Tensor prediction = model->forward(data);
            prediction = torch::argmax(prediction, 1);

            prediction = prediction.to(torch::kCPU);

            auto targetAccessor = target.accessor<int64_t, 1>();
            auto predictionsAccessor = prediction.accessor<int64_t, 1>();
            int size = target.size(0);

            for (int i = 0; i < size; ++i) {
                const int targetClass = targetAccessor[i];
                const int predictionClass = predictionsAccessor[i];
                if (targetClass == predictionClass) {
                    rightAnswersCnt++;
                }
            }
        }

        std::cout << "Test accuracy: " <<  rightAnswersCnt * 100.0f / dataset.second.size().value() << std::endl;
    });

    // Train

    auto loss = std::make_shared<CrossEntropyLoss>();
    emTrainer.train(dataset.first, loss);

    // Eval model

    auto acc = evalModelTestAccEval(dataset.second,
                                    model,
                                    device,
                                    getDefaultCifar10TestTransform());

    std::cout << "Test accuracy: " << std::setprecision(2)
              << acc << "%" << std::endl;
}
