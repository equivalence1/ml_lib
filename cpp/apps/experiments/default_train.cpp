#include "common.h"

#include <experiments//datasets/cifar10/cifar10_reader.h>
#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>

#include <util/json.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

int main(int argc, char* argv[]) {
    using namespace experiments;

    // Init model

    json params = {
            {ParamKeys::ModelKey, {
                    {ParamKeys::ConvKey, {
                            {ParamKeys::ModelArchKey, "LeNet"},
//                            {ParamKeys::ModelArchVersionKey, 16},
                    }},
                    {ParamKeys::ClassifierKey, {
                            {ParamKeys::ClassifierBaselineKey, {
                                    {ParamKeys::ModelArchKey, "MLP"},
                                    {ParamKeys::DimsKey, {16 * 5 * 5, 120, 84, 10}},
                            }},
                    }},
            }},
            {ParamKeys::DeviceKey, "GPU"},
            {ParamKeys::DatasetKey, "cifar-10"},
            {ParamKeys::ModelCheckpointFileKey, "lenet_checkpoint.pt"},
            {ParamKeys::BatchSizeKey, 128},
            {ParamKeys::ReportsPerEpochKey, 10},
            {ParamKeys::NIterationsKey, 1000},
            {ParamKeys::StepSizeKey, 0.001},
    };

    auto device = getDevice(params[ParamKeys::DeviceKey]);
    int batchSize = params[ParamKeys::BatchSizeKey];

    const json& convParams = params[ParamKeys::ModelKey][ParamKeys::ConvKey];
    const json& classParams = params[ParamKeys::ModelKey][ParamKeys::ClassifierKey];

    auto conv = createConvLayers({}, convParams);
    auto classifier = createClassifier(10, classParams);

    auto model = std::make_shared<ConvModel>(conv, classifier);
    model->to(device);

    // Load data
    auto dataset = readDataset(params[ParamKeys::DatasetKey]);

    // Create optimizer
    auto optimizer = getDefaultOptimizer(model, params);
    auto loss = std::make_shared<CrossEntropyLoss>();

    // AttachListeners

    std::string modelCheckpoint = params[ParamKeys::ModelCheckpointFileKey];
    attachDefaultListeners(optimizer, params);
    auto mds = dataset.second.map(getDefaultCifar10TestTransform());

    experiments::Optimizer::emplaceEpochListener<experiments::EpochEndCallback>(optimizer.get(), [&](int epoch, experiments::Model& model) {
        model.eval();

        auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(batchSize));
        int rightAnswersCnt = 0;

        for (auto& batch : *dloader) {
            auto data = batch.data;
            data = data.to(device);
            torch::Tensor target = batch.target;

            torch::Tensor prediction = model.forward(data);
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

    // Train model

    optimizer->train(dataset.first, loss, model);

    return 0;
}
