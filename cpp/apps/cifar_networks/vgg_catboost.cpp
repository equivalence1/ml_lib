#include "common.h"
#include "common_em.h"
#include "catboost_nn.h"

#include <cifar_nn/vgg.h>
#include <cifar_nn/cifar10_reader.h>
#include <cifar_nn/optimizer.h>
#include <cifar_nn/cross_entropy_loss.h>
#include <cifar_nn/em_like_train.h>
#include <cifar_nn/transform.h>

#include <torch/torch.h>
#include <string>
#include <memory>
#include <iostream>
#include <cifar_nn/polynom_model.h>

int main(int argc, char* argv[]) {
    auto device = torch::kCPU;
    if (argc > 1 && std::string(argv[1]) == std::string("CUDA")
        && torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA device for training" << std::endl;
    } else {
        std::cout << "Using CPU device for training" << std::endl;
    }

    // Read dataset

    const std::string& path = "../../../../resources/cifar10/cifar-10-batches-bin";
    auto dataset = cifar::read_dataset(path);

    // Init model


    CatBoostNNConfig catBoostNnConfig;
    catBoostNnConfig.batchSize = 128;
    catBoostNnConfig.dropOut_ = 0.0;
    catBoostNnConfig.lambda_ = 1;
    catBoostNnConfig.sgdStep_ = 0.1;
    catBoostNnConfig.representationsIterations = 10;
    catBoostNnConfig.catboostParamsFile = "../../../../cpp/apps/cifar_networks/catboost_params_gpu.json";
    catBoostNnConfig.catboostInitParamsFile = "../../../../cpp/apps/cifar_networks/catboost_params_init.json";
    catBoostNnConfig.catboostFinalParamsFile = "../../../../cpp/apps/cifar_networks/catboost_params_final.json";
    torch::setNumThreads(16);
    PolynomPtr polynom = std::make_shared<Polynom>();
    polynom->Lambda_ = catBoostNnConfig.lambda_;
    {
        Monom emptyMonom;
        emptyMonom.Structure_ .Splits.push_back({0, 0});
        const auto outDim = 10;
        emptyMonom.Values_.resize(outDim);
        polynom->Ensemble_.push_back(emptyMonom);
    }

    auto classifier = makeClassifierWithBaseline<PolynomModel>(
        makeCifarLinearClassifier(512),
        polynom);

    auto vgg = std::make_shared<Vgg>(VggConfiguration::Vgg16,
        classifier);

    vgg->to(device);

    CatBoostNN nnTrainer(catBoostNnConfig, vgg, device);
    // Attach Listeners

    auto mds = dataset.second.map(getDefaultCifar10TestTransform());


    nnTrainer.registerGlobalIterationListener([&](uint32_t epoch, experiments::ModelPtr model) {
        std::cout << "--------===============CATBOOST learn + test start ====================---------------  "  << std::endl;
        auto learn = nnTrainer.applyConvLayers(dataset.first.map(getCifar10TrainFinalCatboostTransform()));
        auto test =  nnTrainer.applyConvLayers(dataset.second.map(getDefaultCifar10TestTransform()));
        nnTrainer.trainFinalDecision(learn, test);
        std::cout << "--------===============CATBOOST learn + test finish ====================---------------  "  << std::endl;

    });

    nnTrainer.registerGlobalIterationListener([&](uint32_t epoch, experiments::ModelPtr model) {
        nnTrainer.setLambda(10000);
        model->eval();

        auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(256));
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
        nnTrainer.setLambda(catBoostNnConfig.lambda_);

        std::cout << "Test accuracy: " <<  rightAnswersCnt * 100.0f / dataset.second.size().value() << std::endl;
    });



    // Train

    auto loss = std::make_shared<CrossEntropyLoss>();
    nnTrainer.train(dataset.first, loss);

    // Eval model

    auto acc = evalModelTestAccEval(dataset.second,
                                    vgg,
                                    device,
                                    getDefaultCifar10TestTransform());

    std::cout << "VGG EM test accuracy: " << std::setprecision(2)
              << acc << "%" << std::endl;
}
