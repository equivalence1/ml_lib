#include "catboost_nn.h"

#include <utility>

#include <catboost_wrapper.h>
#include <cifar_nn/polynom_model.h>
#include <core/vec.h>
#include <models/model.h>
#include <models/polynom/polynom.h>
#include <random>

experiments::ModelPtr CatBoostNN::getTrainedModel(TensorPairDataset& ds, const LossPtr& loss) {
    train(ds, loss);
    return model_;
}

inline TPool MakePool(const std::vector<float>& features,
                      ConstVecRef<float> labels,
                      float* weights = nullptr) {
    const int fCount = features.size() / labels.size();
    TPool pool;
    pool.Features = features.data();
    pool.Labels =  labels.data();
    pool.FeaturesCount = fCount;
    pool.SamplesCount = labels.size();
    pool.Weights = weights;
    return pool;
}


class LrLinearDecayOptimizerListener : public experiments::OptimizerEpochListener {
public:
  LrLinearDecayOptimizerListener(double from, double to, int lastEpoch)
  : from_(from)
  , to_(to)
  , lastEpoch_(lastEpoch) {

  }

  void epochReset() override {

  }

  void onEpoch(int epoch, double* lr, experiments::ModelPtr model) override {
    *lr = from_ + (to_ - from_) * epoch / lastEpoch_;
  }

private:
  double from_;
  double to_;
  int lastEpoch_;
};


static void attachReprListeners(const experiments::OptimizerPtr& optimizer,
                            int nBatchesReport, int epochs, double startLr, double endLr) {
  // report 10 times per epoch
  auto brListener = std::make_shared<experiments::BatchReportOptimizerListener>(nBatchesReport);
  optimizer->registerListener(brListener);

  auto epochReportOptimizerListener = std::make_shared<experiments::EpochReportOptimizerListener>();
  optimizer->registerListener(epochReportOptimizerListener);

  auto lrDecayListener = std::make_shared<LrLinearDecayOptimizerListener>(startLr, endLr, epochs);
  optimizer->registerListener(lrDecayListener);
}



experiments::OptimizerPtr CatBoostNN::getReprOptimizer(const experiments::ModelPtr& reprModel) {
    auto transform = getDefaultCifar10TrainTransform();
    using TransT = decltype(transform);

    experiments::OptimizerArgs<TransT> args(transform, opts_.representationsIterations, device_);

    torch::optim::SGDOptions opt(lr_);
    opt.momentum_ = 0.9;
//    torch::optim::AdamOptions opt(opts_.adamStep);
    opt.weight_decay_ = 5e-4;
//    auto optim = std::make_shared<torch::optim::Adam>(reprModel->parameters(), opt);
    auto optim = std::make_shared<torch::optim::SGD>(reprModel->parameters(), opt);
    args.torchOptim_ = optim;

    auto lr = &(optim->options.learning_rate_);
    args.lrPtrGetter_ = [=]() { return lr; };

    const auto batchSize = opts_.batchSize;
    auto dloaderOptions = torch::data::DataLoaderOptions(batchSize);
    args.dloaderOptions_ = std::move(dloaderOptions);

    auto optimizer = std::make_shared<experiments::DefaultOptimizer<TransT>>(args);
    attachReprListeners(optimizer, 50000 / batchSize / 10, opts_.representationsIterations, lr_, lr_ / 1000);
//    attachDefaultListeners(optimizer, 50000 / batchSize / 10, "lenet_em_conv_checkpoint.pt");
    return optimizer;
}


namespace {

    class CatBoostOptimizer : public experiments::Optimizer {
    public:

        explicit CatBoostOptimizer(std::string catboostOptions,
            uint64_t seed,
            double lambda,
            double dropOut
            )
        : catBoostOptions_(std::move(catboostOptions))
        , seed_(seed)
        , lambda_(lambda)
        , drouput_(dropOut) {

        }


        void train(TensorPairDataset& ds,
                   LossPtr loss,
                   experiments::ModelPtr model) const override {
            auto polynomModel = dynamic_cast<PolynomModel*>(model.get());
            auto data = ds.data();
            const int samplesCount = data.size(0);
            auto yDim = TorchHelpers::totalSize(data) / samplesCount;
            data = data.reshape({samplesCount, yDim}).to(torch::kCPU);
            auto labels = Vec(ds.targets().to(torch::kCPU, torch::kFloat32));

            std::default_random_engine engine_(seed_);
            auto poisson_ = std::poisson_distribution<int>(1);


            std::vector<int> learnIndices;
            std::vector<int> testIndices;

            std::vector<float> learnTargets;
            std::vector<float> testTargets;
            std::vector<float> learnWeights;

            auto labelsRef = labels.arrayRef();

            for (int32_t i = 0; i < labels.size(); ++i) {
                const int w =  poisson_(engine_);
                const float t = labelsRef[i];
                if (w > 0) {
                    learnIndices.push_back(i);
                    learnTargets.push_back(t);
                } else {
                    testIndices.push_back(i);
                    testTargets.push_back(t);
                }
            }

            const int64_t featuresCount = data.size(1);

            std::vector<float> learn(learnIndices.size() * featuresCount);
            std::vector<float> test(testIndices.size() * featuresCount);

            std::vector<int> usedFeatures;
            usedFeatures.resize(featuresCount);
            std::iota(usedFeatures.begin(), usedFeatures.end(), 0);

            if (drouput_ > 0) {
                std::shuffle(usedFeatures.begin(), usedFeatures.end(), engine_);
                usedFeatures.resize(usedFeatures.size() * (1.0 - drouput_));
            }
            gather(data, learnIndices, featuresCount, learn, usedFeatures);
            gather(data, testIndices, featuresCount, test, usedFeatures);


            TPool trainPool = MakePool(learn, learnTargets, learnWeights.data());
            TPool testPool = MakePool(test, testTargets);

            auto catboost = Train(trainPool, testPool, catBoostOptions_);
            Polynom polynom(PolynomBuilder().AddEnsemble(catboost).Build());
            polynom.Lambda_ = lambda_;
            std::cout << "Model size: " << catboost.Trees.size() << std::endl;
            std::cout << "Polynom size: " << polynom.Ensemble_.size() << std::endl;
            std::set<int> featureIds;
            int fCount = 0;
            for (const auto& monom : polynom.Ensemble_) {
                for (const auto& split : monom.Structure_.Splits) {
                    featureIds.insert(split.Feature);
                    fCount = std::max<int>(fCount, split.Feature);
                }
            }
            std::cout << "Polynom used features: " << fCount << std::endl;
            for (int k = 0; k < fCount; ++k) {
                if (featureIds.count(k)) {
                    std::cout <<"1";
                } else {
                    std::cout<<"0";
                }
            }
            std::cout << std::endl << "===============" << std::endl;
            polynomModel->reset(std::make_shared<Polynom>(polynom));
        }


        void train(const TensorPairDataset& ds,
                   const TensorPairDataset& testDs,
                   experiments::ModelPtr model) const {
            auto polynomModel = dynamic_cast<PolynomModel*>(model.get());
            const int samplesCount = ds.data().size(0);
            const int testsamplesCount = testDs.data().size(0);
            auto yDim = TorchHelpers::totalSize(ds.data()) / samplesCount;
            auto learnData = ds.data().reshape({samplesCount, yDim}).to(torch::kCPU);
            auto testData = testDs.data().reshape({testsamplesCount, yDim}).to(torch::kCPU);
            auto labels = Vec(ds.targets().to(torch::kCPU, torch::kFloat32));
            auto testLabels = Vec(testDs.targets().to(torch::kCPU, torch::kFloat32));

            std::vector<int> learnIndices(samplesCount);
            std::vector<int> testIndices(testsamplesCount);
            std::iota(learnIndices.begin(), learnIndices.end(), 0);
            std::iota(testIndices.begin(), testIndices.end(), 0);


            auto labelsRef = labels.arrayRef();
            auto testLabelsRef = testLabels.arrayRef();

            const int64_t featuresCount = yDim;

            std::vector<float> learn(learnIndices.size() * featuresCount);
            std::vector<float> test(testIndices.size() * featuresCount);

            std::vector<int> usedFeatures;
            usedFeatures.resize(featuresCount);
            std::iota(usedFeatures.begin(), usedFeatures.end(), 0);
            gather(learnData, learnIndices, featuresCount, learn, usedFeatures);
            gather(testData, testIndices, featuresCount, test, usedFeatures);


            TPool trainPool = MakePool(learn, labelsRef);
            TPool testPool = MakePool(test, testLabelsRef);

            auto catboost = Train(trainPool, testPool, catBoostOptions_);
            Polynom polynom(PolynomBuilder().AddEnsemble(catboost).Build());
            polynom.Lambda_ = lambda_;
            std::cout << "Model size: " << catboost.Trees.size() << std::endl;
            std::cout << "Polynom size: " << polynom.Ensemble_.size() << std::endl;
            polynomModel->reset(std::make_shared<Polynom>(polynom));
        }

    private:
        inline void gather(torch::Tensor data, VecRef<int> indices, int64_t featuresCount, VecRef<float> dst, const std::vector<int>& activeFeatures) const {
            for (uint64_t sample = 0; sample < indices.size(); ++sample) {
                Vec features = Vec(data[indices[sample]]);
                VERIFY(features.size() == featuresCount, "err");
                auto featuresRef = features.arrayRef();
                for (auto f : activeFeatures) {
                    dst[f * indices.size() + sample] = featuresRef[f];
                }
            }
        }

    private:
        std::string catBoostOptions_;
        uint64_t seed_ = 0;
        double lambda_ = 1.0;
        double drouput_ = 0.0;
    };
}


inline std::string readFile(const std::string& path) {
    std::ifstream in(path);
    std::stringstream strStream;
    strStream << in.rdbuf(); //read the file
    std::string params = strStream.str();
    return params;
}

experiments::OptimizerPtr CatBoostNN::getDecisionOptimizer(const experiments::ModelPtr& decisionModel) {
    seed_ += 10000;
    std::string params;
    if (Init_) {
        params = opts_.catboostInitParamsFile;
        Init_ = false;
    } else {
        params = opts_.catboostParamsFile;
    }

    return std::make_shared<CatBoostOptimizer>(
        readFile(params),
        seed_,
        opts_.lambda_,
        opts_.dropOut_
        );
}

void CatBoostNN::train(TensorPairDataset& ds, const LossPtr& loss) {
    lr_ = opts_.adamStep;
    initializer_->init(ds, loss, &representationsModel_, &decisionModel_);

    if (initClassifier_) {
      initialTrainRepr(ds, loss);
    }

    trainDecision(ds, loss);

    std::set<uint32_t> decayIters = {100 / opts_.representationsIterations,
                                     200 / opts_.representationsIterations,
                                     300 / opts_.representationsIterations};

    for (uint32_t i = 0; i < iterations_; ++i) {
        std::cout << "EM iteration: " << i << std::endl;
        if (decayIters.count(i)) {
          lr_ /= 10;
        }

        trainRepr(ds, loss);
        trainDecision(ds, loss);

        fireListeners(i);
    }
}



void CatBoostNN::trainDecision(TensorPairDataset& ds, const LossPtr& loss) {
    representationsModel_->train(false);
    decisionModel_->train(true);

    std::cout << "    getting representations" << std::endl;

    auto mds = ds.map(reprTransform_);
    auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(256));
    auto device = representationsModel_->parameters().data()->device();
    std::vector<torch::Tensor> reprList;
    std::vector<torch::Tensor> targetsList;

    for (auto& batch : *dloader) {
        auto res = representationsModel_->forward(batch.data.to(device)).to(torch::kCPU);
        auto target = batch.target;
        reprList.push_back(res);
        targetsList.push_back(target);
    }

    auto repr = torch::cat(reprList, 0);
    auto targets = torch::cat(targetsList, 0);
    reprList.clear();
    targetsList.clear();

    std::cout << "    optimizing decision model" << std::endl;

    auto decisionFuncOptimizer = getDecisionOptimizer(decisionModel_);
    decisionFuncOptimizer->train(repr, targets, loss, decisionModel_);
}

void CatBoostNN::trainRepr(TensorPairDataset& ds, const LossPtr& loss) {
    representationsModel_->train(true);
    decisionModel_->train(false);

    std::cout << "    optimizing representation model" << std::endl;

    LossPtr representationLoss = makeRepresentationLoss(decisionModel_, loss);
    auto representationOptimizer = getReprOptimizer(representationsModel_);
    representationOptimizer->train(ds, representationLoss, representationsModel_);

}
experiments::ModelPtr CatBoostNN::trainFinalDecision(const TensorPairDataset& learn, const TensorPairDataset& test) {
    auto optimizer = std::make_shared<CatBoostOptimizer>(
        readFile(opts_.catboostFinalParamsFile),
        seed_,
        1e10,
        0.0
    );
    auto result = std::make_shared<PolynomModel>();
    optimizer->train(learn, test, result);
    return result;
}
void CatBoostNN::setLambda(double lambda) {
    auto model = dynamic_cast<PolynomModel*>(decisionModel_.get());
    VERIFY(model != nullptr, "model is not polynom");
    model->setLambda(lambda);

}
void CatBoostNN::initialTrainRepr(TensorPairDataset& ds, const LossPtr& loss) {
  auto model = std::make_shared<CompositionalModel>(representationsModel_, initClassifier_);
  model->train(true);
  LossPtr representationLoss = makeRepresentationLoss(initClassifier_, loss);
  auto representationOptimizer = getReprOptimizer(representationsModel_);
  representationOptimizer->train(ds, representationLoss, representationsModel_);
}
//
//std::pair<torch::Tensor, torch::Tensor> CatBoostNN::representation(TensorPairDataset& ds) {
//    auto dloader = torch::data::make_data_loader(ds, torch::data::DataLoaderOptions(100));
//    auto device = representationsModel_->parameters().data()->device();
//    std::vector<torch::Tensor> reprList;
//    std::vector<torch::Tensor> targetsList;
//
//    for (auto& batch : *dloader) {
//        auto res = representationsModel_->forward(batch.data.to(device));
//        auto target = batch.target.to(device);
//        reprList.push_back(res);
//        targetsList.push_back(target);
//    }
//
//    auto repr = torch::cat(reprList, 0);
//    auto targets = torch::cat(targetsList, 0);
//}

