#include "catboost_nn.h"

#include <experiments/core/polynom_model.h>
#include <models/model.h>
#include <models/polynom/polynom.h>
#include <core/vec.h>
#include <util/io.h>

#include <catboost_wrapper.h>

#include <utility>
#include <random>
#include <stdexcept>

experiments::ModelPtr CatBoostNN::getTrainedModel(TensorPairDataset& ds, const LossPtr& loss) {
    train(ds, loss);
    return model_;
}

inline TDataSet MakePool(const std::vector<float>& features,
                      ConstVecRef<float> labels,
                      float* weights = nullptr,
                      const float* baseline = nullptr,
                      int baselineDim = 0
                      ) {
    const int fCount = features.size() / labels.size();
    TDataSet pool;
    pool.Features = features.data();
    pool.Labels =  labels.data();
    pool.FeaturesCount = fCount;
    pool.SamplesCount = labels.size();
    pool.Weights = weights;
    pool.Baseline = baseline;
    pool.BaselineDim = baselineDim;
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



experiments::OptimizerPtr CatBoostNN::getReprOptimizer(const experiments::ModelPtr& model) {
    auto transform = getDefaultCifar10TrainTransform();
    using TransT = decltype(transform);

    experiments::OptimizerArgs<TransT> args(transform, iter_, device_);
//
    torch::optim::SGDOptions opt(lr_);
    opt.momentum_ = 0.9;
//    torch::optim::AdamOptions opt(lr);
    opt.weight_decay_ = 5e-4;
//    auto optim = std::make_shared<torch::optim::Adam>(reprModel->parameters(), opt);
    auto optim = std::make_shared<torch::optim::SGD>(model->parameters(), opt);
    args.torchOptim_ = optim;

    auto lr = &(optim->options.learning_rate_);
    args.lrPtrGetter_ = [=]() { return lr; };

    int batchSize = opts_[BatchSizeKey];
    auto dloaderOptions = torch::data::DataLoaderOptions(batchSize);
    args.dloaderOptions_ = std::move(dloaderOptions);

    auto optimizer = std::make_shared<experiments::DefaultOptimizer<TransT>>(args);
    int representationsIterations = opts_[NIterationsKey][1];
    int reportsPerEpoch = opts_[ReportsPerEpochKey];
    attachReprListeners(optimizer, 50000 / batchSize / reportsPerEpoch, representationsIterations, lr_, lr_);
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
            auto classifier = dynamic_cast<experiments::Classifier*>(model.get());
            auto polynomModel = dynamic_cast<PolynomModel*>(classifier->classifier().get());
            auto data = ds.data();


            const int samplesCount = data.size(0);
            auto yDim = TorchHelpers::totalSize(data) / samplesCount;
            data = data.reshape({samplesCount, yDim}).to(torch::kCPU).contiguous();
            std::vector<float> baseline = maybeMakeBaseline(data, classifier);
            auto baselineDim = baseline.size() / samplesCount;
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
                    learnWeights.push_back(w);
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

            auto learnBaseline = gatherBaseline(baseline, learnIndices, baselineDim);
            auto testBaseline = gatherBaseline(baseline, testIndices, baselineDim);

            TDataSet trainPool = MakePool(learn,
                learnTargets,
                learnWeights.data(),
                learnBaseline.size() ? learnBaseline.data() : nullptr,
                baselineDim);
            TDataSet testPool = MakePool(test, testTargets, nullptr,
                testBaseline.size() ? testBaseline.data() : nullptr,
                baselineDim);

            std::cout << "Training catboost with options: " << catBoostOptions_ << std::endl;

            auto catboost = Train(trainPool, testPool, catBoostOptions_);
            std::cout << "CatBoost was trained " << std::endl;

            Polynom polynom(PolynomBuilder().AddEnsemble(catboost).Build());
            polynom.Lambda_ = lambda_;
            std::cout << "Model size: " << catboost.Trees.size() << std::endl;
            std::cout << "Polynom size: " << polynom.Ensemble_.size() << std::endl;
            std::map<int, int> featureIds;
            int fCount = 0;
            double total = 0;
            for (const auto& monom : polynom.Ensemble_) {
                for (const auto& split : monom.Structure_.Splits) {
                    featureIds[split.Feature]++;
                    fCount = std::max<int>(fCount, split.Feature);
                    ++total;
                }
            }
            std::cout << "Polynom used features: " << featureIds.size() << std::endl;
            for (int k = 0; k < fCount; ++k) {
                std::cout << featureIds[k] / total << " ";
            }
            std::cout << std::endl << "===============" << std::endl;
            std::cout << std::endl << "Polynom values hist" << std::endl;
            polynom.PrintHistogram();
            std::cout << std::endl << "===============" << std::endl;


            polynomModel->reset(std::make_shared<Polynom>(polynom));
        }


        void train(const TensorPairDataset& ds,
                   const TensorPairDataset& testDs,
                   experiments::ModelPtr model) const {
            auto classifier = dynamic_cast<experiments::Classifier*>(model.get());
            auto polynomModel = dynamic_cast<PolynomModel*>(classifier->classifier().get());

            const int samplesCount = ds.data().size(0);
            const int testsamplesCount = testDs.data().size(0);
            auto yDim = TorchHelpers::totalSize(ds.data()) / samplesCount;
            auto learnData = ds.data().reshape({samplesCount, yDim}).to(torch::kCPU).contiguous();
            auto testData = testDs.data().reshape({testsamplesCount, yDim}).to(torch::kCPU).contiguous();
            auto labels = Vec(ds.targets().to(torch::kCPU, torch::kFloat32).contiguous());
            auto testLabels = Vec(testDs.targets().to(torch::kCPU, torch::kFloat32).contiguous());

            auto learnBaseline = maybeMakeBaseline(learnData, classifier);
            auto testBaseline = maybeMakeBaseline(testData, classifier);
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


            auto baselineDim = learnBaseline.size() / labelsRef.size();
            TDataSet trainPool = MakePool(learn, labelsRef, nullptr,
                learnBaseline.size() ? learnBaseline.data() : nullptr,
                baselineDim);
            TDataSet testPool = MakePool(test, testLabelsRef, nullptr,
                testBaseline.size() ? testBaseline.data() : nullptr,
                baselineDim);

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

        std::vector<float> gatherBaseline(ConstVecRef<float> baseline, ConstVecRef<int> indices, int dim) const {
            std::vector<float> result;
            if (!baseline.empty()) {
                const int samplCount = baseline.size() / dim;
                result.resize(indices.size() * dim);
                for (int currentDim = 0; currentDim < dim; ++currentDim) {
                    for (size_t i = 0; i < indices.size(); ++i) {
                        result[currentDim * indices.size() + i] = baseline[currentDim * samplCount  + indices[i]];
                    }
                }
            }
            return result;
        }


        std::vector<float> maybeMakeBaseline(torch::Tensor data, experiments::Classifier* classifier) const {
            std::vector<float> baseline;
            if (classifier->baseline()) {
                classifier->baseline()->to(data.device());
                auto baselineTensor = classifier->baseline()->forward(data).to(torch::kCPU).contiguous();
                const int baselineDim = baselineTensor.size(1);
                const size_t samplesCount = baselineTensor.size(0);
                size_t baselineSize = samplesCount * baselineDim;
                baseline.resize(baselineSize);
                auto srcData = baselineTensor.data<float>();
                for (int  baselineIdx = 0;  baselineIdx<  baselineDim; ++ baselineIdx) {
                    for (int32_t i = 0; i < samplesCount; ++i) {
                        baseline[baselineIdx * samplesCount + i] = srcData[i * baselineDim + baselineIdx];
                    }
                }
            }
            return baseline;
        }

    private:
        std::string catBoostOptions_;
        uint64_t seed_ = 0;
        double lambda_ = 1.0;
        double drouput_ = 0.0;
    };
}

experiments::OptimizerPtr CatBoostNN::getDecisionOptimizer(const experiments::ModelPtr& decisionModel) {
    seed_ += 10000;
    std::string params;
    if (Init_) {
        params = opts_[CatboostParamsKey][InitParamsKey].dump();
        Init_ = false;
    } else {
        params = opts_[CatboostParamsKey][IntermediateParamsKey].dump();
    }

    return std::make_shared<CatBoostOptimizer>(
        params,
        seed_,
        ((double)opts_[ModelKey][ClassifierKey][ClassifierMainKey][LambdaKey]) * lambdaMult_,
        opts_[DropoutKey]
        );
}

void CatBoostNN::train(TensorPairDataset& ds, const LossPtr& loss) {
    lr_ = opts_[SgdStepSizeKey];

    initialTrainRepr(ds, loss);

    trainDecision(ds, loss);

    for (uint32_t i = 0; i < iterations_; ++i) {
        std::cout << "EM iteration: " << i << std::endl;

        fireScheduledParamModifiers(i);

        lr_ = opts_[SgdStepSizeKey];

        trainRepr(ds, loss);

        std::cout << "Repr was trained, calling listeners " << i << std::endl;
//        model_->classifier()->dumpClassifierScale();

        fireListeners(2 * i);
        std::cout << "========== " << i << std::endl;

        trainDecision(ds, loss);

        std::cout << "Decision was trained " << i << std::endl;

        model_->to(device_);

        fireListeners(2 * i + 1);
        std::cout << "========== " << i << std::endl;

    }
}



void CatBoostNN::trainDecision(TensorPairDataset& ds, const LossPtr& loss) {
    model_->to(device_);
    auto representationsModel = model_->conv();
    auto decisionModel = model_->classifier();
    representationsModel->train(false);

    if (model_->classifier()->baseline()) {
        model_->classifier()->enableBaselineTrain(false);
    }
    model_->classifier()->classifier()->train(true);


    std::cout << "    getting representations" << std::endl;

    auto mds = ds.map(reprTransform_);
    auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(256));
    auto device = representationsModel->parameters().data()->device();
    std::vector<torch::Tensor> reprList;
    std::vector<torch::Tensor> targetsList;

    for (auto& batch : *dloader) {
        auto res = representationsModel->forward(batch.data.to(device)).to(torch::kCPU);
        auto target = batch.target;
        reprList.push_back(res);
        targetsList.push_back(target);
    }

    auto repr = torch::cat(reprList, 0);
    auto targets = torch::cat(targetsList, 0);
    reprList.clear();
    targetsList.clear();

    std::cout << "    optimizing decision model" << std::endl;

    auto decisionFuncOptimizer = getDecisionOptimizer(decisionModel);
    decisionFuncOptimizer->train(repr, targets, loss, decisionModel);
}

void CatBoostNN::trainRepr(TensorPairDataset& ds, const LossPtr& loss) {
    model_->to(device_);

    auto representationsModel = model_->conv();
    representationsModel->train(true);
    auto decisionModel = model_->classifier();
    decisionModel->train(false);
    if (decisionModel->baseline()) {
        decisionModel->enableBaselineTrain(true);
    }

    std::cout << "    optimizing representation model" << std::endl;

    LossPtr representationLoss = makeRepresentationLoss(decisionModel, loss);
    auto representationOptimizer = getReprOptimizer(model_);

    representationOptimizer->train(ds, representationLoss, representationsModel);

}
experiments::ModelPtr CatBoostNN::trainFinalDecision(const TensorPairDataset& learn, const TensorPairDataset& test) {
    auto optimizer = std::make_shared<CatBoostOptimizer>(
        opts_[CatboostParamsKey][FinalParamsKey].dump(),
        seed_,
        1e10,
        0.0
    );
    experiments::ClassifierPtr classifier;
    auto baseline = model_->classifier()->baseline();
    if (baseline) {
        classifier = experiments::makeClassifierWithBaseline<PolynomModel>(model_->classifier()->baseline(), std::make_shared<Polynom>());
    } else {
        classifier = experiments::makeClassifier<PolynomModel>(std::make_shared<Polynom>());
    }
    optimizer->train(learn, test, classifier);
    return classifier;
}
void CatBoostNN::setLambda(double lambda) {
    auto model = dynamic_cast<PolynomModel*>(model_->classifier()->classifier().get());
    VERIFY(model != nullptr, "model is not polynom");
    model->setLambda(lambda * lambdaMult_);

}

using namespace experiments;

void CatBoostNN::initialTrainRepr(TensorPairDataset& ds, const LossPtr& loss) {
    iter_ = 3;

    if (initClassifier_) {
        auto model = std::make_shared<ConvModel>(model_->conv(), initClassifier_);
        model->train(true);
        model->to(device_);
        LossPtr representationLoss = makeRepresentationLoss(initClassifier_, loss);
        auto representationOptimizer = getReprOptimizer(model_->conv());
        representationOptimizer->train(ds, representationLoss, model_->conv());
    }

    if (model_->classifier()->baseline()) {
        iter_ = 1;
        trainRepr(ds, loss);
    }

    iter_ =  opts_[NIterationsKey][1];
}

void CatBoostNN::fireScheduledParamModifiers(int iter) {
    const std::vector<json> paramModifiers = opts_[ScheduledParamModifiersKey];

    for (const json& modifier : paramModifiers) {
        std::string field = modifier[FieldKey];
        std::vector<int> iters = modifier[ItersKey];
        std::string type = modifier["type"];

//        std::cout << field << std:: endl;
        if (type == "double") {
            std::vector<double> values = modifier[ValuesKey];
            for (int i = 0; i < iters.size(); ++i) {
                if (iters[i] == iter) {
                    opts_[field] = values[i];
                    break;
                }
            }
        } else {
            std::vector<int> values = modifier[ValuesKey];
            for (int i = 0; i < iters.size(); ++i) {
                if (iters[i] == iter) {
                    opts_[field] = values[i];
                    break;
                }
            }
        }
    }
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
