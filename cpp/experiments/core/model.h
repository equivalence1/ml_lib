#pragma once

#include "tensor_pair_dataset.h"
#include <util/json.h>

#include <torch/torch.h>

#include <memory>
#include <utility>
#include <unordered_map>

namespace experiments {

// Model

class Model : public torch::nn::Module {
public:
    virtual torch::Tensor  forward(torch::Tensor x) = 0;

    // WTF torch, this should be default behaviour
    void train(bool on = true) override {
        if (train_) {
            return;
        }
        for (auto &param : parameters()) {
            param.set_requires_grad(on);
        }
        torch::nn::Module::train(on);
        train_ = on;
    }

private:
    bool train_ = true;
};

using ModelPtr = std::shared_ptr<Model>;

// Classifier

class Classifier : public Model {
public:

    explicit Classifier(ModelPtr classifier) {
        classifier_ = register_module("classifier_", std::move(classifier));
    }

    explicit Classifier(ModelPtr classifier, ModelPtr baseline) {
        classifier_ = register_module("classifier_", std::move(classifier));
        baseline_ = register_module("baseline_", std::move(baseline));
        classifierScale_ = register_parameter("scale_", torch::ones({1}, torch::kFloat32));
    }

    virtual ModelPtr classifier() {
        return classifier_;
    }

    virtual ModelPtr baseline() {
        return baseline_;
    }

    virtual void enableBaselineTrain(bool flag) {
        if (baseline_) {
            baseline_->train(flag);
            classifierScale_.set_requires_grad(flag);
        }
    }

    torch::Tensor forward(torch::Tensor x) override;

private:
    ModelPtr classifier_;
    ModelPtr baseline_;
    torch::Tensor classifierScale_;

};

using ClassifierPtr = std::shared_ptr<Classifier>;

// ZeroClassifier

class ZeroClassifier : public Model {
public:
    ZeroClassifier(int numClasses);

    torch::Tensor forward(torch::Tensor x) override;

    ~ZeroClassifier() override = default;

private:
    int numClasses_;
};

// ConvModel

class ConvModel : public Model {
public:
    ConvModel(ModelPtr conv,
              ClassifierPtr classifier) {
        conv_ = register_module("conv_", std::move(conv));
        classifier_ = register_module("classifier_", std::move(classifier));
    }

    virtual ModelPtr conv() {
        return conv_;
    };

    virtual ClassifierPtr classifier() {
        return classifier_;
    };

    torch::Tensor forward(torch::Tensor x) override;

    void train(bool on = true) override;

private:
    ModelPtr conv_;
    ClassifierPtr classifier_;
};

using ConvModelPtr = std::shared_ptr<ConvModel>;

class MLP : public Model {
public:
    MLP(const std::vector<int>& sizes);

    torch::Tensor forward(torch::Tensor x) override;

    ~MLP() override = default;

private:
    std::vector<torch::nn::Linear> layers_;
};

class LinearCifarClassifier : public Model {
public:
    LinearCifarClassifier(int dim);

    torch::Tensor forward(torch::Tensor x) override;

    ~LinearCifarClassifier() override = default;

private:
    torch::nn::Linear fc1_{nullptr};
};

class SigmoidLinearCifarClassifier : public Model {
public:
    SigmoidLinearCifarClassifier(int dim);

    torch::Tensor forward(torch::Tensor x) override;

    ~SigmoidLinearCifarClassifier() override = default;

private:
    torch::nn::Linear fc1_{nullptr};
};


class Bias : public Model {
public:
    Bias(int dim);

    torch::Tensor forward(torch::Tensor x) override;

    ~Bias() override = default;

private:
    torch::Tensor bias_;
};

// Utils

inline ModelPtr makeCifarLinearClassifier(int inputDim) {
    return std::make_shared<LinearCifarClassifier>(inputDim);
}

inline ModelPtr makeCifarBias() {
    return std::make_shared<Bias>(10);
}

template <class Impl, class... Args>
inline ClassifierPtr makeClassifier(Args&&... args) {
    return std::make_shared<Classifier>(std::make_shared<Impl>(std::forward<Args>(args)...));
}

template <class Impl, class... Args>
inline ClassifierPtr makeClassifierWithBaseline(ModelPtr baseline, Args&&... args) {
    return std::make_shared<Classifier>(std::make_shared<Impl>(std::forward<Args>(args)...), baseline);
}

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params);
ClassifierPtr createClassifier(int numClasses, const json& params);

}
