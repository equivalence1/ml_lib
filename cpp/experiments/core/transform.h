#pragma once

#include "tensor_pair_dataset.h"

#include <torch/torch.h>

#include <vector>
#include <random>

namespace experiments {

// ChannelReplicate

class ChannelReplicate final
        : public torch::data::transforms::Transform<torch::data::Example<>, torch::data::Example<>> {
public:
    explicit ChannelReplicate(int replics);

    torch::data::Example<> apply(torch::data::Example<> x) override;

    ~ChannelReplicate() override = default;

private:
    int replics_;
};

// Padding

class Padding final
        : public torch::data::transforms::Transform<torch::data::Example<>, torch::data::Example<>> {
public:
    explicit Padding(std::vector<int> padding);

    torch::data::Example<> apply(torch::data::Example<> x) override;

    ~Padding() override = default;

private:
    std::vector<int> padding_;
};

// RandomCrop

class RandomCrop final
        : public torch::data::transforms::Transform<torch::data::Example<>, torch::data::Example<>> {
public:
    RandomCrop(std::vector<int> size, std::vector<int> padding);

    torch::data::Example<> apply(torch::data::Example<> input) override;

    ~RandomCrop() override = default;

private:
    torch::Tensor crop(int x, int y, torch::Tensor t);

private:
    std::vector<int> size_;
    Padding padding_;

    std::random_device rd_;
    std::mt19937 eng_;
    std::uniform_int_distribution<> distrX_;
    std::uniform_int_distribution<> distrY_;
};

// RandomHorizontalFlip

class RandomHorizontalFlip final
        : public torch::data::transforms::Transform<torch::data::Example<>, torch::data::Example<>> {
public:
    explicit RandomHorizontalFlip(float p = 0.5);

    torch::data::Example<> apply(torch::data::Example<> input) override;

    ~RandomHorizontalFlip() override = default;

private:
    std::random_device rd_;
    std::mt19937 eng_;
    std::bernoulli_distribution distr_;
};

// Utils

template <typename TransformType>
TensorPairDataset dsApplyTransform(TensorPairDataset& ds, TransformType transform) {
    auto mds = ds.map(transform);

    std::vector<std::size_t> indices;
    for (int i = 0; i < ds.size().value(); ++i) {
        indices.push_back(i);
    }

    torch::data::Example<> dsExamples = mds.get_batch(indices);
    return TensorPairDataset(dsExamples.data, dsExamples.target);
}

}
