#include "mnist_reader.h"

#include "experiments/core/tensor_pair_dataset.h"
#include "experiments/core/transform.h"

#include <torch/torch.h>

#include <utility>

namespace experiments::mnist {

TensorPairDataset toCifarFormat(TensorPairDataset ds) {
    auto mds = ds
            .map(Padding(std::vector<int>({2, 2})))
            .map(ChannelReplicate(3))
            .map(torch::data::transforms::Normalize(
                    std::vector<double>({0.5, 0.5, 0.5}),
                    std::vector<double>({0.5, 0.5, 0.5})))
            .map(torch::data::transforms::Stack<>());

    std::vector<unsigned long> indices;
    for (int i = 0; i < ds.size().value(); ++i) {
        indices.push_back(i);
    }

    auto examples = mds.get_batch(indices);
    return {examples.data, examples.target};
}

std::pair<TensorPairDataset, TensorPairDataset> read_dataset(
        int trainLimit,
        int testLimit) {
    static const std::string folder = "../../../../resources/mnist";

    auto trainDataset = torch::data::datasets::MNIST(folder);
    auto testDataset = torch::data::datasets::MNIST(folder,
                                                    torch::data::datasets::MNIST::Mode::kTest);

    auto trainData = trainDataset.images().slice(0, 0, trainLimit, 1);
    auto trainTargets = trainDataset.targets().slice(0, 0, trainLimit, 1);
    auto trainTensorDataset = TensorPairDataset(std::move(trainData), std::move(trainTargets));
    trainTensorDataset = toCifarFormat(trainTensorDataset);

    auto testData = testDataset.images().slice(0, 0, testLimit, 1);
    auto testTargets = testDataset.targets().slice(0, 0, testLimit, 1);
    auto testTensorDataset = TensorPairDataset(std::move(testData), std::move(testTargets));
    testTensorDataset = toCifarFormat(testTensorDataset);

    return std::make_pair(trainTensorDataset, testTensorDataset);
}

}
