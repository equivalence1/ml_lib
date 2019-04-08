#include "transform.h"
#include "tensor_pair_dataset.h"

#include <vector>
#include <random>

namespace experiments {

// RandomCrop

RandomCrop::RandomCrop(std::vector<int> size, std::vector<int> padding)
        : size_(std::move(size))
        , padding_(std::move(padding))
        , eng_(rd_()) {

}

torch::data::Example<> RandomCrop::apply(torch::data::Example<> input) {
    torch::Tensor out = this->pad(input.data);

    int minX = 0;
    int maxX = out.size(1) - size_[0];

    int minY = 0;
    int maxY = out.size(2) - size_[1];

    distrX_.param(std::uniform_int_distribution<>::param_type(minX, maxX));
    distrY_.param(std::uniform_int_distribution<>::param_type(minY, maxY));

    out = this->crop(distrX_(eng_), distrY_(eng_), out);

    return {out, input.target};
}

torch::Tensor RandomCrop::pad(torch::Tensor x) {
    int padY = padding_[0];
    int padX = padY;
    if (padding_.size() > 1) {
        padX = padding_[1];
    }

    torch::Tensor t = torch::zeros({x.size(0), x.size(1) + padY * 2, x.size(2) + padX * 2},
                                   torch::kFloat32);

    auto tAccessor = t.accessor<float, 3>();
    auto xAccessor = x.accessor<float, 3>();

    for (int c = 0; c < x.size(0); ++c) {
        for (int i = 0; i < x.size(0); ++i) {
            for (int j = 0; j < x.size(1); ++j) {
                tAccessor[c][i + padY][j + padX] = xAccessor[c][i][j];
            }
        }
    }

    return t;
}

torch::Tensor RandomCrop::crop(int x, int y, torch::Tensor t) {
    int sizeY = size_[0];
    int sizeX = sizeY;
    if (size_.size() > 1) {
        sizeX = size_[1];
    }

    torch::Tensor out = torch::zeros({t.size(0), sizeY, sizeX},
                                     torch::kFloat32);

    auto tAccessor = t.accessor<float, 3>();
    auto outAccessor = out.accessor<float, 3>();

    for (int c = 0; c < t.size(0); ++c) {
        for (int i = 0; i < sizeY; ++i) {
            for (int j = 0; j < sizeX; ++j) {
                outAccessor[c][i][j] = tAccessor[c][y + i][x + j];
            }
        }
    }

    return out;
}

// RandomHorizontalFlip

RandomHorizontalFlip::RandomHorizontalFlip(float p)
        : eng_(rd_())
        , distr_(p) {

}

torch::data::Example<> RandomHorizontalFlip::apply(torch::data::Example<> input) {
    if (distr_(eng_)) {
        return {input.data.flip(2), input.target};
    } else {
        return input;
    }
}

}
