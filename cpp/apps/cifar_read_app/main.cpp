#include "data/cifar10_reader.hpp"

#include <cstdint>
#include <vector>
#include <iostream>
#include <string>

int main() {
    const std::string& path = "../../../../python/resources/cifar10/cifar-10-batches-bin";
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(path);

    std::cout << dataset.training_images.size() << " " << dataset.test_images.size() << std::endl;
}