#include "../../experiments/cifar_nn/linear_function.h"

#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor x = torch::randn({7, 7}, torch::requires_grad(true));
    torch::Tensor w = torch::randn({7, 7}, torch::requires_grad(true));

    LinearFunction f;
    auto r = f.apply({x, w})[0];

    std::cout << "r = " << r << std::endl;

    auto s = torch::sum(r);
    std::cout << "s = " << s << std::endl;

    std::cout << "backward" << std::endl;
    s.backward();

    std::cout << "x.grad = " << x.grad() << std::endl;
    std::cout << "w.grad = " << w.grad() << std::endl;
}