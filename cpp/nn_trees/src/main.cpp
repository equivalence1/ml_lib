#include <stdio.h>
#include <vector>

#include "least_squares.h"
#include "nntree/cpu_tensor.h"

int main() {
    std::vector<uint64_t> x_shape({2, 2});
    nntree::core::CpuTensor<double> x(x_shape);
    for (uint64_t i = 0; i < 2 * 2; i++) {
        x.SetVal(i, i + 1.0f);
    }

    std::vector<uint64_t> y_shape({2, 1});
    nntree::core::CpuTensor<double> y(y_shape);
    for (uint64_t i = 0; i < 2 * 1; i++) {
        y.SetVal(i, i + 1.0f);
    }

    nntree::core::CpuTensor<double> res;
    nntree::core::LeastSquares(x, y, res);

    printf("res:\n");
    for (uint64_t i = 0; i < 2 * 1; i++) {
        printf("%f ", res.GetVal(i));
        printf("\n");
    }

}
