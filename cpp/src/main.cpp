#include "least_squares.h"

#include <stdio.h>
#include <vector>

int main() {
  std::vector<double> x(2 * 2);
  for (int i = 0; i < 2 * 2; i++) {
    x[i] = 1.0f;
  }

  std::vector<double> y(2 * 1);
  for (int i = 0; i < 2 * 1; i++) {
    y[i] = 1.0f;
  }

  auto res = nntree::core::LeastSquares(x.data(), y.data(), 2, 2, 1);

  printf("res:\n");
  for (int i = 0; i < 2 * 1; i++) {
    printf("%f ", res[i]);
    printf("\n");
  }

}