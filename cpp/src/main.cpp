#include "least_squares.h"

#include <stdio.h>
#include <vector>

int main() {
  std::vector<float> x(10 * 1000);
  for (int i = 0; i < 10 * 1000; i++) {
    x[i] = 0;
    if (i % 10 == i / 10) {
      x[i] = 1;
    }
  }

  std::vector<float> y(10 * 1);
  for (int i = 0; i < 10 * 1; i++) {
    y[i] = 1;
  }

  float *res = nntree::core::least_squares(x.data(), y.data(), 10, 1000, 1);

  printf("res:\n");
  for (int i = 0; i < 10 * 1; i++) {
    printf("%f ", res[i]);
    printf("\n");
  }

}