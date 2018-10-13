#include <stdlib.h>
#include <stdio.h>
#include <eigen3/Eigen/Core>
#include <Eigen/LU>
#include <eigen3/Eigen/Cholesky>
#include <iostream>
#include <vector>

#include <stdio.h>

using namespace Eigen;

float* least_squares(float* X, float* y, int num_elems_, int dim_) {
    printf("%d %d\n", num_elems_, dim_);
    Map<MatrixXf> mxX(X, dim_, num_elems_);
    Map<VectorXf> vecY(y, dim_);

    auto solution = (mxX.transpose() * mxX).inverse() * (mxX.transpose() * vecY);
    float* result = nullptr;
    Map<MatrixXf> resultMap(result, 1, dim_);
    resultMap = solution;
    assert(result != nullptr);
    return result;
}