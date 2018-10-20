#include <eigen3/Eigen/Core>
#include <Eigen/LU>
#include <eigen3/Eigen/Cholesky>
#include <vector>

#include <stdio.h>
#include <nntree/dataset.h>

using namespace Eigen;

namespace nntree {
namespace core {

float* least_squares(float* X, float* y, int rows, int colsX, int colsY) {
  Map<MatrixXf> mxX(X, rows, colsX);
  Map<MatrixXf> mxY(y, rows, colsY);

  auto solution = (mxX.transpose() * mxX).inverse() * mxX.transpose() * mxY;
  auto result = new float[solution.rows() * solution.cols()];
  assert(result != nullptr);
  Map<MatrixXf> resultMap(result, solution.rows(), solution.cols());
  resultMap = solution;
  return result;
}

// TODO(equivalence1) reference is not const
struct buffer_info<float> least_squares(core::DataSet<float, float> &ds) {
  struct buffer_info<float> res;
  auto x = ds.GetInput();
  auto y = ds.GetOutput();
  assert(y.shape[0] == x.shape[0]);
  res.ptr = least_squares(x.ptr, y.ptr, (int)x.shape[0], (int)x.shape[1], (int)y.shape[1]);
  res.ndim = y.ndim;
  res.strides = y.strides;
  res.shape = {x.shape[1], y.shape[1]};
  res.size = y.size;
  return res;
}

}
}