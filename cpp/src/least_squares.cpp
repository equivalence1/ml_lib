#include <eigen3/Eigen/Core>
#include <Eigen/LU>
#include <eigen3/Eigen/Cholesky>
#include <vector>

#include <stdio.h>
#include <nntree/dataset.h>

using namespace Eigen;

namespace nntree {
namespace core {

// TODO(equivalence1) memory leaks here, but we will rewrite it anyways
float* LeastSquares(float *X, float *y, int rows, int colsX, int colsY) {
  Map<MatrixXf> mxX_t(X, colsX, rows);
  auto mxX = mxX_t.transpose();
  Map<MatrixXf> mxY_t(y, colsY, rows);
  auto mxY = mxY_t.transpose();

  auto solution = (mxX.transpose() * mxX).inverse() * mxX.transpose() * mxY;
  auto solution_t = solution.transpose();
  auto result = new float[solution_t.rows() * solution_t.cols()];
  assert(result != nullptr);
  Map<MatrixXf> resultMap(result, solution_t.rows(), solution_t.cols());
  resultMap = solution_t;
  return result;
}

// TODO(equivalence1) reference is not const
struct buffer_info<float> LeastSquares(core::DataSet<float, float> &ds) {
  struct buffer_info<float> res;
  auto x = ds.GetInput();
  auto y = ds.GetOutput();
  assert(y.shape[0] == x.shape[0]);
  res.ptr = LeastSquares(x.ptr, y.ptr, (int)x.shape[0], (int)x.shape[1], (int)y.shape[1]);
  res.ndim = y.ndim;
  res.strides = y.strides;
  res.shape = {x.shape[1], y.shape[1]};
  res.size = y.size;
  return res;
}

}
}