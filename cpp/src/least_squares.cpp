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
double* LeastSquares(double *X, double *y, int rows, int colsX, int colsY) {
  Map<MatrixXd> mxX_t(X, colsX, rows);
  auto mxX = mxX_t.transpose();
  Map<MatrixXd> mxY_t(y, colsY, rows);
  auto mxY = mxY_t.transpose();

  auto solution = (mxX.transpose() * mxX).inverse() * mxX.transpose();
  auto solution_t = solution.transpose();

  auto result = new double[solution_t.rows() * solution_t.cols()];
  assert(result != nullptr);
  Map<MatrixXd> resultMap(result, solution_t.rows(), solution_t.cols());
  resultMap = solution_t;
  return result;
}

// TODO(equivalence1) reference is not const
struct buffer_info<double> LeastSquares(core::DataSet<double, double> &ds) {
  struct buffer_info<double> res;
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
