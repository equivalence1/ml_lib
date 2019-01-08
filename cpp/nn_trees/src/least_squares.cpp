#include <eigen3/Eigen/Core>
#include <Eigen/LU>
#include <vector>

#include "nntree/dataset.h"

namespace nntree {
    namespace core {

// TODO(equivalence1) memory leaks here, but we will rewrite it anyways
        void LeastSquares(Tensor<double>& x, Tensor<double>& y, Tensor<double>& res) {
            namespace E = Eigen;

            assert(x.Ndim() == 2);
            assert(y.Ndim() == 2);

            E::Map<E::MatrixXd, 0, E::Stride<E::Dynamic, E::Dynamic>>
                mxX_t(x.Data(),
                      x.Ncols(),
                      x.Nrows(),
                      E::Stride<E::Dynamic, E::Dynamic>(x.Strides()[0] / sizeof(double),
                                                        x.Strides()[1] / sizeof(double)));
            auto mxX = mxX_t.transpose();
            E::Map<E::MatrixXd, 0, E::Stride<E::Dynamic, E::Dynamic>>
                mxY_t(y.Data(),
                      y.Ncols(),
                      y.Nrows(),
                      E::Stride<E::Dynamic, E::Dynamic>(y.Strides()[0] / sizeof(double),
                                                        y.Strides()[1] / sizeof(double)));
            auto mxY = mxY_t.transpose();

            auto solution = (mxX.transpose() * mxX).inverse() * mxX.transpose() * mxY;
            auto solution_t = solution.transpose();

            auto result = new double[solution_t.rows() * solution_t.cols()];
            assert(result != nullptr);
            E::Map<E::MatrixXd> resultMap(result, solution_t.rows(), solution_t.cols());
            resultMap = solution_t;

            std::vector<uint64_t> shape({(uint64_t) solution.rows(),
                                         (uint64_t) solution.cols()});
            std::vector<uint64_t> strides({(uint64_t) resultMap.outerStride() * sizeof(double),
                                           (uint64_t) resultMap.innerStride() * sizeof(double)});
            res.FromMem(resultMap.data(), shape, strides, true);
        }

// TODO(equivalence1) reference is not const
        void LeastSquares(core::DataSet<double, double>& ds, Tensor<double>& res) {
            auto& x = ds.GetInput();
            auto& y = ds.GetOutput();
            assert(y.Nrows() == x.Nrows());
            LeastSquares(x, y, res);
        }

    }
}
