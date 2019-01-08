#include <core/mx_tools/ops.h>


Vec MxTools::multiply(const Mx& mx, const Vec& x, Vec to) {
    //TODO(noxoomo): do smth with immutability, we could write Vec instead of const Vec and everything will go wrong in runtime
    const Vec mxVec = mx;
    const auto& asMxTensor = mxVec.data().view({mx.ydim(), mx.xdim()});
    torch::matmul_out(to.data(), asMxTensor, x);
    return to;
}
