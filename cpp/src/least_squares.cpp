#include <stdlib.h>
#include <stdio.h>
#include <mkl.h>
#include <mkl_scalapack.h>
#include <iostream>

float* least_squares(float* X, float* y, int num_elems_, int dim_, bool col_malor) {
    MKL_LAYOUT layout = col_malor ? MKL_COL_MAJOR : MKL_ROW_MAJOR;
    MKL_INT info;
    MKL_INT dim = dim_;
    MKL_INT num_elems = num_elems_;
    MKL_INT ldx;
    MKL_INT ldxt;
    MKL_INT ldw;
    MKL_INT ldy;
    MKL_INT ldxtx = dim;

    float *xtx = (float *)mkl_malloc(dim * dim * sizeof(float), 128);
    float *weights = (float *)mkl_malloc(dim * 1 * sizeof(float), 128);

    if (layout == MKL_COL_MAJOR) {
        ldx = num_elems;
        ldxt = dim;
        ldw = dim;
        ldy = num_elems;
    } else {
        ldx = dim;
        ldxt = num_elems;
        ldw = 1;
        ldy = 1;
    }

    float alpha = 1.00f;
    float beta = 0.00f;
    CBLAS_TRANSPOSE notrans = CblasNoTrans;
    CBLAS_TRANSPOSE trans = CblasTrans;
    CBLAS_LAYOUT cb_layout = col_malor ? CblasColMajor : CblasRowMajor;

    cblas_sgemm (cb_layout, trans, notrans, dim, num_elems, dim, alpha, X, ldxt, X, ldx, beta, xtx, ldxtx);

    LAPACKE_sgetrf(layout, dim, dim, xtx, ldxtx, &info);

    LAPACKE_sgetri(layout, dim, xtx, ldxtx, &info);

    cblas_sgemm (cb_layout, notrans, trans, dim, dim, num_elems, alpha, xtx, ldxt, X, ldxt, beta, X, ldxt);

    cblas_sgemm (cb_layout, notrans, notrans, dim, num_elems, 1, alpha, X, ldxt, y, ldy, beta, weights, ldw);

    mkl_free(xtx);

    return weights;
}