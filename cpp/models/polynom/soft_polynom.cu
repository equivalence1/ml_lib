#include <cuda_runtime.h>
__forceinline__ __device__  float Sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


__forceinline__ __device__  float SigmoidDer(float x) {
    const float p = 1.0f / (1.0f + expf(-x));
    return p * (1.0f - p);
}


__global__ void PolynomProbsImpl(const float* features,
                                 int fCount,
                                 int batchSize,
                                 const int* splits,
                                 const float* conditions,
                                 const int* polynomOffsets,
                                 const int* polynomDepth,
                                 int polynomCount,
                                 float* probs) {
    if (threadIdx.x < batchSize) {
        int polynomId = blockIdx.x;

        while (polynomId < polynomCount) {
            const int depth = polynomDepth[polynomId];
            int offset = polynomOffsets[polynomId];

            conditions += offset;
            splits += offset;

            float logProb = 0;
            for (int i = 0; i < depth; ++i) {
                const int f = __ldg(splits + i);
                const float c = __ldg(conditions + i);
                const float x = __ldg(features + f * batchSize + threadIdx.x);
                const float val = -(x - c);
                const float expVal = 1.0f + exp(val);

//            p( split = 1) = 1.0 / (1.0 + exp(-(x - c)))
//            c = 0, x= inf, p = 1.0 / (1.0 + exp(-inf) = 0
//            log(p) = -log(1.0 + exp(-(x - c))
                const float isTrueLogProb = isfinite(expVal) ? log(expVal) : val;
                logProb -= isTrueLogProb;
            }
            const float prob = exp(logProb);
            probs[polynomId * batchSize + threadIdx.x] = prob;
            polynomId += gridDim.x;
        }
    }
}


//batch size should be equal to BlockSize
//we need to reduce polynoms for each output dim
__global__ void PolynomForwardImpl(const float* probs,
                                   int batchSize,
                                   const float* values,
                                   int polynomCount,
                                   int outputDim,
                                   float* out) {
    //out: batch_elem0 dim0, dim1, dimk batch_elem1 dim0 dim1 dimk
    //so threads


    const size_t totalElems = outputDim * batchSize * polynomCount;
    const int flatElems = outputDim * batchSize;

    const int stripeSize = gridDim.x * flatElems;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    const int elemId = tid % flatElems;
    const int dimId = elemId % outputDim;
    const int batchId = elemId / outputDim;

    /*
     * batch size should be 2^k (128/256/512), so this will be always false
     */
    if (tid >= stripeSize) {
        return;
    }

    float sum = 0;

    while (tid < totalElems) {
        const int polynomId = tid / flatElems;
        const float polynomProb = __ldg(probs + polynomId);
        const float out = __ldg(values + polynomId * outputDim + dimId);
        sum += polynomProb * out;

        tid += stripeSize;
    }
    atomicAdd(out + batchId * outputDim + dimId, sum);
}
