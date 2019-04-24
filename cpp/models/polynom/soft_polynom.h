#include <iostream>
#include <exception>


void PolynomForward(
    const float lambda,
    const float* features,
    int fCount,
    int batchSize,
    const int* splits,
    const float* conditions,
    const int* polynomOffsets,
    const float* values,
    int polynomCount,
    int outDim,
    float* tempProbs,
    float* output
    );

void PolynomBackward(int batchSize,
                     float lambda,
                     const float* features,
                     int featuresCount,
                     const float* outDer,
                     int outputDim,
                     const int* featureIds,
                     const float* conditions,
                     const float* values,
                     const int* polynomOffsets,
                     int polynomCount,
                     float* featuresDer);
