#pragma once

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

#ifdef _WIN32
#ifdef _WINDLL
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif
#else
#define EXPORT
#endif

typedef void* ResultHandle;

EXPORT void FreeHandle(ResultHandle* modelHandle);

EXPORT const char* GetErrorString();

EXPORT int TreesCount(ResultHandle handle);
EXPORT int OutputDim(ResultHandle handle);
EXPORT int TreeDepth(ResultHandle handle, int treeIndex);
EXPORT bool CopyTree(ResultHandle handle, int treeIndex, int* features, float* conditions, float* leaves, float* weights);


EXPORT bool TrainCatBoost(const float* features,
                          const float* labels,
                          const float* weights,
                          int featuresCount, int samplesCount,
                          const float* testFeatures,
                          const float* testLabels,
                          const float* testWeights,
                          int testSamplesCount,
                          const char* params,
                          ResultHandle* handle);


#if defined(__cplusplus)
}
#endif

