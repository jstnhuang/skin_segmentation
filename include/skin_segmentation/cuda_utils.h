#ifndef _SKINSEG_CUDA_UTILS_H_
#define _SKINSEG_CUDA_UTILS_H_

#include <cuda_runtime.h>

namespace skinseg {
void HandleError(cudaError_t err);
}  // namespace skinseg

#endif  // _SKINSEG_CUDA_UTILS_H_
