#ifndef INN_COMMON_H
#define INN_COMMON_H

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

const float MAXNUMF = 3.4028234663852885981170418348451692544e38;
const float MAXLOGF = 88.72283905206835;
const float MINLOGF = -103.278929903431851103; /* log(2^-149) */

const float LOG2EF = 1.44269504088896341;
const float LOGE2F = 0.693147180559945309;
const float SQRTHF = 0.707106781186547524;
const float SQRT2 = 1.4142135623730950488;
#endif
