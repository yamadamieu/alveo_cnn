#ifndef DNNKERNEL_AFFINE2_H
#define DNNKERNEL_AFFINE2_H
#include <stdint.h>
#include <algorithm>
#include <ap_fixed.h>
typedef ap_fixed<16, 7, AP_RND, AP_SAT> fixed;
namespace dnnk {
    
    
void affine2(const fixed *x, const fixed *weight, const fixed* bias,int in_features, int out_features, fixed *y) {
        const int UNROLL_OCH=1;
for (int block_i = 0; block_i < out_features; block_i += UNROLL_OCH) {//
    fixed sum[UNROLL_OCH];
#pragma HLS array_partition variable=sum complete

    for (int j = 0; j < in_features; ++j) {
      for (int local_i = 0; local_i < UNROLL_OCH; local_i++) {
#pragma HLS unroll
        int i = block_i + local_i;
        if (i < out_features) {
          fixed last = (j == 0) ? (fixed)0 : sum[local_i];
          sum[local_i] = last + x[j] * weight[i * in_features + j];
        }
      }
    }

    for (int local_i = 0; local_i < UNROLL_OCH; local_i++) {
#pragma HLS unroll
      int i = block_i + local_i;
      if (i < out_features) {//バイアスを足す
        y[i] = sum[local_i] + bias[i];
      }
    }
  }
}

}  // namespace dnnk

#endif  // DNNKERNEL_LINEAR_H
