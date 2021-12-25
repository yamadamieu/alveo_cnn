#ifndef DNNKERNEL_AFFINE_H
#define DNNKERNEL_AFFINE_H
#include <stdint.h>
#include <algorithm>
#include <ap_fixed.h>
typedef ap_fixed<16, 7, AP_RND, AP_SAT> fixed;
namespace dnnk {
    
    
void affine(const fixed *x, const fixed *weight1, const fixed *weight2, const fixed *weight3, const fixed *weight4, const fixed* bias, int in_features, int out_features, fixed *y) {
       
        const int UNROLL_OCH=2;
for (int block_i = 0; block_i < out_features; block_i += UNROLL_OCH) {//
    fixed sum[UNROLL_OCH];
#pragma HLS array_partition variable=sum complete

    for (int j = 0; j < in_features; ++j) {
      for (int local_i = 0; local_i < UNROLL_OCH; local_i++) {
#pragma HLS unroll
        int i = block_i + local_i;
        if (i < out_features) {
          fixed last = (j == 0) ? (fixed)0 : sum[local_i];
          int w_in = i * in_features + j;
          if(w_in < 16384){
            sum[local_i] = last + x[j] * weight1[w_in];
          }
          else if(w_in < 32768){
            sum[local_i] = last + x[j] * weight2[w_in - 16384];
          }
          else if(w_in < 49152){
            sum[local_i] = last + x[j] * weight3[w_in - 32768];
          }
          else{
            sum[local_i] = last + x[j] * weight4[w_in - 49152];
          }
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
