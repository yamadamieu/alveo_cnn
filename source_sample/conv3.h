#ifndef DNNKERNEL_CONV3_H
#define DNNKERNEL_CONV3_H
#include <ap_fixed.h>
typedef ap_fixed<16, 7, AP_RND, AP_SAT> fixed;
namespace dnnk {


void conv3(const fixed* x, const fixed* weight, const fixed* bias, int width, int height,int in_channels, int out_channels, int ksize, fixed* y) {
    const int UNROLL_OCH = 8;
    const int UNROLL_X = 1;
    for (int block_och = 0; block_och < out_channels; block_och += UNROLL_OCH) {
    for (int h = 0; h < height; ++h) {
      for (int block_w = 0; block_w < width; block_w += UNROLL_X) {
        fixed sum[UNROLL_OCH][UNROLL_X];
#pragma HLS array_partition variable=sum complete dim=0

        for (int ich = 0; ich < in_channels; ++ich) {
          for (int kh = 0; kh < ksize; ++kh) {             
            for (int kw = 0; kw < ksize; ++kw) {
              for (int local_och = 0; local_och < UNROLL_OCH; local_och++) {
#pragma HLS unroll
                for (int local_w = 0; local_w < UNROLL_X; local_w++) {
#pragma HLS unroll
                  if (block_w + local_w < width && block_och + local_och < out_channels) {

                    int och = block_och + local_och;
                    int w = block_w + local_w;

                    int ph = h + kh - ksize/2;
                    int pw = w + kw - ksize/2;

                    fixed last = (ich == 0 && kh == 0 && kw == 0) ? (fixed)0 : sum[local_och][local_w];

                    // zero padding
                    if (ph < 0 || ph >= height || pw < 0 || pw >= width) {
                      sum[local_och][local_w] = last;
                      continue;
                    }

                    int pix_idx = (ich * height + ph) * width + pw;
                    int weight_idx = ((och * in_channels + ich) * ksize + kh) * ksize + kw;

                    sum[local_och][local_w] = last + x[pix_idx] * weight[weight_idx];
                  }
                }
              }
            }
          }
        }

        for (int local_och = 0; local_och < UNROLL_OCH; local_och++) {
#pragma HLS unroll
          for (int local_w = 0; local_w < UNROLL_X; local_w++) {
#pragma HLS unroll
            if (block_w + local_w < width && block_och + local_och < out_channels) {
              int och = block_och + local_och;
              int w = block_w + local_w;

              // add bias
              y[(och * height + h) * width + w] = sum[local_och][local_w] + bias[och];
            }
          }
        }
      }
    }
  }
}
}  // namespace dnnk

#endif  // DNNKERNEL_CONV_H



 