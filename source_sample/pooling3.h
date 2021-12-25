#ifndef DNNKERNEL_POOL3_H
#define DNNKERNEL_POOL3_H

#include <stdint.h>
#include <float.h>
#include <ap_fixed.h>
typedef ap_fixed<16, 7, AP_RND, AP_SAT> fixed;
namespace dnnk {


void pooling3(const fixed *x, int width, int height, int channels, int stride, fixed *y) {
  for (int ch = 0; ch < channels; ++ch){
    for (int h = 0; h < height; h += stride) {
      for (int w = 0; w < width; w += stride) {
      	fixed maxval = (fixed)-FLT_MAX;

        for (int bh = 0; bh < stride; ++bh) {
          for (int bw = 0; bw < stride; ++bw) {
          	fixed st = x[(ch * height + h + bh) * width + w + bw];
          	if (maxval < st){
          		maxval = st;
          	}
          	else{
     			;
          	}
          }
        }

        y[(ch * (height / stride) + (h / stride)) * (width / stride) + w / stride] = maxval;
      }
    }
  }
}

}  // namespace dnnk

#endif  // DNNKERNEL_POOL_H