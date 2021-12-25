#ifndef DNNKERNEL_POOL_H
#define DNNKERNEL_POOL_H

#include <stdint.h>
#include <float.h>

namespace dnnk {


void pooling(const float *x, int width, int height, int channels, int stride, float *y) {
  for (int ch = 0; ch < channels; ++ch){
    for (int h = 0; h < height; h += stride) {
      for (int w = 0; w < width; w += stride) {
      	float maxval = -FLT_MAX;

        for (int bh = 0; bh < stride; ++bh) {
          for (int bw = 0; bw < stride; ++bw) {
          	float st = x[(ch * height + h + bh) * width + w + bw];
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