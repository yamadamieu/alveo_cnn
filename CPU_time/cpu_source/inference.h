#ifndef DNNKERNEL_INFERENCE_H
#define DNNKERNEL_INFERENCE_H

#include "conv.h"
#include "pooling.h"
#include "relu.h"
#include "affine.h"

#include <stdint.h>
#include <algorithm>

namespace dnnk {

void inference(const float* x,
                             const float* weight1, const float* bias1,
                             const float* weight2, const float* bias2,
                             const float* weight3, const float* bias3,
                             const float* weight4, const float* bias4,
                             const float* weight5, const float* bias5,
                             const float* weight6, const float* bias6,
                             float* y) {
    
  static const int kWidths[] = {32, 16, 8, 4, 2};
  static const int kHeights[] = {32, 16, 8, 4, 2};
  static const int kChannels[] = {3, 32, 64, 256, 10};

  float x1[kWidths[0] * kHeights[0] * kChannels[1]];
  float x2[kWidths[0] * kHeights[0] * kChannels[1]];
  float x3[kWidths[1] * kHeights[1] * kChannels[1]];
  float x4[kWidths[1] * kHeights[1] * kChannels[1]];
  float x5[kWidths[1] * kHeights[1] * kChannels[1]];
  float x6[kWidths[2] * kHeights[2] * kChannels[1]];
    
  float x7[kWidths[2] * kHeights[2] * kChannels[2]];
  float x8[kWidths[2] * kHeights[2] * kChannels[2]];
  float x9[kWidths[3] * kHeights[3] * kChannels[2]];
  float x10[kWidths[3] * kHeights[3] * kChannels[2]];
  float x11[kWidths[3] * kHeights[3] * kChannels[2]];
  float x12[kWidths[4] * kHeights[4] * kChannels[2]]; 
  
  float x13[kChannels[3]];
  float x14[kChannels[3]];

  // 1st layer
  dnnk::conv(x, weight1, bias1, 32, 32, 3, 32, 3, x1);
  dnnk::relu(x1, 32 * 32 * 32, x2);
  dnnk::pooling(x2, 32, 32, 32, 2, x3);

  // 2nd layer
  dnnk::conv(x3, weight2, bias2, 16, 16, 32, 32, 3, x4);
  dnnk::relu(x4, 16 * 16 * 32, x5);
  dnnk::pooling(x5, 16, 16, 32, 2, x6);
    
    // 3st layer
  dnnk::conv(x6, weight3, bias3, 8, 8, 32, 64, 3, x7);
  dnnk::relu(x7, 8 * 8 * 64, x8);
  dnnk::pooling(x8, 8, 8, 64, 2, x9);

  // 4nd layer
  dnnk::conv(x9, weight4, bias4, 4, 4, 64, 64, 3, x10);
  dnnk::relu(x10, 4 * 4 * 64, x11);
  dnnk::pooling(x11, 4, 4, 64, 2, x12);

  // 5rd layer
  dnnk::affine(x12, weight5, bias5, 2 * 2 * 64, 256, x13);
  dnnk::relu(x13, 256, x14);

  // 6th layer
  dnnk::affine(x14, weight6, bias6, 256, 10, y);
    
}
}

#endif  // DNNKERNEL_INFERENCE_H
