#ifndef DNNKERNEL_AFFINE_H
#define DNNKERNEL_AFFINE_H

namespace dnnk {

void affine(const float *x, const float* weight, const float* bias,
            int in_features, int out_features, float *y) {
  for (int i = 0; i < out_features; ++i) {//出力の特徴量で回す(高さ決める)
    float sum = 0.f;
    for (int j = 0; j < in_features; ++j) {//入力の特徴量で回す(横幅決める)
      sum += x[j] * weight[i * in_features + j];
    }
    y[i] = sum + bias[i];
  }
}


}  // namespace dnnk

#endif  // DNNKERNEL_AFFINE_H

