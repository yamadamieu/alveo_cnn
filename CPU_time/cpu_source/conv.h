#ifndef DNNKERNEL_CONV_H
#define DNNKERNEL_CONV_H

namespace dnnk {


void conv(const float* x, const float* weight, const float* bias, int width, int height,int in_channels, int out_channels, int ksize, float* y) {
  for (int och = 0; och < out_channels; ++och) {//出力チャネルごとにループ
    for (int h = 0; h < height; ++h) {//高さごとにループ
      for (int w = 0; w < width; ++w) {//幅ごとにループ
        //以上で出力画像上の位置を決定した
        float sum = 0.f;

        for (int ich = 0; ich < in_channels; ++ich) {//入力チャネルごとにループ
          for (int kh = 0; kh < ksize; ++kh) {//カーネルサイズごとにループ
            for (int kw = 0; kw < ksize; ++kw) {//カーネルサイズごとにループ
              int ph = h + kh - ksize/2;//高さ+カーネル高さ-カーネルサイズ/2
              int pw = w + kw - ksize/2;//幅+カーネル幅-カーネルサイズ/2
              if (ph < 0 || ph >= height || pw < 0 || pw >= width) {
                continue;
              }
              
              int pix_idx = (ich * height + ph) * width + pw;//入力データの演算対象の場所を決める
              int weight_idx = ((och * in_channels + ich) * ksize + kh) * ksize + kw;//重みの演算対象の場所を決める

              sum += x[pix_idx] * weight[weight_idx];
              
            }
          }
        }

        // add bias
        sum += bias[och];

        y[(och * height + h) * width + w] = sum;
      }
    }
  }
}
}  // namespace dnnk

#endif  // DNNKERNEL_CONV_H



 