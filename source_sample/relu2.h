#ifndef DNNKERNEL_RELU2_H
#define DNNKERNEL_RELU2_H

#include <stdint.h>
#include <algorithm>
#include <ap_fixed.h>
typedef ap_fixed<16, 7, AP_RND, AP_SAT> fixed;
namespace dnnk {

void relu2(const fixed *x, int size, fixed *y) {
   const fixed zero=0; 
  for (int i = 0; i < size; ++i) {
    if(x[i] < zero){  
    y[i] = zero;
    }
    else{
    y[i] = x[i];
    }
  }
}
}  // namespace dnnk

#endif  // DNNKERNEL_RELU_H
