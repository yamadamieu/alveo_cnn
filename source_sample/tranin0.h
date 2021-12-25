#ifndef DNNKERNEL_TRANIN0_H
#define DNNKERNEL_TRANIN0_H

#include <stdint.h>
#include <algorithm>
#include <ap_fixed.h>
typedef ap_fixed<16, 7, AP_RND, AP_SAT> fixed;
namespace dnnk {

void tranin0(const float *x, float *x_local,int xin) {
    for(int i=0; i<3072; i++){
    x_local[i] = x[3072*xin+i];
    }
}
}  // namespace dnnk

#endif  // DNNKERNEL_X_H