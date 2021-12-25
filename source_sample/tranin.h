#ifndef DNNKERNEL_TRANIN_H
#define DNNKERNEL_TRANIN_H

#include <stdint.h>
#include <algorithm>
#include <ap_fixed.h>
typedef ap_fixed<16, 7, AP_RND, AP_SAT> fixed;
namespace dnnk {

void tranin(const float *x, fixed *x_local) {
    for(int i=0; i<3072; i++){
    x_local[i] = (fixed)x[i];
    }
}
}  // namespace dnnk

#endif  // DNNKERNEL_X_H