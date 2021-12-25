#ifndef DNNKERNEL_TRANOUT0_H
#define DNNKERNEL_TRANOUT0_H

#include <ap_fixed.h>
typedef ap_fixed<16, 7, AP_RND, AP_SAT> fixed;
namespace dnnk {
void tranout0(float *y_local, float *y,int yin) {   
    for(int i=0; i<10; i++){
    y[10*yin+i] = y_local[i];
    }
}
}  // namespace dnnk
#endif  // DNNKERNEL_Y_H