#ifndef DNNKERNEL_TRANOUT_H
#define DNNKERNEL_TRANOUT_H

#include <ap_fixed.h>
typedef ap_fixed<16, 7, AP_RND, AP_SAT> fixed;
namespace dnnk {
void tranout(fixed *y_local, float *y) {   
    for(int i=0; i<10; i++){
    y[i] = y_local[i].to_float();
    }
}
}  // namespace dnnk
#endif  // DNNKERNEL_Y_H