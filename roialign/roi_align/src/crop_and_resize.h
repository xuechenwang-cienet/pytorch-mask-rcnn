#include <TH/TH.h>
#include <THC/THC.h>
#include <stdio.h>
#include <math.h>


void crop_and_resize_forward(
    THFloatTensor * image,
    THFloatTensor * boxes,      // [y1, x1, y2, x2]
    THIntTensor * box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    THFloatTensor * crops
);

void crop_and_resize_backward(
    THFloatTensor * grads,
    THFloatTensor * boxes,      // [y1, x1, y2, x2]
    THIntTensor * box_index,    // range in [0, batch_size)
    THFloatTensor * grads_image // resize to [bsize, c, hc, wc]
);

void crop_and_resize_gpu_forward(
    THCudaTensor * image,
    THCudaTensor * boxes,           // [y1, x1, y2, x2]
    THCudaIntTensor * box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    THCudaTensor * crops
);

void crop_and_resize_gpu_backward(
    THCudaTensor * grads,
    THCudaTensor * boxes,      // [y1, x1, y2, x2]
    THCudaIntTensor * box_index,    // range in [0, batch_size)
    THCudaTensor * grads_image // resize to [bsize, c, hc, wc]
);