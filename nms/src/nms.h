
#include <ATen/ATen.h>
#include <THC/THC.h>
#include <TH/TH.h>
#include <math.h>
#include <stdio.h>

int cpu_nms(THLongTensor * keep_out, THLongTensor * num_out, THFloatTensor * boxes, THLongTensor * order, THFloatTensor * areas, float nms_overlap_thresh);
int gpu_nms(THLongTensor * keep_out, THLongTensor* num_out, THCudaTensor * boxes, float nms_overlap_thresh);
