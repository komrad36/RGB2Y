Fastest CPU (AVX/SSE) implementation of RGB to grayscale.
Roughly 2.5x to 3x faster than OpenCV's implementation.

You can use equal weighting by calling the templated
function with weight set to 'false', or you
can specify custom weights in RGB2Y.h (only slightly slower).

The default weights match OpenCV's default.

For even more speed see the CUDA version:
github.com/komrad36/CUDARGB2Y

All functionality is contained in RGB2Y.h.
'main.cpp' is a demo and test harness.