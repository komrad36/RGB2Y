Fastest CPU (AVX/SSE) implementation of RGB to grayscale.
Roughly 3x faster than OpenCV's implementation with AVX2, or 2x faster
than OpenCV's implementation if using SSE only.

Converts an RGB color image to grayscale.

You can use equal weighting by calling the templated
function with weight set to 'false', or you
can specify custom weights in RGB2Y.h (only slightly slower).

The default weights match OpenCV's default.

For even more speed see the CUDA version:
github.com/komrad36/CUDARGB2Y

If you do not have AVX2, uncomment the '#define NO_AVX_PLEASE' in RGB2Y.h to route the code
through SSE isntructions only. NOTE THAT THIS IS ABOUT 50% SLOWER.
A processor with full AVX2 support is highly recommended.

All functionality is contained in RGB2Y.h.
'main.cpp' is a demo and test harness.