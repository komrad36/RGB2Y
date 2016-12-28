/*******************************************************************
*   RGB2Y.h
*   RGB2Y
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Dec 9, 2016
*******************************************************************/
//
// Fastest CPU (AVX/SSE) implementation of RGB to grayscale.
// Roughly 3x faster than OpenCV's implementation with AVX2, or 2x faster
// than OpenCV's implementation if using SSE only.
//
// Converts an RGB color image to grayscale.
//
// You can use equal weighting by calling the templated
// function with weight set to 'false', or you
// can specify custom weights in RGB2Y.h (only slightly slower).
//
// The default weights match OpenCV's default.
//
// For even more speed see the CUDA version:
// github.com/komrad36/CUDARGB2Y
//
// If you do not have AVX2, uncomment the #define below to route the code
// through only SSE isntructions. NOTE THAT THIS IS ABOUT 50% SLOWER.
// A processor with full AVX2 support is highly recommended.
//
// All functionality is contained in RGB2Y.h.
// 'main.cpp' is a demo and test harness.
//

#pragma once

//#define NO_AVX_PLEASE

// Set your weights here.
constexpr double B_WEIGHT = 0.114;
constexpr double G_WEIGHT = 0.587;
constexpr double R_WEIGHT = 0.299;


// Internal; do NOT modify
constexpr uint16_t B_WT = static_cast<uint16_t>(64.0 * B_WEIGHT + 0.5);
constexpr uint16_t G_WT = static_cast<uint16_t>(64.0 * G_WEIGHT + 0.5);
constexpr uint16_t R_WT = static_cast<uint16_t>(64.0 * R_WEIGHT + 0.5);

#include <algorithm>
#include <cstdint>
#include <future>
#include <immintrin.h>

// 241
template<const bool last_row_and_col, bool weight>
void process(const uint8_t* __restrict const pt, const int32_t cols_minus_j, uint8_t* const __restrict out) {
	__m128i h3;
	if (weight) {
#ifdef NO_AVX_PLEASE
		__m128i p1aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
		__m128i p1aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 8))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));
		__m128i p1bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 18))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
		__m128i p1bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 26))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));
		__m128i p2aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 1))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));
		__m128i p2aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 9))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
		__m128i p2bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 19))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));
		__m128i p2bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 27))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
		__m128i p3aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 2))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));
		__m128i p3aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 10))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));
		__m128i p3bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 20))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));
		__m128i p3bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 28))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));
		__m128i sumaL = _mm_add_epi16(p3aL, _mm_add_epi16(p1aL, p2aL));
		__m128i sumaH = _mm_add_epi16(p3aH, _mm_add_epi16(p1aH, p2aH));
		__m128i sumbL = _mm_add_epi16(p3bL, _mm_add_epi16(p1bL, p2bL));
		__m128i sumbH = _mm_add_epi16(p3bH, _mm_add_epi16(p1bH, p2bH));
		__m128i sclaL = _mm_srli_epi16(sumaL, 6);
		__m128i sclaH = _mm_srli_epi16(sumaH, 6);
		__m128i sclbL = _mm_srli_epi16(sumbL, 6);
		__m128i sclbH = _mm_srli_epi16(sumbH, 6);
		__m128i shftaL = _mm_shuffle_epi8(sclaL, _mm_setr_epi8(0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
		__m128i shftaH = _mm_shuffle_epi8(sclaH, _mm_setr_epi8(-1, -1, -1, 18, 24, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
		__m128i shftbL = _mm_shuffle_epi8(sclbL, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 6, 12, -1, -1, -1, -1, -1, -1, -1));
		__m128i shftbH = _mm_shuffle_epi8(sclbH, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 24, 30, -1, -1, -1, -1));
		__m128i accumL = _mm_or_si128(shftaL, shftbL);
		__m128i accumH = _mm_or_si128(shftaH, shftbH);
		h3 = _mm_blendv_epi8(accumL, accumH, _mm_setr_epi8(0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 1));
#else
		__m256i p1a = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt))), _mm256_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));
		__m256i p1b = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 18))), _mm256_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));
		__m256i p2a = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 1))), _mm256_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
		__m256i p2b = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 19))), _mm256_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
		__m256i p3a = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 2))), _mm256_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));
		__m256i p3b = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 20))), _mm256_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));
		__m256i suma = _mm256_add_epi16(p3a, _mm256_add_epi16(p1a, p2a));
		__m256i sumb = _mm256_add_epi16(p3b, _mm256_add_epi16(p1b, p2b));
		__m256i scla = _mm256_srli_epi16(suma, 6);
		__m256i sclb = _mm256_srli_epi16(sumb, 6);
		__m256i shfta = _mm256_shuffle_epi8(scla, _mm256_setr_epi8(0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 24, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
		__m256i shftb = _mm256_shuffle_epi8(sclb, _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 24, 30, -1, -1, -1, -1));
		__m256i accum = _mm256_or_si256(shfta, shftb);
		h3 = _mm_blendv_epi8(_mm256_castsi256_si128(accum), _mm256_extracti128_si256(accum, 1), _mm_setr_epi8(0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 1));
#endif
	}
	else {
#ifdef NO_AVX_PLEASE
		__m128i p1aL = _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt)));
		__m128i p1aH = _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 8)));
		__m128i p1bL = _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 18)));
		__m128i p1bH = _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 26)));
		__m128i p2aL = _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 1)));
		__m128i p2aH = _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 9)));
		__m128i p2bL = _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 19)));
		__m128i p2bH = _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 27)));
		__m128i p3aL = _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 2)));
		__m128i p3aH = _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 10)));
		__m128i p3bL = _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 20)));
		__m128i p3bH = _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 28)));
		__m128i sumaL = _mm_add_epi16(p3aL, _mm_add_epi16(p1aL, p2aL));
		__m128i sumaH = _mm_add_epi16(p3aH, _mm_add_epi16(p1aH, p2aH));
		__m128i sumbL = _mm_add_epi16(p3bL, _mm_add_epi16(p1bL, p2bL));
		__m128i sumbH = _mm_add_epi16(p3bH, _mm_add_epi16(p1bH, p2bH));
		__m128i sclaL = _mm_srli_epi16(_mm_mullo_epi16(sumaL, _mm_set1_epi16(85)), 8);
		__m128i sclaH = _mm_srli_epi16(_mm_mullo_epi16(sumaH, _mm_set1_epi16(85)), 8);
		__m128i sclbL = _mm_srli_epi16(_mm_mullo_epi16(sumbL, _mm_set1_epi16(85)), 8);
		__m128i sclbH = _mm_srli_epi16(_mm_mullo_epi16(sumbH, _mm_set1_epi16(85)), 8);
		__m128i shftaL = _mm_shuffle_epi8(sclaL, _mm_setr_epi8(0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
		__m128i shftaH = _mm_shuffle_epi8(sclaH, _mm_setr_epi8(-1, -1, -1, 18, 24, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
		__m128i shftbL = _mm_shuffle_epi8(sclbL, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 6, 12, -1, -1, -1, -1, -1, -1, -1));
		__m128i shftbH = _mm_shuffle_epi8(sclbH, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 24, 30, -1, -1, -1, -1));
		__m128i accumL = _mm_or_si128(shftaL, shftbL);
		__m128i accumH = _mm_or_si128(shftaH, shftbH);
		h3 = _mm_blendv_epi8(accumL, accumH, _mm_setr_epi8(0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 1));
#else
		__m256i p1a = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt)));
		__m256i p1b = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 18)));
		__m256i p2a = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 1)));
		__m256i p2b = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 19)));
		__m256i p3a = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 2)));
		__m256i p3b = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 20)));
		__m256i suma = _mm256_add_epi16(p3a, _mm256_add_epi16(p1a, p2a));
		__m256i sumb = _mm256_add_epi16(p3b, _mm256_add_epi16(p1b, p2b));
		__m256i scla = _mm256_srli_epi16(_mm256_mullo_epi16(suma, _mm256_set1_epi16(85)), 8);
		__m256i sclb = _mm256_srli_epi16(_mm256_mullo_epi16(sumb, _mm256_set1_epi16(85)), 8);
		__m256i shfta = _mm256_shuffle_epi8(scla, _mm256_setr_epi8(0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 24, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
		__m256i shftb = _mm256_shuffle_epi8(sclb, _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 24, 30, -1, -1, -1, -1));
		__m256i accum = _mm256_or_si256(shfta, shftb);
		h3 = _mm_blendv_epi8(_mm256_castsi256_si128(accum), _mm256_extracti128_si256(accum, 1), _mm_setr_epi8(0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 1));
#endif
	}
	if (last_row_and_col) {
		switch (cols_minus_j) {
		case 12:
			out[11] = static_cast<uint8_t>(_mm_extract_epi8(h3, 11));
		case 11:
			out[10] = static_cast<uint8_t>(_mm_extract_epi8(h3, 10));
		case 10:
			out[9] = static_cast<uint8_t>(_mm_extract_epi8(h3, 9));
		case 9:
			out[8] = static_cast<uint8_t>(_mm_extract_epi8(h3, 8));
		case 8:
			out[7] = static_cast<uint8_t>(_mm_extract_epi8(h3, 7));
		case 7:
			out[6] = static_cast<uint8_t>(_mm_extract_epi8(h3, 6));
		case 6:
			out[5] = static_cast<uint8_t>(_mm_extract_epi8(h3, 5));
		case 5:
			out[4] = static_cast<uint8_t>(_mm_extract_epi8(h3, 4));
		case 4:
			out[3] = static_cast<uint8_t>(_mm_extract_epi8(h3, 3));
		case 3:
			out[2] = static_cast<uint8_t>(_mm_extract_epi8(h3, 2));
		case 2:
			out[1] = static_cast<uint8_t>(_mm_extract_epi8(h3, 1));
		case 1:
			out[0] = static_cast<uint8_t>(_mm_extract_epi8(h3, 0));
		}
	}
	else {
		_mm_storeu_si128(reinterpret_cast<__m128i*>(out), h3);
	}
}

template<bool last_row, bool weight>
void processRow(const uint8_t* __restrict pt, const int32_t cols, uint8_t* const __restrict out) {
	int j = 0;
	for (; j < cols - 12; j += 12, pt += 36) {
		process<false, weight>(pt, cols - j, out + j);
	}
	process<last_row, weight>(pt, cols - j, out + j);
}

template<bool weight>
void _RGB2Y(const uint8_t* __restrict const data, const int32_t cols, const int32_t start_row, const int32_t rows, const int32_t stride, uint8_t* const __restrict out) {
	int i = start_row;
	for (; i < start_row + rows - 1; ++i) {
		processRow<false, weight>(data + 3 * i * stride, cols, out + i*cols);
	}
	processRow<true, weight>(data + 3 * i * stride, cols, out + i*cols);
}

template<bool multithread, bool weight>
void RGB2Y(const uint8_t* const __restrict image, const int width, const int height, const int stride, uint8_t* const __restrict out) {
	if (multithread) {
		const int32_t hw_concur = std::min(height >> 4, static_cast<int32_t>(std::thread::hardware_concurrency()));
		if (hw_concur > 1) {
			std::vector<std::future<void>> fut(hw_concur);
			const int thread_stride = (height - 1) / hw_concur + 1;
			int i = 0, start = 0;
			for (; i < std::min(height - 1, hw_concur - 1); ++i, start += thread_stride) {
				fut[i] = std::async(std::launch::async, _RGB2Y<weight>, image, width, start, thread_stride, stride, out);
			}	
			fut[i] = std::async(std::launch::async, _RGB2Y<weight>, image, width, start, height - start, stride, out);
			for (int j = 0; j <= i; ++j) fut[j].wait();
		}
		else {
			_RGB2Y<weight>(image, width, 0, height, stride, out);
		}
	}
	else {
		_RGB2Y<weight>(image, width, 0, height, stride, out);
	}
}

