/*******************************************************************
*   RGB2Y.h
*   RGB2Y
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Oct 27, 2016
*******************************************************************/
//
// Fastest CPU (AVX/SSE) implementation of RGB to grayscale.
// Roughly 2.5x to 3x faster than OpenCV's implementation.
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
// All functionality is contained in RGB2Y.h.
// 'main.cpp' is a demo and test harness.
//

#pragma once

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

constexpr uint16_t mask[] { B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT };

template<const bool last_row_and_col, bool weight>
void process(const uint8_t* __restrict const pt, const int32_t cols_minus_j, uint8_t* const __restrict out) {
	__m256i h2;
	if (weight) {
		__m256i p1 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt))), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask)));
		__m256i p2 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 1))), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask + 1)));
		__m256i p3 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 2))), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask + 2)));

		__m256i p1p = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 16))), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask + 1)));
		__m256i p2p = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 17))), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask + 2)));
		__m256i p3p = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 18))), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask)));

		h2 = _mm256_shuffle_epi8(_mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_srli_epi16(_mm256_add_epi16(p3, _mm256_add_epi16(p1, p2)), 6), _mm256_srli_epi16(_mm256_add_epi16(p3p, _mm256_add_epi16(p1p, p2p)), 6)), 0b11011000), _mm256_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 21, 24, 27, 30, -1, -1, -1, -1, -1));
	}
	else {
		h2 = _mm256_shuffle_epi8(_mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(_mm256_add_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 2))), _mm256_add_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt))), _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 1))))), _mm256_set1_epi16(85)), 8), _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_add_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 18))), _mm256_add_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 16))), _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pt + 17))))), _mm256_set1_epi16(85)), 8)), 0b11011000), _mm256_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 21, 24, 27, 30, -1, -1, -1, -1, -1));
	}
	__m128i h3 = _mm_blend_epi16(_mm256_castsi256_si128(h2), _mm256_extracti128_si256(h2, 1), 0b11111000);
	if (last_row_and_col) {
		switch (cols_minus_j) {
		case 11:
			out[10] = _mm_extract_epi8(h3, 10);
		case 10:
			out[9] = _mm_extract_epi8(h3, 9);
		case 9:
			out[8] = _mm_extract_epi8(h3, 8);
		case 8:
			out[7] = _mm_extract_epi8(h3, 7);
		case 7:
			out[6] = _mm_extract_epi8(h3, 6);
		case 6:
			out[5] = _mm_extract_epi8(h3, 5);
		case 5:
			out[4] = _mm_extract_epi8(h3, 4);
		case 4:
			out[3] = _mm_extract_epi8(h3, 3);
		case 3:
			out[2] = _mm_extract_epi8(h3, 2);
		case 2:
			out[1] = _mm_extract_epi8(h3, 1);
		case 1:
			out[0] = _mm_extract_epi8(h3, 0);
		}
	}
	else {
		_mm_storeu_si128(reinterpret_cast<__m128i*>(out), h3);
	}
}

template<bool last_row, bool weight>
void processRow(const uint8_t* __restrict pt, const int32_t cols, uint8_t* const __restrict out) {
	int j = 0;
	for (; j < cols - 11; j += 11, pt += 33) {
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

