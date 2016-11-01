/*******************************************************************
*   main.cpp
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

#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "RGB2Y.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

using namespace std::chrono;

void printReport(const std::string& name, const nanoseconds& dur, const nanoseconds& comp = nanoseconds(0)) {
	std::cout << std::left << std::setprecision(6) << std::setw(10) << name << " took " << std::setw(7) << static_cast<double>(dur.count()) * 1e-3 << " us";
	if (comp.count() && comp < dur) {
		std::cout << " (" << std::setprecision(4) << std::setw(4) << static_cast<double>(dur.count()) / comp.count() << "x slower than RGB2Y)." << std::endl;
	}
	else {
		std::cout << '.' << std::endl;
	}
}

void RGB2Y_ref(const uint8_t* __restrict const data, const int32_t cols, const int32_t rows, const int32_t stride, uint8_t* const __restrict out) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			const auto idx = 3 * (i*stride + j);
			out[i*stride + j] = (static_cast<uint32_t>(data[idx]) + static_cast<uint32_t>(data[idx + 1]) + static_cast<uint32_t>(data[idx + 2])) / 3;
		}
	}
}


int main() {
	// ------------- Configuration ------------
	constexpr bool display_image = false;
	constexpr auto warmups = 200;
	constexpr auto runs = 500;
	constexpr bool multithread = true;
	constexpr bool weighted_averaging = false;
	constexpr char name[] = "test.jpg";
	// --------------------------------


	// ------------- Image Read ------------
	cv::Mat image = cv::imread(name);
	if (!image.data) {
		std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
		return EXIT_FAILURE;
	}
	// --------------------------------

	
	// ------------- Ref ------------
	uint8_t* refresult = new uint8_t[image.cols*image.rows];
	RGB2Y_ref(image.data, image.cols, image.rows, image.cols, refresult);
	// --------------------------------


	// ------------- RGB2Y ------------
	nanoseconds RGB2Y_ns;
	uint8_t* newimage = new uint8_t[image.cols*image.rows];
	std::cout << std::endl << "------------- RGB2Y ------------" << std::endl << "Warming up..." << std::endl;
	for (int i = 0; i < warmups; ++i) RGB2Y<multithread, weighted_averaging>(image.data, image.cols, image.rows, image.cols, newimage);
	std::cout << "Testing..." << std::endl;
	{
		const high_resolution_clock::time_point start = high_resolution_clock::now();
		for (int32_t i = 0; i < runs; ++i) {
			RGB2Y<multithread, weighted_averaging>(image.data, image.cols, image.rows, image.cols, newimage);
		}
		const high_resolution_clock::time_point end = high_resolution_clock::now();
		RGB2Y_ns = (end - start) / runs;
	}
	// --------------------------------


	// ------------- OpenCV ------------
	nanoseconds CV_ns;
	cv::Mat newimage_cv;
	std::cout << "------------ OpenCV ------------" << std::endl << "Warming up..." << std::endl;
	for (int i = 0; i < warmups; ++i) cv::cvtColor(image, newimage_cv, CV_BGR2GRAY);
	std::cout << "Testing..." << std::endl;
	{
		const high_resolution_clock::time_point start = high_resolution_clock::now();
		for (int32_t i = 0; i < runs; ++i) {
			cv::cvtColor(image, newimage_cv, CV_BGR2GRAY);
		}
		const high_resolution_clock::time_point end = high_resolution_clock::now();
		CV_ns = (end - start) / runs;
	}
	// --------------------------------


	// ------------- Verification ------------
	int i = 0;
	for (; i < image.cols*image.rows; ++i) {
		if (abs(newimage[i] - (weighted_averaging ? newimage_cv.data[i] : refresult[i])) > 1) {
			std::cerr << "ERROR! One or more pixels disagree!" << std::endl;
			std::cerr << i << ": got " << +newimage[i] << ", should be " << (weighted_averaging ? +newimage_cv.data[i] : +refresult[i]) << std::endl;
			break;
		}
	}
	if (i == image.cols*image.rows) std::cout << "All pixels agree! Test valid." << std::endl << std::endl;
	// --------------------------------


	// ------------- Output ------------
	printReport("RGB2Y", RGB2Y_ns);
	printReport("OpenCV", CV_ns, RGB2Y_ns);
	std::cout << std::endl;
	if (display_image) {
		cv::namedWindow("RGB2Y", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		cv::imshow("RGB2Y", cv::Mat(image.rows, image.cols, CV_8U, newimage));
		cv::namedWindow("OpenCV", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		cv::imshow("OpenCV", newimage_cv);
		cv::waitKey(0);
	}
	// --------------------------------
}
