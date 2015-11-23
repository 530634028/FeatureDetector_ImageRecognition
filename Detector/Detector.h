/*
 *
 *
 *
 *
 *
*/

#ifndef DETECTOR_H
#define DETECTOR_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"

const int nOctaveLayers = 2;
const double sigma = 0.1;

using namespace std;
using namespace cv;

class Detector
{
public:
	Detector(Mat &I) : image(I) {};
	~Detector() { };
	void Lapls(const Mat &, vector<Point2d> &);
	void Harris(const Mat &, vector<Point2d> &);
	void OrientativeFliter(const Mat &, vector<Point2d> &);
	Mat &ReadImage(const string &);
	void Display(const Mat &, vector<Point2d> &);
	void WriteImage(const Mat &, string &);

private:
	vector<int> points;
	Mat image;
};

void Detector::Lapls(const Mat &image, vector<Point2d> &points)
{
	Mat laplsCal = Mat::zeros(3, 3, CV_64F);

}


// 构建nOctaves组（每组nOctaves+3层）高斯金字塔  
void buildGaussianPyramid(const Mat& base, vector<Mat>& pyr, int nOctaves)
{
	vector<double> sig(nOctaveLayers + 3);
	pyr.resize(nOctaves*(nOctaveLayers + 3));

	// precompute Gaussian sigmas using the following formula:  
	//  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2、  
	// 计算对图像做不同尺度高斯模糊的尺度因子  
	sig[0] = sigma;
	double k = pow(2., 1. / nOctaveLayers);
	for (int i = 1; i < nOctaveLayers + 3; i++)
	{
		double sig_prev = pow(k, (double)(i - 1))*sigma;
		double sig_total = sig_prev*k;
		sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
	}

	for (int o = 0; o < nOctaves; o++)
	{
		// DoG金子塔需要nOctaveLayers+2层图像来检测nOctaves层尺度  
		// 所以高斯金字塔需要nOctaveLayers+3层图像得到nOctaveLayers+2层DoG金字塔  
		for (int i = 0; i < nOctaveLayers + 3; i++)
		{
			// dst为第o组（Octave）金字塔  
			Mat& dst = pyr[o*(nOctaveLayers + 3) + i];
			// 第0组第0层为原始图像  
			if (o == 0 && i == 0)
				dst = base;

			// base of new octave is halved image from end of previous octave  
			// 每一组第0副图像时上一组倒数第三幅图像隔点采样得到  
			else if (i == 0)
			{
				const Mat& src = pyr[(o - 1)*(nOctaveLayers + 3) + nOctaveLayers];
				resize(src, dst, Size(src.cols / 2, src.rows / 2),
					0, 0, INTER_NEAREST);
			}
			// 每一组第i副图像是由第i-1副图像进行sig[i]的高斯模糊得到  
			// 也就是本组图像在sig[i]的尺度空间下的图像  
			else
			{
				const Mat& src = pyr[o*(nOctaveLayers + 3) + i - 1];
				GaussianBlur(src, dst, Size(), sig[i], sig[i]);
			}
		}
	}
}


#endif