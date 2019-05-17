#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/features2d.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

const int THRESHOLD_FAST = 50;
const int THRESHOLD_HARRIS = 175;
const int PIXEL_EPS = 3;
const int eps = PIXEL_EPS * PIXEL_EPS;

void DrawPoints(cv::Mat& img, std::vector<cv::KeyPoint>& kpts) {
	for (int i = 0; i < kpts.size(); ++i) {
		cv::Point point = kpts[i].pt;
		//std::cout << "X: " << point.x << "    Y: " << point.y << '\n';
		cv::circle(img, point, 2, cv::Scalar(0, 0, 255), -1, 8, 0);
	}
}



void CornerDetector(std::string type, cv::Mat& img, cv::Mat& img_out, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {
	if (type == "orb") {
		cv::Ptr<cv::ORB> orb = cv::ORB::create();
		orb->detectAndCompute(img, cv::Mat(), kpts, desc);
	}
	if (type == "fast") {
		cv::FAST(img, kpts, THRESHOLD_FAST, true);
	}
	if (type == "brisk") {
		cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
		brisk->detectAndCompute(img, cv::Mat(), kpts, desc);
	}
	DrawPoints(img_out, kpts);
	std::cout << type << '\n' << kpts.size() << '\n';
}


int len(cv::Point& point1, cv::Point& point2) {
	int l = (point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y);
	return l;
}

bool len2(cv::Point& point1, cv::Point& point2) {
	int l1 = point1.x - point2.x;
	int l2 = point1.y - point2.y;
	return (std::abs(l1) < PIXEL_EPS && std::abs(l2) < PIXEL_EPS);
}

void merge(std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2, std::vector<cv::KeyPoint>& kpts_merge) {
	//std::cout << "keypoints count 1: " << kpts1.size() << '\n' << "keypoints count 2: " << kpts2.size() << '\n';


	for (int i = 0; i < kpts1.size(); ++i) {
		cv::Point point1 = kpts1[i].pt;
		for (int j = 0; j < kpts2.size(); ++j) {
			cv::Point point2 = kpts2[j].pt;
			if (len2(point1, point2)) {
				//std::cout << i << " " << j << '\n' << "keypoint1: " << point1.x << ' ' << point1.y << ' ' << "keypoint2: " << point2.x << ' ' << point2.y <<'\n';
				kpts_merge.push_back(kpts1[i]);
				break;
			}
		}
	}
	std::cout << kpts_merge.size();
}


int main() {
	cv::Mat img, gray, thresh, img_fast, img_orb, desc_orb, img_merge, img_brisk, desc_brisk;
	std::vector<cv::KeyPoint> kpts_fast, kpts_orb, kpts_merge, kpts_brisk;

	img = cv::imread("./imgs/Number.jpg");
	img.copyTo(img_fast);
	img.copyTo(img_orb);
	img.copyTo(img_merge);
	img.copyTo(img_brisk);

	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	cv::threshold(gray, thresh, THRESHOLD_HARRIS, 255, cv::THRESH_BINARY);

	CornerDetector("orb", gray, img_orb, kpts_orb, desc_orb);
	CornerDetector("fast", gray, img_fast, kpts_fast, desc_orb);
	CornerDetector("brisk", gray, img_brisk, kpts_brisk, desc_brisk);

	cv::imshow("ORB",img_orb);
	cv::imshow("FAST", img_fast);
	cv::imshow("BRISK", img_brisk);

	cv::imwrite("./results/ORB.jpg", img);

	merge(kpts_brisk, kpts_orb, kpts_merge);
	
	DrawPoints(img_merge, kpts_merge);
	cv::imshow("MERGE", img_merge);
	
	cv::waitKey(0);

}

//cv::Mat harris(cv::Mat img) {
//	cv::Mat dst;
////	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
//	cv::cornerHarris(img, dst,2,3,0.04);
//	cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_32FC1);
//	return dst;
//}

//	dst = harris(gray);
//	for (int j = 0; j < dst.rows; j++) {
//		for (int i = 0; i < dst.cols; i++) {
//			if ((int)dst.at<float>(j, i) > THRESHOLD_HARRIS) {
//
//				cv::circle(img, cv::Point(i, j), 2, cv::Scalar(0, 0, 255), -1, 8, 0);
//			}
//		}
//	}
//	cv::imshow("harris", img);
//	cv::waitKey(0);
