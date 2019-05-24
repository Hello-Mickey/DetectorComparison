#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/features2d.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

typedef cv::Vec<uchar, 3> Vec3b;

const int THRESHOLD_FAST = 50;
const int THRESHOLD_HARRIS = 175;
const int PIXEL_EPS = 2;
const int eps = PIXEL_EPS * PIXEL_EPS;
std::string result_path = "./results/";

void DrawPoints(cv::Mat& img, std::vector<cv::KeyPoint>& kpts, cv::Scalar color) {
	for (int i = 0; i < kpts.size(); ++i) {
		cv::Point point = kpts[i].pt;
		//std::cout << "X: " << point.x << "    Y: " << point.y << '\n';
		cv::circle(img, point, 2, color, -1, 8, 0);
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
	DrawPoints(img_out, kpts, cv::Scalar(0, 0, 255));
	std::cout << type << '\n' << kpts.size() << '\n';
	std::string path = "./results/" + type + ".jpg";
	cv::imwrite(path, img_out);
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
	std::cout << "keypoints count 1: " << kpts1.size() << '\n' << "keypoints count 2: " << kpts2.size() << '\n';


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
	std::cout << kpts_merge.size() << '\n';
}

void init(cv::Mat& img, cv::Mat& gray, cv::Mat& img_fast, cv::Mat& img_orb, cv::Mat& img_merge, cv::Mat& img_brisk) {
	img.copyTo(img_fast);
	img.copyTo(img_orb);
	img.copyTo(img_merge);
	img.copyTo(img_brisk);
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
}

//example_16 - 02.cpp
typedef cv::Vec<uchar, 2> Vec2b;
void printMat(cv::Mat& rot) {
	std::cout << rot.rows << " " << rot.cols << '\n';
	for (int i = 0; i < rot.rows; ++i) {
		for (int j = 0; j < rot.cols; ++j) {
			std::cout << rot.at<double>(i, j) << ' ';
		}
		std::cout << '\n';
	}
}

std::vector<cv::KeyPoint> CoordTransform(std::vector<cv::KeyPoint>& k,  cv::Mat& tr) {
	std::vector<cv::KeyPoint> kr = k;
	for (int i = 0; i < k.size(); ++i) {
		kr[i].pt.x = tr.at<double>(0, 0) * k[i].pt.x + tr.at<double>(0, 1) * k[i].pt.y + tr.at<double>(0, 2);
		kr[i].pt.y = tr.at<double>(1, 0) * k[i].pt.x + tr.at<double>(1, 1) * k[i].pt.y + tr.at<double>(1, 2);
		
		//std::cout << kr[i].pt.x << " " << kr[i].pt.y<<'\n';
	}
	return kr;
}
int rotateAndCheck(cv::Mat& img, int alpha) {
	
	std::vector<cv::KeyPoint> result, kpts_rotated, kpts_transform, kpts;
	cv::Mat gray, img_rotated, gray_rotated, desc, img_result;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	
	cv::Mat M = cv::getRotationMatrix2D(cv::Point(img.cols / 2.0, img.rows / 2.0), alpha, 1);
	printMat(M);
	cv::warpAffine(img, img_rotated, M, img.size());
	cv::cvtColor(img_rotated, gray_rotated, cv::COLOR_BGR2GRAY);
	img_rotated.copyTo(img_result);

	CornerDetector("fast", gray, img, kpts, desc);
	CornerDetector("fast", gray_rotated, img_rotated, kpts_rotated, desc);
	cv::imshow("source", img);
	cv::imshow("source rotated", img_rotated);
	cv::waitKey(0);

	kpts_transform = CoordTransform(kpts, M);

	merge(kpts_transform, kpts_rotated, result);
	DrawPoints(img_result, result, cv::Scalar(0, 255, 0));
	cv::imshow("result", img_result);
	cv::waitKey(0);
	
	int merge_count = result.size();
	return merge_count;
}

void statistics(std::string& type, std::vector<std::string>&imgs, int alpha, int betta, int step) {
	
	std::vector<std::vector<int>> stat;
	for (int i = 0; i < imgs.size(); ++i) {
		cv::Mat img;
		std::vector< int> img_stat;
		std::string path = imgs[i] + ".jpeg";
		std::cout << path << '\n';
		cv::imread(path);
		for (int j = alpha; j <= betta; j += step) {
			img_stat.push_back(rotateAndCheck(img, j));
		}
		stat.push_back(img_stat);
	}
}

int main() {
	cv::Mat img, desc, gray, thresh;
	std::vector<cv::KeyPoint> kpts, result, kpts1, kpts2, kpts_merge;
	img = cv::imread("./imgs/Number.jpg");
	int a = rotateAndCheck(img, 60);
	
}

int main1() {
	cv::Mat img, gray, thresh, img_fast, img_orb, desc_orb, img_merge, img_brisk, desc_brisk;
	std::vector<cv::KeyPoint> kpts_fast, kpts_orb, kpts_merge, kpts_merge1, kpts_merge3, kpts_brisk, kpts_merge_brisk_orb;

	cv::Mat img2, gray2, thresh2, img_fast2, img_orb2, desc_orb2, img_merge2, img_brisk2, desc_brisk2, dst;
	std::vector<cv::KeyPoint> kpts_fast2, kpts_orb2, kpts_merge2, kpts_brisk2, kpts_merge_brisk_orb2;

	cv::Mat img_transform, img_transform_merge;
	std::vector<cv::KeyPoint> kpts_orb_transform, kpts_orb_transform_merge;
	double angle = 180;

	img = cv::imread("./imgs/Number.jpg");
	img2 = cv::imread("./imgs/Number180.jpg");

	img2.copyTo(img_transform);
	img2.copyTo(img_transform_merge);

	init(img, gray, img_fast, img_orb, img_merge, img_brisk);
	init(img2, gray2, img_fast2, img_orb2, img_merge2, img_brisk2);
	cv::Mat M = cv::getRotationMatrix2D(cv::Point(img.cols / 2.0, img.rows / 2.0), 180, 1);
	printMat(M);

	CornerDetector("fast", gray, img_fast, kpts_fast, desc_orb);
	CornerDetector("orb", gray, img_orb, kpts_orb, desc_orb);
	CornerDetector("brisk", gray, img_brisk, kpts_brisk, desc_brisk);

	std::cout << '\n' << "comparing fast and orb" << '\n';
	merge(kpts_fast, kpts_orb, kpts_merge1);

	std::cout << '\n' << "comparing fast and brisk" << '\n';
	merge(kpts_fast, kpts_brisk, kpts_merge2);

	std::cout << '\n' << "comparing orb and brisk" << '\n';
	merge(kpts_orb, kpts_brisk, kpts_merge3);

	std::cout << '\n' << "comparing fast orb brisk, same points" << '\n';
	merge(kpts_merge1, kpts_brisk, kpts_merge);
	DrawPoints(img, kpts_merge, cv::Scalar(0, 255, 0));//transform source2
	cv::imshow(" ", img);
	cv::waitKey(0);
	cv::imshow("fast", img_fast);
	cv::imshow("orb", img_orb);
	cv::imshow("brisk", img_brisk);

	kpts_orb_transform = CoordTransform(kpts_orb, M);
	DrawPoints(img_transform, kpts_orb_transform, cv::Scalar(0, 255, 0));//transform copy source2
	
	merge(kpts_orb_transform, kpts_orb2, kpts_orb_transform_merge);
	DrawPoints(img_orb2, kpts_orb_transform_merge, cv::Scalar(0, 255, 0));//transform source2

	DrawPoints(img_transform_merge, kpts_orb_transform_merge, cv::Scalar(0, 255, 0));//transform source2

	
	cv::imshow("ORB_2_transform", img_transform);
//	cv::imshow("ORB_transform_merge1", img_transform_merge);
	cv::imshow("ORB_transform_merge2", img_orb2);

//	std::string path = result_path + "transformMerge.jpg";
//	cv::imwrite(path, img_orb2);
	cv::waitKey(0);





}




