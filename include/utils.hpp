#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <system_error>
#include <string>
#include <vector>
#include <memory>

#include "opencv2/opencv.hpp"
#include "common.hpp"
#include "fkyaml.hpp"

struct ResizePadInfo {
    cv::Mat     img;
    float       scale;
    int         padTop;
    int         padLeft;
};

bool fileExists(const std::string fileName);
bool ensure_dir(const std::string &dir);
std::string getFileName(std::string filePath);
std::vector<float> toCHWFloat(cv::Mat &src, const float *meanVals, const float *normVals);
void toCHWFloat(const cv::Mat& src, float* dst, const float* meanVals, const float* stdVals);
void toCHWFloat(const cv::Mat& src, float* dst, const float* meanVals, const float* stdVals, const float scale);
ResizePadInfo resizeAndPad(const cv::Mat& src, int targetH, int targetW, cv::Scalar paddValue = cv::Scalar(255, 255, 255));
cv::Mat drawBoxes(const cv::Mat& src,const std::vector<std::vector<cv::Point2f>>& boxes);

template<typename T>
T getFkyamlValue(const fkyaml::node& n, const std::string& key, T defVal) {
    if (n.contains(key)) {
        try {
            return n[key].get_value<T>();
        } catch (...) {
        }
    }
    return defVal;
}

#endif //__UTILS_HPP__
