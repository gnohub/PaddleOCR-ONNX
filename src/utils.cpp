#include <experimental/filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "utils.hpp"
#include "model.hpp"
#include "logger.hpp"

using namespace std;

namespace fs = std::experimental::filesystem;

bool ensure_dir(const std::string &dir) {
    std::error_code ec;
    bool created = fs::create_directories(dir, ec);
    if (ec) {
        LOGE("Failed to create directory:%s, reason:%s", dir, ec.message());
        return false;
    }
    return true;
}

bool fileExists(const string fileName) {
    if (!experimental::filesystem::exists(experimental::filesystem::path(fileName))){
        LOGE("file:%s not exists", fileName.c_str());
        return false;
    }else{
        return true;
    }
}

ResizePadInfo resizeAndPad(const cv::Mat& src, int targetH, int targetW, cv::Scalar paddValue) {
    int h = src.rows;
    int w = src.cols;

    float scale = std::min(
        static_cast<float>(targetH) / h,
        static_cast<float>(targetW) / w
    );

    int new_h = static_cast<int>(std::round(h * scale));
    int new_w = static_cast<int>(std::round(w * scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    int pad_h = targetH - new_h;
    int pad_w = targetW - new_w;

    int top    = pad_h / 2;
    int bottom = pad_h - top;
    int left   = pad_w / 2;
    int right  = pad_w - left;

    cv::Mat output;
    cv::copyMakeBorder(
        resized,
        output,
        top, bottom, left, right,
        cv::BORDER_CONSTANT,
        paddValue
    );

    return {output, scale, top, left};
}

cv::Mat drawBoxes(const cv::Mat& src, const std::vector<std::vector<cv::Point2f>>& boxes) {
    cv::Mat dst = src.clone();
    for (const auto& box : boxes) {
        std::vector<cv::Point> pts;
        pts.reserve(box.size());

        for (const auto& p : box) {
            pts.emplace_back(cv::Point(cvRound(p.x), cvRound(p.y)));
        }

        const cv::Point* pts_ptr = pts.data();
        int npts = pts.size();
        cv::polylines(
            dst,
            &pts_ptr,
            &npts,
            1,
            true,
            cv::Scalar(0, 0, 255),
            2
        );
    }
    return dst;
}

std::vector<float> toCHWFloat(cv::Mat &src, const float *meanVals, const float *stdVals) {
    int H = src.rows;
    int W = src.cols;
    int C = src.channels();
    int image_size = H * W;
    std::vector<float> inputTensor(C * image_size);

    const uchar* ptr = src.data;

    for (int i = 0; i < image_size; ++i) {
        for (int c = 0; c < C; ++c) {
            float val = ptr[i * C + c] / 255.0f;
            val = (val - meanVals[c]) / stdVals[c];
            inputTensor[c * image_size + i] = val;
        }
    }

    return inputTensor;
}

void toCHWFloat(const cv::Mat& src, float* dst, const float* meanVals, const float* stdVals) {
    int H = src.rows;
    int W = src.cols;
    int C = src.channels();  // 3
    int image_size = H * W;

    const uchar* ptr = src.data;

    // BGR -> RGB + normalize + CHW
    for (int i = 0; i < image_size; ++i) {
        uchar b = ptr[i * 3 + 0];
        uchar g = ptr[i * 3 + 1];
        uchar r = ptr[i * 3 + 2];

        float r_f = (r / 255.0f - meanVals[0]) / stdVals[0];
        float g_f = (g / 255.0f - meanVals[1]) / stdVals[1];
        float b_f = (b / 255.0f - meanVals[2]) / stdVals[2];

        dst[0 * image_size + i] = r_f;  // R
        dst[1 * image_size + i] = g_f;  // G
        dst[2 * image_size + i] = b_f;  // B
    }
}

void toCHWFloat(const cv::Mat& src, float* dst, const float* meanVals, const float* stdVals, const float scale) {
    int H = src.rows;
    int W = src.cols;
    int C = src.channels();  // 3
    int image_size = H * W;

    const uchar* ptr = src.data;

    // BGR -> RGB + normalize + CHW
    for (int i = 0; i < image_size; ++i) {
        uchar b = ptr[i * 3 + 0];
        uchar g = ptr[i * 3 + 1];
        uchar r = ptr[i * 3 + 2];

        float r_f = (r * scale - meanVals[0]) / stdVals[0];
        float g_f = (g * scale - meanVals[1]) / stdVals[1];
        float b_f = (b * scale - meanVals[2]) / stdVals[2];

        dst[0 * image_size + i] = r_f;  // R
        dst[1 * image_size + i] = g_f;  // G
        dst[2 * image_size + i] = b_f;  // B
    }
}

vector<unsigned char> loadFile(const string &file) {
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);
        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

string getFileName(string filePath) {
    int pos = filePath.rfind("/");
    string suffix;
    suffix = filePath.substr(pos + 1, filePath.length());
    return suffix;
}