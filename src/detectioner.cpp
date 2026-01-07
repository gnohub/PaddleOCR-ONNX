#include <string>
#include <numeric>
#include <fstream>

#include "opencv2/core/persistence.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include "model.hpp"
#include "utils.hpp" 
#include "logger.hpp"
#include "detectioner.hpp"
#include "clipper.hpp"
#include "fkyaml.hpp"

using namespace std;

namespace model{

namespace detectioner {

Detectioner::Detectioner(ModelParams &params, logger::Level level, float minSide) : Model(params, level), m_minSide(minSide) {

    std::ifstream ifs(params.inferYaml.c_str());
    if (!ifs.is_open()) {
        LOGE("Failed to open infer yaml: %s", params.inferYaml.c_str());
        return;
    }

    fkyaml::node root;
    try {
        root = fkyaml::node::deserialize(ifs);
    } catch (const std::exception& e) {
        LOGE("Failed to parse yaml: %s", e.what());
        return;
    }

    if (!root.contains("PostProcess")) {
        LOGE("PostProcess not found in yaml");
        return;
    }

    fkyaml::node post = root["PostProcess"];
    if (!post.is_mapping()) {
        LOGE("PostProcess is not a map");
        return;
    }

    m_textThresh    = getFkyamlValue(post, "thresh",         0.3f);
    m_scoreThresh   = getFkyamlValue(post, "box_thresh",     0.6f);
    m_maxCandidates = getFkyamlValue(post, "max_candidates", 1000);
    m_unClipRatio   = getFkyamlValue(post, "unclip_ratio",   1.5f);

    if (!root.contains("PreProcess")) {
        LOGE("PreProcess not found in yaml");
        return;
    }

    fkyaml::node preprocess = root["PreProcess"];
    if (!preprocess.contains("transform_ops")) {
        LOGE("transform_ops not found in PreProcess");
        return;
    }

    fkyaml::node ops = preprocess["transform_ops"];
    if (!ops.is_sequence()) {
        LOGE("transform_ops is not a sequence");
        return;
    }

    bool foundNormalize = false;

    for (const auto& op : ops) {
        if (!op.is_mapping() || !op.contains("NormalizeImage")) {
            continue;
        }

        fkyaml::node norm = op["NormalizeImage"];
        if (!norm.is_mapping()) {
            LOGW("NormalizeImage is not a map");
            continue;
        }

        // mean
        if (!norm.contains("mean") || !norm["mean"].is_sequence() ||
            norm["mean"].size() < 3) {
            LOGE("NormalizeImage.mean invalid");
            continue;
        }

        // std
        if (!norm.contains("std") || !norm["std"].is_sequence() ||
            norm["std"].size() < 3) {
            LOGE("NormalizeImage.std invalid");
            continue;
        }

        auto mean = norm["mean"];
        auto std  = norm["std"];

        m_meanValues[2] = mean[0].get_value<float>();
        m_meanValues[1] = mean[1].get_value<float>();
        m_meanValues[0] = mean[2].get_value<float>();

        m_normValues[2] = std[0].get_value<float>();
        m_normValues[1] = std[1].get_value<float>();
        m_normValues[0] = std[2].get_value<float>();

        foundNormalize = true;
        break;
    }

    if (!foundNormalize) {
        LOGW("NormalizeImage not found, using default mean/std");
    }
}

void Detectioner::setup(void const* data, size_t size) {
    LOG("Detectioner model setup success!!");
}

bool Detectioner::preProcessCpu(InferContext& ctx) {
    // Read Imgage
    if(ctx.srcMat.empty()){
        ctx.srcMat = cv::imread(ctx.imagePath);
        if (ctx.srcMat.data == nullptr) {
            LOGE("ERROR: Image file not founded! Program terminated"); 
            return false;
        }
    }

    m_timer->startCpu();
    // BGR2RGB
    cv::Mat rgb_img;
    cvtColor(ctx.srcMat, rgb_img, cv::COLOR_BGR2RGB);

    // Padding
    auto pad_info = resizeAndPad(rgb_img, m_params->img.h, m_params->img.w);
    cv::Mat dst_img = pad_info.img;
    m_scale   = pad_info.scale;
    m_padTop  = pad_info.padTop;
    m_padLeft = pad_info.padLeft;
    m_srcWidth = ctx.srcMat.cols;
    m_srcHeight = ctx.srcMat.rows;

    // Normalize and to tensor(bchw)
    ctx.inputValues = toCHWFloat(dst_img, m_meanValues, m_normValues);
    ctx.inputShape = {1, dst_img.channels(), dst_img.rows, dst_img.cols};
    ctx.inputTensor = Ort::Value::CreateTensor<float>(m_onnxMemInfo, 
        ctx.inputValues.data(), 
        ctx.inputValues.size(), 
        ctx.inputShape.data(), 
        ctx.inputShape.size());

    m_timer->stopCpu();
    ctx.preTime = m_timer->durationCpu<timer::Timer::ms>("Detectioner preprocess(CPU)");

    return true;
}

std::pair<std::vector<cv::Point2f>, float>
Detectioner::getMiniBoxes(const std::vector<cv::Point2f> &contour) {
    cv::RotatedRect box = cv::minAreaRect(contour);

    std::vector<cv::Point2f> points(4);
    box.points(points.data());

    std::sort(
    points.begin(), points.end(), [](const cv::Point2f &a, const cv::Point2f &b) { return a.x < b.x; });

    int index_1 = 0, index_2 = 1, index_3 = 2, index_4 = 3;
    if (points[1].y > points[0].y) {
        index_1 = 0;
        index_4 = 1;
    } else {
        index_1 = 1;
        index_4 = 0;
    }

    if (points[3].y > points[2].y) {
        index_2 = 2;
        index_3 = 3;
    } else {
        index_2 = 3;
        index_3 = 2;
    }

    std::vector<cv::Point2f> box_points = {points[index_1], points[index_2], points[index_3], points[index_4]};

    float sside = std::min(box.size.width, box.size.height);
    return std::make_pair(box_points, sside);
}

float Detectioner::getScoreFast(const cv::Mat &bitmap,
                                const std::vector<cv::Point2f> &contour) {
    if (contour.empty()) return 0.0f;

    int h = bitmap.size[bitmap.dims - 2];
    int w = bitmap.size[bitmap.dims - 1];

    float xmin_f = contour[0].x, xmax_f = contour[0].x;
    float ymin_f = contour[0].y, ymax_f = contour[0].y;

    for (const auto &pt : contour) {
        if (pt.x < xmin_f) xmin_f = pt.x;
        if (pt.x > xmax_f) xmax_f = pt.x;
        if (pt.y < ymin_f) ymin_f = pt.y;
        if (pt.y > ymax_f) ymax_f = pt.y;
    }

    int xmin = std::max(0, static_cast<int>(std::floor(xmin_f)));
    int xmax = std::min(w - 1, static_cast<int>(std::ceil(xmax_f)));
    int ymin = std::max(0, static_cast<int>(std::floor(ymin_f)));
    int ymax = std::min(h - 1, static_cast<int>(std::ceil(ymax_f)));

    if (xmax <= xmin || ymax <= ymin) return 0.0f;

    cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
    std::vector<cv::Point> contour_int;
    contour_int.reserve(contour.size());

    for (const auto &pt : contour) {
        int x = cvRound(pt.x - xmin);
        int y = cvRound(pt.y - ymin);
        contour_int.emplace_back(x, y);
    }

    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{contour_int}, cv::Scalar(1));
    cv::Mat roi = bitmap(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
    cv::Scalar mean_val = cv::mean(roi, mask);

    return static_cast<float>(mean_val[0]);
}

std::vector<cv::Point2f>
Detectioner::unClip(const std::vector<cv::Point2f> &box, float unClipRatio) {
    std::vector<cv::Point2f> result;
    float area = cv::contourArea(box);
    float length = cv::arcLength(box, true);
    float distance = area * unClipRatio / length;

    ClipperLib::Path path;
    for (const auto &point : box) {
        path << ClipperLib::IntPoint(point.x, point.y);
    }

    ClipperLib::ClipperOffset co;
    co.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

    ClipperLib::Paths solution;
    co.Execute(solution, distance);

    if (solution.empty()) {
        return result;
    }

    for (const auto &p : solution[0]) {
        result.emplace_back(p.X, p.Y);
    }

    return result;
}

static void orderPoints(const std::vector<cv::Point2f>& pts, cv::Point2f ordered[4]) {
    std::vector<cv::Point2f> sorted = pts;
    std::sort(sorted.begin(), sorted.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

    cv::Point2f tl, tr, bl, br;
    if (sorted[0].x < sorted[1].x) { tl = sorted[0]; tr = sorted[1]; } 
    else { tl = sorted[1]; tr = sorted[0]; }

    if (sorted[2].x < sorted[3].x) { bl = sorted[2]; br = sorted[3]; } 
    else { bl = sorted[3]; br = sorted[2]; }

    ordered[0] = tl; ordered[1] = tr; ordered[2] = br; ordered[3] = bl;
}

bool Detectioner::postProcessCpu(InferContext& ctx) {
    m_timer->startCpu();
    assert(!ctx.outputTensor.empty());

    float* float_array = ctx.outputTensor[0].GetTensorMutableData<float>();
    cv::Mat out_mat(m_params->img.h, m_params->img.w, CV_32FC1, float_array);

    cv::Mat bit_mat;
    cv::threshold(out_mat, bit_mat, m_textThresh, 255, cv::THRESH_BINARY);
    bit_mat.convertTo(bit_mat, CV_8UC1);

    std::vector<std::vector<cv::Point>> contours_i;
    cv::findContours(bit_mat, contours_i, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point2f>> contours;
    for (auto& c : contours_i) {
        std::vector<cv::Point2f> cf;
        for (auto& p : c) cf.emplace_back(p.x, p.y);
        contours.push_back(cf);
    }

    std::sort(contours.begin(), contours.end(),
              [](const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b) {
                  return cv::contourArea(a) > cv::contourArea(b);
              });

    int num = std::min((int)contours.size(), m_maxCandidates);
    std::vector<BoxWithCoord> valid_boxes;

    for (int i = 0; i < num; i++) {
        // find mini boxs
        auto box_ret = getMiniBoxes(contours[i]);
        if (box_ret.second < m_minSide) continue;

        // find rect
        float score = getScoreFast(out_mat, box_ret.first);
        if (score < m_scoreThresh) continue;

        auto unclip = unClip(box_ret.first, m_unClipRatio);
        if (unclip.size() < 4) continue;

        auto minbox = getMiniBoxes(unclip);
        if (minbox.second < m_minSide) continue;

        for (auto& p : minbox.first) {
            p.x = (p.x - m_padLeft) / m_scale;
            p.y = (p.y - m_padTop) / m_scale;
            p.x = std::max(0.f, std::min(p.x, (float)ctx.srcMat.cols - 1));
            p.y = std::max(0.f, std::min(p.y, (float)ctx.srcMat.rows - 1));
        }

        float top = std::min({minbox.first[0].y, minbox.first[1].y, minbox.first[2].y, minbox.first[3].y});
        float left = std::min({minbox.first[0].x, minbox.first[1].x, minbox.first[2].x, minbox.first[3].x});
        valid_boxes.push_back({minbox.first, top, left});
    }

    std::sort(valid_boxes.begin(), valid_boxes.end(),
              [](const BoxWithCoord& a, const BoxWithCoord& b) {
                  return a.top != b.top ? a.top < b.top : a.left < b.left;
              });

    const cv::Mat& src_mat = ctx.srcMat;
    int idx = 0;
    for (auto& b : valid_boxes) {
        if (m_params->saveImg) {
            cv::Rect bbox = cv::boundingRect(b.box) & cv::Rect(0, 0, src_mat.cols, src_mat.rows);
            cv::Mat roi = src_mat(bbox).clone();
            std::string path = "output/det_mat_" + std::to_string(idx) + ".png";
            cv::imwrite(path, roi);
        }

        cv::Point2f src_pts[4];
        orderPoints(b.box, src_pts);

        float dx = src_pts[1].x - src_pts[0].x;
        float dy = src_pts[1].y - src_pts[0].y;
        float width = std::hypot(dx, dy);

        dx = src_pts[3].x - src_pts[0].x;
        dy = src_pts[3].y - src_pts[0].y;
        float height = std::hypot(dx, dy);

        //swap width/height and rotate
        if (height > width) {
            std::swap(width, height);
            cv::Point2f tmp[4];
            tmp[0] = src_pts[3]; tmp[1] = src_pts[0];
            tmp[2] = src_pts[1]; tmp[3] = src_pts[2];
            for (int i=0;i<4;i++) src_pts[i] = tmp[i];
        }

        if (width < 2.f) width = 2.f;
        if (height < 2.f) height = 2.f;

        cv::Point2f top_center = (src_pts[0] + src_pts[1]) * 0.5f;
        cv::Point2f bottom_center = (src_pts[2] + src_pts[3]) * 0.5f;
        if (top_center.y > bottom_center.y) {
            std::swap(src_pts[0], src_pts[3]);
            std::swap(src_pts[1], src_pts[2]);
        }

        float left_sum = src_pts[0].x + src_pts[3].x;
        float right_sum = src_pts[1].x + src_pts[2].x;
        if (left_sum > right_sum) {
            std::swap(src_pts[0], src_pts[1]);
            std::swap(src_pts[3], src_pts[2]);
        }

        cv::Point2f dst_pts[4] = {
            {0.f, 0.f},
            {width - 1.f, 0.f},
            {width - 1.f, height - 1.f},
            {0.f, height - 1.f}
        };

        cv::Mat perspect_mat = cv::getPerspectiveTransform(src_pts, dst_pts);
        cv::Mat final_mat;
        cv::warpPerspective(src_mat, final_mat, perspect_mat,
                            cv::Size((int)width, (int)height),
                            cv::INTER_CUBIC,
                            cv::BORDER_REPLICATE);

        if (m_params->saveImg) {
            std::string path = "output/corrected_mat_" + std::to_string(idx) + ".png";
            cv::imwrite(path, final_mat);
        }

        ctx.roiMats.push_back(final_mat);
        ctx.boxes.push_back(b.box);
        idx++;
    }

    LOGV("Boxes count:%d", ctx.boxes.size());
    LOGV("Child mat count:%d", ctx.roiMats.size());
    m_timer->stopCpu();
    ctx.postTime = m_timer->durationCpu<timer::Timer::ms>("Detectioner postprocess(CPU)");

    if(m_params->saveImg){
        cv::imwrite("output/dec_dst.png", drawBoxes(ctx.srcMat, ctx.boxes));
    }
    return !ctx.boxes.empty();
}

shared_ptr<Detectioner> makeDetectioner(model::ModelParams &params, logger::Level level, float minSide)
{
    auto detectioner = make_shared<Detectioner>(params, level, minSide);
    detectioner->initModel();
    return detectioner;
}

}; // namespace detectioner

}; // namespace model
