#ifndef __DETECTIONER_HPP__
#define __DETECTIONER_HPP__

#include <memory>
#include <vector>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "common.hpp"
#include "logger.hpp"
#include "model.hpp"

namespace model{

namespace detectioner {
class Detectioner : public Model{

public:
    Detectioner(ModelParams &params, logger::Level level, float minSide=3);

public:
    virtual void setup(void const* data, std::size_t size) override;
    virtual bool preProcessCpu(InferContext& ctx) override;
    virtual bool postProcessCpu(InferContext& ctx) override;

private:
    std::pair<std::vector<cv::Point2f>, float> getMiniBoxes(const std::vector<cv::Point2f> &contour);
    float getScoreFast(const cv::Mat &bitmap, const std::vector<cv::Point2f> &contour);
    std::vector<cv::Point2f> unClip(const std::vector<cv::Point2f> &box, float unClipRatio);

private:
    float   m_textThresh;
    float   m_scoreThresh;
    float   m_minSide;
    float   m_unClipRatio;
    int     m_maxCandidates;

    float   m_scale;
    int     m_padTop;
    int     m_padLeft;
};

std::shared_ptr<Detectioner> makeDetectioner(ModelParams &params, logger::Level level, float minSide=3);

}; // namespace detectioner
}; // namespace model

#endif //__DETECTIONER_HPP__
