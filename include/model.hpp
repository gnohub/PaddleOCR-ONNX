#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <memory>
#include <vector>
#include <string>
#include "common.hpp"
#include "timer.hpp"
#include "logger.hpp"
#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"

namespace model{

constexpr int NORMALIZE_DIMS_MAX = 3;

struct ImageInfo {
    int c;
    int w;
    int h;
    ImageInfo(int channel, int height, int width) : c(channel), h(height), w(width) {}
};

struct BoxWithCoord {
    std::vector<cv::Point2f> box;
    float top;
    float left;
};

struct ModelParams {
    common::infer_backend       inferBackend        = common::infer_backend::ORT_CPU;
    common::task_type           task                = common::task_type::DETECTION;
    common::precision           prec                = common::FP32;
    ImageInfo                   img                 = {3, 960, 960};
    std::string                 onnxPath;
    std::string                 inferYaml;
    int                         intraThreadnum      = 1;
    int                         interThreadnum      = 1;
    bool                        saveImg             = false;
};

struct InferContext {
    std::string                           imagePath;
    cv::Mat                               srcMat;
    std::vector<float>                    inputValues;
    std::vector<int64_t>                  inputShape;
    Ort::Value                            inputTensor{nullptr};
    std::vector<Ort::Value>               outputTensor;
    std::vector<std::vector<cv::Point2f>> boxes;
    std::vector<cv::Mat>                  roiMats;
    std::vector<int>                      roiRoutes;
    std::vector<std::string>              regResults;
    double                                preTime;
    double                                inferTime;
    double                                postTime;
};

struct InferResult {
    std::vector<std::vector<cv::Point2f>>   decBoxes;
    std::vector<cv::Mat>                    decRets;
    std::vector<int>                        angleRets;
    std::vector<std::string>                regRets;
    double                                  preTime = 0.0;
    double                                  inferTime = 0.0;
    double                                  postTime = 0.0;
};

class OrtEnvSingleton {
public:
    static Ort::Env& ort_env() {
        static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "PaddleOCR-ONNX");
        return env;
    }
private:
    OrtEnvSingleton() = delete;
};

class Model {

public:
    Model(ModelParams &params, logger::Level level); 
    virtual ~Model() {};
    void loadData(); 
    void initModel();
    void inference(InferContext& ctx, std::string imagePath);

public:
    bool enqueueBindings(InferContext& ctx);
    virtual void setup(void const* data, std::size_t size)      = 0;
    virtual bool preProcessCpu(InferContext& ctx)               = 0;
    virtual bool postProcessCpu(InferContext& ctx)              = 0;
    virtual bool preProcessCuda(InferContext& ctx)              = 0;
    virtual bool postProcessCuda(InferContext& ctx)             = 0;

public:
    ModelParams*                                m_params = nullptr;
    Ort::Env&                                   m_onnxEnv = OrtEnvSingleton::ort_env();
    std::shared_ptr<Ort::Session>               m_onnxSession;
    Ort::SessionOptions                         m_onnxOptions;
    std::unique_ptr<char[], decltype(&free)>    m_inputName;
    std::unique_ptr<char[], decltype(&free)>    m_outputName;

    int                                         m_srcWidth;
    int                                         m_srcHeight;
    float                                       m_meanValues[NORMALIZE_DIMS_MAX] = {0.406, 0.456, 0.485};
    float                                       m_normValues[NORMALIZE_DIMS_MAX] = {0.225, 0.224, 0.229};
    std::shared_ptr<logger::Logger>             m_logger;
    std::shared_ptr<timer::Timer>               m_timer;
};

}; // namespace model

#endif //__MODEL_HPP__
