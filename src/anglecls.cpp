#include <string>
#include <numeric>
#include <fstream>

#include "model.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.hpp" 
#include "logger.hpp"
#include "fkyaml.hpp"
#include "anglecls.hpp"

using namespace std;

namespace model{

namespace anglecls {

Anglecls::Anglecls(ModelParams &params, logger::Level level) : Model(params, level) {

    std::ifstream ifs(params.inferYaml.c_str());
    if (!ifs.is_open()) {
        LOGE("Failed to open infer yaml: %s", params.inferYaml.c_str());
        return;
    }

    fkyaml::node root;
    try {
        root = fkyaml::node::deserialize(ifs);
    } catch (const std::exception &e) {
        LOGE("Failed to parse yaml: %s", e.what());
        return;
    }

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
    bool foundResize = false;

    for (const auto &op : ops) {
        if (!op.is_mapping()) continue;

        if (!foundResize && op.contains("ResizeImage")) {
            fkyaml::node resize = op["ResizeImage"];
            if (resize.is_mapping() && resize.contains("size") && resize["size"].is_sequence() &&
                resize["size"].size() >= 2) {
                // YAML: size: [width, height]
                m_dstWidth  = resize["size"][0].get_value<int>();
                m_dstHeight = resize["size"][1].get_value<int>();
                foundResize = true;
            } else {
                LOGE("ResizeImage.size missing or invalid");
            }
        }

        if (!foundNormalize && op.contains("NormalizeImage")) {
            fkyaml::node norm = op["NormalizeImage"];
            if (!norm.is_mapping()) {
                LOGW("NormalizeImage is not a map");
                continue;
            }

            // scale
            m_scale = getFkyamlValue(norm, "scale", 0.00392156862745098f);

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

            // channel_num
            if (norm.contains("channel_num")) {
                m_channels = norm["channel_num"].get_value<int>();
            } else {
                LOGW("NormalizeImage.channel_num missing, using default m_channels=0");
            }

            foundNormalize = true;
        }

        if (foundResize && foundNormalize) break;
    }

    if (!foundResize) {
        LOGW("ResizeImage not found, m_dstWidth/m_dstHeight uninitialized");
    }
    if (!foundNormalize) {
        LOGW("NormalizeImage not found, using default mean/std and m_channels");
    }
}

void Anglecls::setup(void const* data, size_t size) {
    LOG("Anglecls model setup success!!");
}

bool Anglecls::preProcessCpu(InferContext& ctx) {
    if(ctx.roiMats.empty()){
        ctx.srcMat =cv::imread(ctx.imagePath);
        if (ctx.srcMat.data == nullptr) {
            LOGE("ERROR: Image file not founded! Program terminated"); 
            return false;
        }
        ctx.roiMats.emplace_back(ctx.srcMat);
    }

    m_timer->startCpu();
    int batch = static_cast<int>(ctx.roiMats.size());

    ctx.inputValues.clear();
    ctx.inputShape.clear();
    size_t single_size = m_channels * m_dstHeight * m_dstWidth;
    ctx.inputValues.resize(batch*single_size);
    int index = 0;
    for(auto &src_mat : ctx.roiMats){
        cv::Mat resize_mat;
        cv::resize(src_mat, resize_mat, cv::Size(m_dstWidth, m_dstHeight));
        float* dst_ptr = ctx.inputValues.data() + index * single_size;
        toCHWFloat(resize_mat, dst_ptr, m_meanValues, m_normValues, m_scale);
        index++;
    }

    ctx.inputShape = {batch, m_channels, m_dstHeight, m_dstWidth};

    switch(m_params->inferBackend){
        case common::infer_backend::ORT_CUDA:
        case common::infer_backend::ORT_CPU:{
            auto mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
            ctx.inputTensor = Ort::Value::CreateTensor<float>(mem_info, 
                ctx.inputValues.data(), 
                ctx.inputValues.size(), 
                ctx.inputShape.data(), 
                ctx.inputShape.size());
            break;
        }
    }

    m_timer->stopCpu();
    ctx.preTime = m_timer->durationCpu<timer::Timer::ms>("Anglecls preprocess(CPU)");
    return true;
}

bool Anglecls::preProcessCuda(InferContext& ctx){
    return preProcessCpu(ctx);
}

bool Anglecls::postProcessCpu(InferContext& ctx) {
    m_timer->startCpu();
    auto shape = ctx.outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    int batch = static_cast<int>(shape[0]);
    int num_classes = static_cast<int>(shape[1]);
    float* output_data = ctx.outputTensor[0].GetTensorMutableData<float>();

    ctx.roiRoutes.clear();
    ctx.roiRoutes.reserve(batch);

    for (int b = 0; b < batch; ++b) {
        float* batch_data = output_data + b * num_classes;
        int max_index = 0;
        float max_value = batch_data[0];
        for (int c = 1; c < num_classes; ++c) {
            if (batch_data[c] > max_value) {
                max_value = batch_data[c];
                max_index = c;
            }
        }
        ctx.roiRoutes.emplace_back(max_index);

        auto toAngle = [](int idx) { return idx == 0 ? 0 : 180; };
        int angle = toAngle(max_index);
        LOG("Batch %d: angle=%d°, score=%.2f", b, angle, max_value*100);
    }

    m_timer->stopCpu();
    ctx.postTime = m_timer->durationCpu<timer::Timer::ms>("Anglecls postprocess(CPU)");
    return true;
}

bool Anglecls::postProcessCuda(InferContext& ctx){
    return postProcessCpu(ctx);
}

shared_ptr<Anglecls> makeAnglecls(ModelParams &params, logger::Level level)
{
    auto anglecls = make_shared<Anglecls>(params, level);
    anglecls->initModel();
    return anglecls;
}

}; // namespace anglecls

}; // namespace model
