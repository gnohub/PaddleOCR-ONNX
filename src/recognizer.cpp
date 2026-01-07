#include <string>
#include <numeric>
#include <fstream>

#include "model.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.hpp" 
#include "logger.hpp"
#include "fkyaml.hpp"
#include "recognizer.hpp"

using namespace std;

namespace model{

namespace recognizer {

Recognizer::Recognizer(ModelParams &params, logger::Level level) : Model(params, level) {
    std::ifstream ifs(params.inferYaml.c_str());
    if (!ifs.is_open()) {
        LOGE("Failed to open %s", params.inferYaml.c_str());
        assert(false);
        return;
    }

    fkyaml::node root;
    try {
        root = fkyaml::node::deserialize(ifs);
    } catch (const std::exception &e) {
        LOGE("Failed to parse yaml: %s", e.what());
        assert(false);
        return;
    }

    if (!root.contains("PreProcess")) {
        LOGE("PreProcess not found in yaml");
        assert(false);
        return;
    }

    fkyaml::node preprocess = root["PreProcess"];
    if (!preprocess.contains("transform_ops") || !preprocess["transform_ops"].is_sequence()) {
        LOGE("transform_ops not found or is not a sequence");
        assert(false);
        return;
    }

    fkyaml::node ops = preprocess["transform_ops"];
    bool foundResize = false;

    for (const auto &op : ops) {
        if (!op.is_mapping() || !op.contains("RecResizeImg")) {
            continue;
        }

        fkyaml::node rec_resize = op["RecResizeImg"];
        if (!rec_resize.contains("image_shape") || !rec_resize["image_shape"].is_sequence() ||
            rec_resize["image_shape"].size() < 3) {
            LOGE("RecResizeImg.image_shape missing or invalid");
            continue;
        }

        fkyaml::node image_shape = rec_resize["image_shape"];
        m_channels = image_shape[0].get_value<int>();
        m_dstHeight = image_shape[1].get_value<int>();
        m_dstWidth  = image_shape[2].get_value<int>();
        foundResize = true;
        break;
    }

    if (!foundResize) {
        LOGW("RecResizeImg not found, m_channels/m_dstHeight/m_dstWidth uninitialized");
    }

    if (!root.contains("PostProcess")) {
        LOGE("PostProcess not found in yaml");
        assert(false);
        return;
    }

    fkyaml::node post_node = root["PostProcess"];
    if (!post_node.contains("character_dict") || !post_node["character_dict"].is_sequence()) {
        LOGE("character_dict missing or not a sequence");
        assert(false);
        return;
    }

    fkyaml::node charSeq = post_node["character_dict"];
    m_characterList.reserve(charSeq.size() + 2);
    for (auto &item : charSeq) {
        if (item.is_scalar()) {
            m_characterList.emplace_back(item.get_value<std::string>());
        }
    }
    m_characterList.emplace_back(std::string(" "));
    m_characterList.insert(m_characterList.begin(), "blank");

    m_keys.reserve(m_characterList.size());
    for (int i = 0; i < m_characterList.size(); i++) {
        m_keys[i] = m_characterList[i];
    }

    LOG("Recognizer input channels:%d", m_channels);
    LOG("Recognizer input height:%d", m_dstHeight);
    LOG("Recognizer input width:%d", m_dstWidth);
    LOG("Recognizer keys count:%d", m_keys.size());
}

void Recognizer::setup(void const* data, size_t size) {
    for(int i = 0; i<NORMALIZE_DIMS_MAX; i++){
        m_meanValues[i] = 0.5;
        m_normValues[i] = 0.5;
    }
    LOG("Recognizer model setup success!!");
}

bool Recognizer::preProcessCpu(InferContext& ctx) {
    // read to rgb
    if(ctx.roiMats.empty()){
        ctx.srcMat =cv::imread(ctx.imagePath);
        if (ctx.srcMat.data == nullptr) {
            LOGE("ERROR: Image file not founded! Program terminated"); 
            return false;
        }
        ctx.roiRoutes.emplace_back(0);
        ctx.roiMats.emplace_back(ctx.srcMat);
    }


    if(ctx.roiRoutes.size() != ctx.roiMats.size()){
        LOGE("Error: angle vector info error");
        assert(false);
    }

    m_timer->startCpu();
    int batch = static_cast<int>(ctx.roiMats.size());
    int index = 0;
    int max_width = 0;

    ctx.inputValues.clear();
    ctx.inputShape.clear();

    std::vector<cv::Mat> resize_mats;
    resize_mats.reserve(batch);
    for(auto &src_mat : ctx.roiMats){
        cv::Mat res_mat;
        float scale = (float) m_dstHeight / (float) src_mat.rows;
        int rec_width = static_cast<int>(src_mat.cols*scale);
        cv::resize(src_mat, res_mat, cv::Size(rec_width, m_dstHeight));
        resize_mats.emplace_back(res_mat);
        if(max_width < res_mat.cols){
            max_width = res_mat.cols;
        }
    }

    size_t single_size = m_channels * m_dstHeight * max_width;
    ctx.inputValues.resize(batch*single_size);
    for(auto &src_mat : resize_mats){
        cv::Mat padd_mat = src_mat;
        int padd_right = max_width - src_mat.cols;
        if(padd_right){
            cv::copyMakeBorder(
                src_mat,
                padd_mat,
                0,
                0,
                0,
                padd_right,
                cv::BORDER_CONSTANT,
                cv::Scalar(255, 255, 255)
            );
        }

        cv::Mat dst_mat = padd_mat;
        if(ctx.roiRoutes[index]){
            cv::rotate(padd_mat, dst_mat, cv::ROTATE_180);
        }

        float* dst_ptr = ctx.inputValues.data() + index * single_size;
        toCHWFloat(dst_mat, dst_ptr, m_meanValues, m_normValues);
        index++;
    }

    ctx.inputShape = {batch, m_channels, m_dstHeight, max_width};
    ctx.inputTensor = Ort::Value::CreateTensor<float>(m_onnxMemInfo, 
        ctx.inputValues.data(), 
        ctx.inputValues.size(), 
        ctx.inputShape.data(), 
        ctx.inputShape.size());

    m_timer->stopCpu();
    ctx.preTime = m_timer->durationCpu<timer::Timer::ms>("Recognizer preprocess(CPU)");
    return true;
}

bool Recognizer::postProcessCpu(InferContext& ctx) {
    m_timer->startCpu();

    auto shape = ctx.outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

    int batch = static_cast<int>(shape[0]);
    int time_steps = static_cast<int>(shape[1]);
    int num_classes = static_cast<int>(shape[2]);

    float* output_data = ctx.outputTensor[0].GetTensorMutableData<float>();

    LOG("Recognizer output shape: [%d, %d, %d]", batch, time_steps, num_classes);

    ctx.regResults.clear();
    ctx.regResults.reserve(batch);
    
    for (int b = 0; b < batch; ++b) {
        std::string result;
        int last_index = -1;
        int same_count = 0;

        float* batch_data = output_data + b * time_steps * num_classes;

        for (int t = 0; t < time_steps; ++t) {
            float* step_logits = batch_data + t * num_classes;

            // argmax + second max
            int max_index = 0;
            float max_value = step_logits[0];
            int second_index = -1;
            float second_value = -1e9f;

            for (int c = 1; c < num_classes; ++c) {
                float v = step_logits[c];
                if (v > max_value) {
                    second_value = max_value;
                    second_index = max_index;
                    max_value = v;
                    max_index = c;
                } else if (v > second_value) {
                    second_value = v;
                    second_index = c;
                }
            }

            // CTC collapse
            if (max_index != 0 && max_index != last_index) {
                if (max_index < (int)m_characterList.size()) {
                    result += m_characterList[max_index];
                }
            }

            if (max_index == last_index) {
                same_count++;
            }

            last_index = max_index;
        }

        LOG("Batch %d: OCR Result: %s", b, result.c_str());
        ctx.regResults.emplace_back(std::move(result));
    }

    m_timer->stopCpu();
    ctx.postTime = m_timer->durationCpu<timer::Timer::ms>("Recognizer postprocess(CPU)");
    return true;
}

shared_ptr<Recognizer> makeRecognizer(ModelParams &params, logger::Level level)
{
    auto recognizer = make_shared<Recognizer>(params, level);
    recognizer->initModel();
    return recognizer;
}

}; // namespace recognizer

}; // namespace model
