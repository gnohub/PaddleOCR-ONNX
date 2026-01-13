#include <string>
#include <iostream>
#include "utils.hpp" 
#include "model.hpp"
#include "logger.hpp"

using namespace std;
namespace model{

Model::Model(ModelParams &params, logger::Level level):m_inputName(nullptr, &free), m_outputName(nullptr, &free){
    m_logger        = make_shared<logger::Logger>(level);
    m_timer         = make_shared<timer::Timer>();
    m_params        = new ModelParams(params);
    assert(fileExists(m_params->onnxPath));
    assert(fileExists(m_params->inferYaml));
    LOG("Model:%s", getFileName(m_params->onnxPath).c_str());
}

void Model::loadData(){}

void Model::initModel() {
    if ( (m_params->inferBackend == common::infer_backend::ORT_CPU || m_params->inferBackend == common::infer_backend::ORT_CUDA)
     && m_onnxSession == nullptr) {
        // init onnx runtime
        m_onnxOptions.SetInterOpNumThreads(m_params->interThreadnum);
        m_onnxOptions.SetIntraOpNumThreads(m_params->intraThreadnum);
        m_onnxOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        m_onnxSession = std::make_shared<Ort::Session>(m_onnxEnv, m_params->onnxPath.c_str(), m_onnxOptions);

#if INFTER_BACKEND_ID == INFER_ORT_CUDA
        if(m_params->inferBackend == common::infer_backend::ORT_CUDA)
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(m_onnxOptions, 0));
#endif
        Ort::AllocatorWithDefaultOptions allocator;

        Ort::AllocatedStringPtr         inputptr    = m_onnxSession->GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr         outputptr   = m_onnxSession->GetOutputNameAllocated(0, allocator);
        m_inputName.reset(strdup(inputptr.get()));
        m_outputName.reset(strdup(outputptr.get()));

        auto input_type_info = m_onnxSession->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_dims = input_tensor_info.GetShape();
        std::ostringstream input_oss;
        input_oss << "[";
        for (size_t i = 0; i < input_dims.size(); ++i) {
            if (i > 0) input_oss << ", ";
            input_oss << input_dims[i];
        }
        input_oss << "]";

        LOG("Input Name:%s", m_inputName.get());
        LOG("Input Shape:%s", input_oss.str().c_str());

        auto output_type_info = m_onnxSession->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_dims = output_tensor_info.GetShape();
        std::ostringstream output_oss;
        output_oss << "[";
        for (size_t i = 0; i < output_dims.size(); ++i) {
            if (i > 0) output_oss << ", ";
            output_oss << output_dims[i];
        }
        output_oss << "]";

        LOG("Output Name:%s", m_outputName.get());
        LOG("Output Shape:%s", output_oss.str().c_str());
        setup(nullptr, 0x00);
    }
}

void Model::inference(InferContext& ctx, std::string imagePath) {
    ctx.imagePath = imagePath;
    assert(fileExists(imagePath));
    if (m_params->inferBackend == common::infer_backend::ORT_CPU) {
        preProcessCpu(ctx);
    }else if(m_params->inferBackend == common::infer_backend::ORT_CUDA){
        preProcessCuda(ctx);
    }

    enqueueBindings(ctx);

    if (m_params->inferBackend == common::infer_backend::ORT_CPU) {
        postProcessCpu(ctx);
    }else if(m_params->inferBackend == common::infer_backend::ORT_CUDA){
        postProcessCuda(ctx);
    }
}

bool Model::enqueueBindings(InferContext& ctx) {
    m_timer->startCpu();
    const char* inputNames[]  = { m_inputName.get() };
    const char* outputNames[] = { m_outputName.get() };
    
    if(ctx.inputTensor == nullptr){
        LOGD("inputTensor is nullptr!!");
        return false;
    }
    ctx.outputTensor = m_onnxSession->Run(Ort::RunOptions{nullptr}, inputNames, &ctx.inputTensor, 1, outputNames, 1);
    m_timer->stopCpu();
    ctx.inferTime = m_timer->durationCpu<timer::Timer::ms>("enqueue_bindings(CPU)");
    return true;
}

} // namespace model
