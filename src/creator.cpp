#include <memory>
#include "creator.hpp"
#include "logger.hpp"
#include "detectioner.hpp"
#include "recognizer.hpp"
#include "anglecls.hpp"

namespace ocrcreator{

Creator::Creator(std::vector<model::ModelParams> &paramList, logger::Level level) {
    m_logger = logger::createLogger(level);

    for (auto params : paramList) {
        switch (params.task) {
            case common::task_type::DETECTION:
                m_detectioner = model::detectioner::makeDetectioner(params, level);
                break;
            case common::task_type::RECOGNIZE:
                m_recognizer = model::recognizer::makeRecognizer(params, level);
                break;
            case common::task_type::ANGLECLS:
                m_anglecls = model::anglecls::makeAnglecls(params, level);
                break;
            default:
                LOGE("Unsupported task type");
                break;
        }
    }
}

std::shared_ptr<model::InferResult> Creator::inference(const std::string &imagePath) {
    auto rets = std::make_shared<model::InferResult>();
    model::InferContext det_ctx;
    if (m_detectioner) {
        m_detectioner->inference(det_ctx, imagePath);
        rets->preTime   += det_ctx.preTime;
        rets->inferTime += det_ctx.inferTime;
        rets->postTime  += det_ctx.postTime;
    }

    if (m_anglecls) {
        m_anglecls->inference(det_ctx, imagePath);
        rets->preTime   += det_ctx.preTime;
        rets->inferTime += det_ctx.inferTime;
        rets->postTime  += det_ctx.postTime;
    }

    if (m_recognizer) {
        model::InferContext rec_ctx;
        rec_ctx.imagePath = imagePath;

        if (!det_ctx.roiMats.empty()) {
            int num_rois = static_cast<int>(det_ctx.roiMats.size());
            rets->regRets.reserve(num_rois);

            for (int i = 0; i < num_rois; ++i) {
                rec_ctx.roiMats.clear();
                rec_ctx.roiMats.emplace_back(det_ctx.roiMats[i]);

                rec_ctx.roiRoutes.clear();
                rec_ctx.roiRoutes.emplace_back(det_ctx.roiRoutes[i]);

                m_recognizer->inference(rec_ctx, imagePath);

                if (!rec_ctx.regResults.empty()) {
                    rets->regRets.push_back(std::move(rec_ctx.regResults[0]));
                }

                rets->preTime   += rec_ctx.preTime;
                rets->inferTime += rec_ctx.inferTime;
                rets->postTime  += rec_ctx.postTime;
            }
        } else {
            m_recognizer->inference(rec_ctx, imagePath);
            rets->regRets = std::move(rec_ctx.regResults);

            rets->preTime   += rec_ctx.preTime;
            rets->inferTime += rec_ctx.inferTime;
            rets->postTime  += rec_ctx.postTime;
        }
    }

    // Multi-batch accuracy drops and speed decreases; batch needs to be organized
    // if (m_recognizer != nullptr) {
    //     m_recognizer->inference(det_ctx, imagePath);
    //     rets->preTime += dec_ctx.preTime;
    //     rets->inferTime += det_ctx.inferTime;
    //     rets->postTime += det_ctx.postTime;
    // }

    if (m_detectioner) {
        rets->decBoxes = std::move(det_ctx.boxes);
        rets->decRets  = std::move(det_ctx.roiMats);
    }

    if (m_anglecls) {
        rets->angleRets = std::move(det_ctx.roiRoutes);
    }
    return rets;
}

std::shared_ptr<Creator> createCreator(std::vector<model::ModelParams> &paramList, logger::Level level) 
{
    return std::make_shared<Creator>(paramList, level);
}

}; // namespace ocrcreator
