#ifndef __RECOGNIZER_HPP__
#define __RECOGNIZER_HPP__

#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "common.hpp"
#include "logger.hpp"
#include "model.hpp"

namespace model{

namespace recognizer {
class Recognizer : public Model{
public:
    Recognizer(ModelParams &params, logger::Level level);

public:
    virtual void setup(void const* data, std::size_t size) override;
    virtual bool preProcessCpu(InferContext& ctx) override;
    virtual bool postProcessCpu(InferContext& ctx) override;
    virtual bool preProcessCuda(InferContext& ctx) override;
    virtual bool postProcessCuda(InferContext& ctx) override;
private:
    int                                     m_channels  = 3;
    int                                     m_dstHeight = 48;
    int                                     m_dstWidth  = 320;
    float                                   m_scale;
    std::vector<std::string>                m_characterList;
    std::unordered_map<int, std::string>    m_keys;
};

std::shared_ptr<Recognizer> makeRecognizer(ModelParams &params, logger::Level level);

}; // namespace recognizer
}; // namespace model

#endif //__RECOGNIZER_HPP__
