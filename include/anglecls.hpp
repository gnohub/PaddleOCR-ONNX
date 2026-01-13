#ifndef __ANGLECLS_HPP__
#define __ANGLECLS_HPP__

#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "common.hpp"
#include "logger.hpp"
#include "model.hpp"

namespace model{

namespace anglecls {
class Anglecls : public Model{
public:
    Anglecls(ModelParams &params, logger::Level level);

public:
    virtual void setup(void const* data, std::size_t size) override;
    virtual bool preProcessCpu(InferContext& ctx) override;
    virtual bool postProcessCpu(InferContext& ctx) override;
    virtual bool preProcessCuda(InferContext& ctx) override;
    virtual bool postProcessCuda(InferContext& ctx) override;
private:
    int                                     m_channels  = 3;
    int                                     m_dstHeight = 80;
    int                                     m_dstWidth  = 160;
    float                                   m_scale     = 0.00392156862745098f;
};

std::shared_ptr<Anglecls> makeAnglecls(ModelParams &params, logger::Level level);

}; // namespace anglecls
}; // namespace model

#endif //__ANGLECLS_HPP__
