#ifndef __WORKER_HPP__
#define __WORKER_HPP__

#include <memory>
#include <vector>
#include <map>
#include "model.hpp"
#include "logger.hpp"

namespace ocrcreator{

class Creator {
public:
    Creator(std::vector<model::ModelParams> &paramList, logger::Level level);
    std::shared_ptr<model::InferResult> inference(const std::string &imagePath);

private:
    std::shared_ptr<logger::Logger>     m_logger;
    std::shared_ptr<model::Model>       m_detectioner;
    std::shared_ptr<model::Model>       m_recognizer;
    std::shared_ptr<model::Model>       m_anglecls;
};

std::shared_ptr<Creator> createCreator(std::vector<model::ModelParams> &paramList, logger::Level level);

}; //namespace ocrcreator

#endif //__WORKER_HPP__
