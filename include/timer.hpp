#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <chrono>
#include <ratio>
#include <string>
#include "common.hpp"
#include "logger.hpp"

namespace timer{

class Timer {
public:
    using s  = std::ratio<1, 1>;
    using ms = std::ratio<1, 1000>;
    using us = std::ratio<1, 1000000>;
    using ns = std::ratio<1, 1000000000>;

public:
    Timer();
    ~Timer();

public:
    void startCpu();
    void stopCpu();

    template <typename span>
    double durationCpu(std::string msg);

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStop;
    float _timeElasped;
};

template <typename span>
double Timer::durationCpu(std::string msg){
    std::string str;

    if(std::is_same<span, s>::value) { str = "s"; }
    else if(std::is_same<span, ms>::value) { str = "ms"; }
    else if(std::is_same<span, us>::value) { str = "us"; }
    else if(std::is_same<span, ns>::value) { str = "ns"; }

    std::chrono::duration<double, span> time = _cStop - _cStart;
    LOG("%-60s uses %.6lf %s", msg.c_str(), time.count(), str.c_str());
    return time.count();
}

} // namespace timer

#endif //__TIMER_HPP__
