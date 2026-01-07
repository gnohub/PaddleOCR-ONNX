#include <chrono>
#include <iostream>
#include <memory>

#include "timer.hpp"

namespace timer {

Timer::Timer(){
    _timeElasped = 0;
    _cStart = std::chrono::high_resolution_clock::now();
    _cStop = std::chrono::high_resolution_clock::now();
}

Timer::~Timer(){
}

void Timer::startCpu() {
    _cStart = std::chrono::high_resolution_clock::now();
}

void Timer::stopCpu() {
    _cStop = std::chrono::high_resolution_clock::now();
}

} //namespace timer
