#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#define INFER_ORT_CPU  0
#define INFER_ORT_CUDA 1
#define INFER_TRT  2

#define WORKSPACESIZE 1<<28

namespace common{
    enum task_type {
        DETECTION = 0,
        ANGLECLS,
        RECOGNIZE,
        OCR,
    };

    enum precision {
        FP32,
        FP16,
        INT8,
    };

    enum infer_backend {
        ORT_CPU,
        ORT_CUDA,
        TRT,
    };

};

#endif //__COMMON_HPP__
