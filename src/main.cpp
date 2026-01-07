#include <iostream>
#include <string>
#include <vector>
#include <cstring>

#include "logger.hpp"
#include "creator.hpp"
#include "utils.hpp"

using namespace std;

void print_help() {
    cout << "Usage: testocr [options]\n\n";
    cout << "Options:\n";
    cout << "  --help                                Display this help message\n";
    cout << "  --task [dec/angle/reg/ocr]            Task type, default ocr\n";
    cout << "  --log_level [0-5]                     Log level (0=FATAL,1=ERROR,2=WARN,3=INFO,4=VERB,5=DEBUG), default INFO (3)\n";
    cout << "  --det_model [path]                    Path to detection model, default models/PP-OCRv5_mobile_det_infer/inference.onnx\n";
    cout << "  --det_yaml [path]                     Path to detection YAML config, default models/PP-OCRv5_mobile_det_infer/inference.yml\n";
    cout << "  --angle_model [path]                  Path to text angle cls model, default models/PP-LCNet_x1_0_textline_ori_infer/inference.onnx\n";
    cout << "  --angle_yaml [path]                   Path to text angle cls YAML config, default models/PP-LCNet_x1_0_textline_ori_infer/inference.yml\n";
    cout << "  --rec_model [path]                    Path to recognition model, default models/PP-OCRv5_mobile_rec_infer/inference.onnx\n";
    cout << "  --rec_yaml [path]                     Path to recognition YAML config, default models/PP-OCRv5_mobile_rec_infer/inference.yml\n";
    cout << "  --infer_backend [ORTCPU/ORTCUDA/TRT]  Inference infer backend type, default ORTCPU\n";
    cout << "  --save_image [0/1]                    Whether to save inference image, default 1\n";
    cout << "  --image [path]                        Path to the image for inference (required)\n";
    cout << "  --intra_threads [num]                 ORT intra-op threads, default 1\n";
    cout << "  --inter_threads [num]                 ORT inter-op threads, default 1\n";
}

common::task_type parse_task(const string &task_str) {
    if(task_str == "dec") return common::task_type::DETECTION;
    if(task_str == "angle") return common::task_type::ANGLECLS;
    if(task_str == "reg") return common::task_type::RECOGNIZE;
    if(task_str == "ocr") return common::task_type::OCR;
    cerr << "Unknown task type: " << task_str << ". Using ocr as default.\n";
    return common::task_type::OCR;
}

int main(int argc, char const *argv[])
{
    auto log_level           = logger::Level::INFO;
    string task_str         = "ocr";
    string det_model_path   = "models/PP-OCRv5_mobile_det_infer/inference.onnx";
    string det_yaml_path    = "models/PP-OCRv5_mobile_det_infer/inference.yml";

    string angle_model_path = "models/PP-LCNet_x1_0_textline_ori_infer/inference.onnx";
    string angle_yaml_path  = "models/PP-LCNet_x1_0_textline_ori_infer/inference.yml";

    string rec_model_path   = "models/PP-OCRv5_mobile_rec_infer/inference.onnx";
    string rec_yaml_path    = "models/PP-OCRv5_mobile_rec_infer/inference.yml";

    string infer_backend_str    = "ORTCPU";
    bool save_image             = true;
    string image_path           = "";
    int intra_threads           = 1;
    int inter_threads           = 1;

    for(int i = 1; i < argc; ++i) {
        if(strcmp(argv[i], "--help") == 0) {
            print_help();
            return 0;
        }
        if(strcmp(argv[i], "--log_level") == 0 && i + 1 < argc) {
            int lv = stoi(argv[++i]);
            if(lv < 0) lv = 0;
            if(lv > 5) lv = 5;
            log_level = static_cast<logger::Level>(lv);
        }
        else if(strcmp(argv[i], "--task") == 0 && i + 1 < argc) {
            task_str = argv[++i];
        }
        else if(strcmp(argv[i], "--det_model") == 0 && i + 1 < argc) {
            det_model_path = argv[++i];
        }
        else if(strcmp(argv[i], "--det_yaml") == 0 && i + 1 < argc) {
            det_yaml_path = argv[++i];
        }
        else if(strcmp(argv[i], "--angle_model") == 0 && i + 1 < argc) {
            angle_model_path = argv[++i];
        }
        else if(strcmp(argv[i], "--angle_yaml") == 0 && i + 1 < argc) {
            angle_yaml_path = argv[++i];
        }
        else if(strcmp(argv[i], "--rec_model") == 0 && i + 1 < argc) {
            rec_model_path = argv[++i];
        }
        else if(strcmp(argv[i], "--rec_yaml") == 0 && i + 1 < argc) {
            rec_yaml_path = argv[++i];
        }
        else if(strcmp(argv[i], "--infer_backend") == 0 && i + 1 < argc) {
            infer_backend_str = argv[++i];
        }
        else if(strcmp(argv[i], "--save_image") == 0 && i + 1 < argc) {
            save_image = (stoi(argv[++i]) != 0);
        }
        else if(strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        }
        else if(strcmp(argv[i], "--intra_threads") == 0 && i + 1 < argc) {
            intra_threads = stoi(argv[++i]);
        }
        else if(strcmp(argv[i], "--inter_threads") == 0 && i + 1 < argc) {
            inter_threads = stoi(argv[++i]);
        }
        else {
            cerr << "Unknown option: " << argv[i] << "\n";
            print_help();
            return 1;
        }
    }

    if(image_path.empty()) {
        cerr << "Error: --image is null\n";
        print_help();
        return 1;
    }
    
    if (!ensure_dir("output")) {
        return -1;
    }

    auto level = static_cast<logger::Level>(log_level);

    common::infer_backend infer_backend = common::infer_backend::ORT_CPU;
    if(infer_backend_str == "ORTCUDA") infer_backend = common::infer_backend::ORT_CUDA;
    else if(infer_backend_str == "TRT") infer_backend = common::infer_backend::TRT;

    auto det_params = model::ModelParams();
    det_params.task         = common::task_type::DETECTION;
    det_params.inferBackend = infer_backend;
    det_params.saveImg      = save_image;
    det_params.onnxPath     = det_model_path;
    det_params.inferYaml    = det_yaml_path;
    det_params.intraThreadnum = intra_threads;
    det_params.interThreadnum = inter_threads;

    auto angle_params = model::ModelParams();
    angle_params.task           = common::task_type::ANGLECLS;
    angle_params.inferBackend   = infer_backend;
    angle_params.saveImg        = save_image;
    angle_params.onnxPath       = angle_model_path;
    angle_params.inferYaml      = angle_yaml_path;
    angle_params.intraThreadnum = intra_threads;
    angle_params.interThreadnum = inter_threads;

    auto rec_params = model::ModelParams();
    rec_params.task         = common::task_type::RECOGNIZE;
    rec_params.inferBackend = infer_backend;
    rec_params.saveImg      = save_image;
    rec_params.onnxPath     = rec_model_path;
    rec_params.inferYaml    = rec_yaml_path;
    rec_params.intraThreadnum = intra_threads;
    rec_params.interThreadnum = inter_threads;

    std::vector<model::ModelParams> param_list;
    auto task = parse_task(task_str);
    if (task == common::task_type::OCR) {
        param_list.emplace_back(det_params);
        param_list.emplace_back(angle_params);
        param_list.emplace_back(rec_params);
    } else if (task == common::task_type::DETECTION) {
        param_list.emplace_back(det_params);
    } else if (task == common::task_type::ANGLECLS) {
        param_list.emplace_back(angle_params);
    } else if (task == common::task_type::RECOGNIZE) {
        param_list.emplace_back(rec_params);
    }

    auto creator = ocrcreator::createCreator(param_list, level);

    auto rets = creator->inference(image_path);
    for (size_t j = 0; j < rets->regRets.size(); ++j) {
        LOG("Batch[%zu] OCR Result: %s", j, rets->regRets[j].c_str());
    }
    LOG("Total preprocess time: %0.6lf ms", rets->preTime);
    LOG("Total inference time: %0.6lf ms", rets->inferTime);
    LOG("Total postprocess time: %0.6lf ms", rets->postTime);
    return 0;
}
