#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>

#include "logger.hpp"
#include "creator.hpp"
#include "utils.hpp"

struct StatsNode{
    common::task_type       task;
    common::infer_backend   inferBackend;
    std::string             filename;
    int                     intraThnum;
    int                     interThnum;
    int                     warmupIters;
    int                     benchIters;
    double                  avgPre;
    double                  avgInfer; 
    double                  avgPost;
    double                  avgTotal;
    double                  p90Total;
    double                  p99Total;
};

static double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * v.size());
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

static std::string task2str(const common::task_type task){
    switch(task){
        case common::task_type::DETECTION:
            return "Detection";
        case common::task_type::ANGLECLS:
            return "Anglecls";
        case common::task_type::RECOGNIZE:
            return "Recognize";
        default:
            return "OCR";
    }
}

static std::string backend2str(const common::infer_backend b) {
    switch (b) {
        case common::infer_backend::ORT_CUDA: return "ORT-CUDA";
        case common::infer_backend::TRT:      return "TRT";
        default:                      return "ORT-CPU";
    }
}

void statsShow(const StatsNode& stats){
    std::cout << "============================== Benchmark ==============================\n";
    std::cout << "FileName              : " << stats.filename << "\n";
    std::cout << "Model                 : " << task2str(stats.task)<<"\n";
    std::cout << "Infer backend         : " << backend2str(stats.inferBackend) <<"\n";
    std::cout << "Warmup iters          : " << stats.warmupIters << "\n";
    std::cout << "Bench iters           : " << stats.benchIters << "\n";
    std::cout << "ORT intra thread nnum : " << stats.intraThnum << "\n";
    std::cout << "ORT iter thread num   : " << stats.interThnum << "\n\n";

    std::cout << "[Average]\n";
    std::cout << "Preprocess            : " << stats.avgPre << " ms\n";
    std::cout << "Inference             : " << stats.avgInfer << " ms\n";
    std::cout << "Postprocess           : " << stats.avgPost << " ms\n";
    std::cout << "Total                 : " << stats.avgTotal << " ms\n\n";

    std::cout << "[P90]\n";
    std::cout << "Total                 : " << stats.p90Total << " ms\n";

    std::cout << "[P99]\n";
    std::cout << "Total                 : " << stats.p99Total << " ms\n";

    std::cout << "=======================================================================\n";
}

void exportCSV(const common::task_type task, const std::vector<StatsNode>& statsTab) {
    std::string filename = "output/benchmark/" + task2str(task) + ".csv";
    std::ofstream ofs(filename);
    ofs << "Filename,Infer-Backend,Intra-Thread,Inter-Thread,AvgPre(ms),AvgInfer(ms),AvgPost(ms),AvgTotal(ms),P90Total(ms),P99total(ms)\n";
    for (size_t i = 0; i < statsTab.size(); ++i) {
        ofs << std::left << std::setw(20) << statsTab[i].filename << ","
        << std::setw(12) << backend2str(statsTab[i].inferBackend) << ","
        << std::setw(8) << statsTab[i].intraThnum << ","
        << std::setw(8) << statsTab[i].interThnum << ","
        << std::setw(12) << statsTab[i].avgPre << ","
        << std::setw(12) << statsTab[i].avgInfer << ","
        << std::setw(12) << statsTab[i].avgPost << ","
        << std::setw(12) << statsTab[i].avgTotal << ","
        << std::setw(12) << statsTab[i].p90Total << ","
        << std::setw(12) << statsTab[i].p99Total
        << "\n";
    }
    ofs.close();
}

std::shared_ptr<StatsNode> benchmark(const std::string imagePath, int intraThnum, int interThnum, common::task_type task) {
    const int warmup_iters = 10;
    const int bench_iters  = 100;
    const auto log_level = logger::Level::ERROR;
    std::shared_ptr<StatsNode> stats = std::make_shared<StatsNode>();

    model::ModelParams det;
    det.task                = common::task_type::DETECTION;
    det.inferBackend        = common::infer_backend::ORT_CPU;
    det.saveImg             = false;
    det.onnxPath            = "models/PP-OCRv5_mobile_det_infer/inference.onnx";
    det.inferYaml           = "models/PP-OCRv5_mobile_det_infer/inference.yml";
    det.intraThreadnum      = intraThnum;
    det.interThreadnum      = interThnum;

    model::ModelParams angle;
    angle.task              = common::task_type::ANGLECLS;
    angle.inferBackend      = common::infer_backend::ORT_CPU;
    angle.saveImg           = false;
    angle.onnxPath          = "models/PP-LCNet_x1_0_textline_ori_infer/inference.onnx";
    angle.inferYaml         = "models/PP-LCNet_x1_0_textline_ori_infer/inference.yml";
    angle.intraThreadnum    = intraThnum;
    angle.interThreadnum    = interThnum;

    model::ModelParams rec;
    rec.task                = common::task_type::RECOGNIZE;
    rec.inferBackend        = common::infer_backend::ORT_CPU;
    rec.saveImg             = false;
    rec.onnxPath            = "models/PP-OCRv5_mobile_rec_infer/inference.onnx";
    rec.inferYaml           = "models/PP-OCRv5_mobile_rec_infer/inference.yml";
    rec.intraThreadnum      = intraThnum;
    rec.interThreadnum      = interThnum;

    std::vector<model::ModelParams> params;
    if (task == common::task_type::OCR) {
        params.emplace_back(det);
        params.emplace_back(angle);
        params.emplace_back(rec);
    } else if (task == common::task_type::DETECTION) {
        params.emplace_back(det);
    } else if (task == common::task_type::ANGLECLS) {
        params.emplace_back(angle);
    } else if (task == common::task_type::RECOGNIZE) {
        params.emplace_back(rec);
    }

    auto creator = ocrcreator::createCreator(params, log_level);

    for (int i = 0; i < warmup_iters; ++i) {
        creator->inference(imagePath);
    }

    std::vector<double> pre_times;
    std::vector<double> infer_times;
    std::vector<double> post_times;
    std::vector<double> total_times;

    pre_times.reserve(bench_iters);
    infer_times.reserve(bench_iters);
    post_times.reserve(bench_iters);
    total_times.reserve(bench_iters);

    for (int i = 0; i < bench_iters; ++i) {
        auto rets = creator->inference(imagePath);

        const double pre   = rets->preTime;
        const double infer = rets->inferTime;
        const double post  = rets->postTime;
        const double total = pre + infer + post;

        pre_times.emplace_back(pre);
        infer_times.emplace_back(infer);
        post_times.emplace_back(post);
        total_times.emplace_back(total);
    }

    stats->inferBackend = common::infer_backend::ORT_CPU;
    stats->task         = task;
    stats->filename     = imagePath;
    stats->warmupIters  = warmup_iters;
    stats->benchIters   = bench_iters;
    stats->intraThnum   = intraThnum;
    stats->interThnum   = interThnum;
    stats->avgPre       = mean(pre_times);
    stats->avgInfer     = mean(infer_times);
    stats->avgPost      = mean(post_times);
    stats->avgTotal     = mean(total_times);
    stats->p90Total     = percentile(total_times, 0.90);
    stats->p99Total     = percentile(total_times, 0.99);
    return stats;
}

int main() {

    std::vector<int> values = {1, 2, 4, 8};
    
    if (!ensure_dir("output/benchmark")) {
        return -1;
    }

    // 文本检测
    std::vector<StatsNode> stats_array;
    const std::string dec_image_path = "data/images/general_ocr_0.png";
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = 0; j < values.size(); ++j) {
            auto bk = benchmark(dec_image_path, values[i], values[j], common::task_type::DETECTION);
            statsShow(*bk);
            stats_array.emplace_back(std::move(*bk));
        }
    }
    exportCSV(common::task_type::DETECTION, stats_array);
    
    // 文本角度识别
    stats_array.clear();
    const std::string  angle_image_path = "data/images/angle.png";
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = 0; j < values.size(); ++j) {
            auto bk = benchmark(angle_image_path, values[i], values[j], common::task_type::ANGLECLS);
            statsShow(*bk);
            stats_array.emplace_back(std::move(*bk));
        }
    }
    exportCSV(common::task_type::ANGLECLS, stats_array);

    // 文本识别
    stats_array.clear();
    const std::string reg_image_path = "data/images/reg.png";
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = 0; j < values.size(); ++j) {
            auto bk = benchmark(reg_image_path, values[i], values[j], common::task_type::RECOGNIZE);
            statsShow(*bk);
            stats_array.emplace_back(std::move(*bk));
        }
    }
    exportCSV(common::task_type::RECOGNIZE, stats_array);

    // OCR
    stats_array.clear();
    const std::string ocr_image_path = "data/images/general_ocr_0.png";
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t j = 0; j < values.size(); ++j) {
            auto bk = benchmark(ocr_image_path, values[i], values[j], common::task_type::OCR);
            statsShow(*bk);
            stats_array.emplace_back(std::move(*bk));
        }
    }
    exportCSV(common::task_type::OCR, stats_array);

    return 0;
}