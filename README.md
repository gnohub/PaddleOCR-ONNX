
# PaddleOCR ONNX C++推理

## 简介
PaddleOCR 的 OnnxRuntime 推理版本，目前只支持 **CPU 推理**。  
模型是直接使用 PaddleOCR 提供的教程导出的，参考链接: [获取 ONNX 模型](https://www.paddleocr.ai/latest/version3.x/deployment/obtaining_onnx_models.html)

## 模型版本

- **文本检测模型版本**：PP-OCRv5_mobile_det_infer
- **文本识别模型版本**：PP-OCRv5_mobile_rec_infer
- **文本方向分类模型版本**：PP-LCNet_x1_0_textline_ori_infer
- 模型及其配置文件总大小约 28MB

## 依赖环境

1. **操作系统**：Linux  
2. **C++ 标准**：C++14  
3. **OpenCV 版本**：4.6.0  
   - 参考：[OpenCV GitHub](https://github.com/opencv/opencv.git)  
4. **OnnxRuntime 版本**：1.18.0  
   - 参考：[OnnxRuntime GitHub](https://github.com/microsoft/onnxruntime)

## 编译步骤

1. 配置 `config/Makefile.config` 中 OpenCV 和 OnnxRuntime 依赖库的路径：

```makefile
OPENCV_DIR      := /usr/local/include/opencv4
ONNXRUNTIME_DIR := /usr/local/onnxruntime
```

2. 执行编译

```
make
```

## 示例参数

1. `--help`：示例程序帮助信息。  
2. `--task`：任务类型，每个模型都支持独立推理，`ocr` 选项是三个模型顺序推理。  
3. `--log_level`：日志等级，默认 `INFO` 级别。  
4. `--det_model`：文本检测模型文件路径，默认 `models/PP-OCRv5_mobile_det_infer/inference.onnx`。  
5. `--det_yaml`：文本检测模型配置文件路径，默认 `models/PP-OCRv5_mobile_det_infer/inference.yml`。  
6. `--angle_model`：文本方向分类模型文件路径，默认 `models/PP-LCNet_x1_0_textline_ori_infer/inference.onnx`。  
7. `--angle_yaml`：文本方向分类模型配置文件路径，默认 `models/PP-LCNet_x1_0_textline_ori_infer/inference.yml`。  
8. `--rec_model`：文本识别模型文件路径，默认 `models/PP-OCRv5_mobile_rec_infer/inference.onnx`。  
9. `--rec_yaml`：文本识别模型配置文件路径，默认 `models/PP-OCRv5_mobile_rec_infer/inference.yml`。  
10. `--infer_backend`：推理后端类型，当前版本只支持 ORT-CPU 推理，默认 `ORTCPU`。  
11. `--save_image`：是否保存推理过程中产生的临时图片，默认开启。  
12. `--image`：目标图片文件路径。  
13. `--intra_threads`：ORT 算子内部并发线程数，默认为 1。  
14. `--inter_threads`：ORT 算子间并发线程数，默认为 1。  

## 运行示例
```bash
./bin/testocr --image data/images/general_ocr_90.png
```
运行完示例程序后，在 `output` 目录下会生成三类临时图片文件：
- `dec_dst.png`：原图上的文本框标注  
- `det_mat_x.png`：从原图中裁剪出的文字区域图  
- `corrected_mat_x.png`：矫正为水平后的文字区域图 
 
## 测试环境

- **CPU**: i5-10400  
- **Warmup 轮次**: 10  
- **推理轮次**: 90

## 测试过程

- 测试过程包含不同模型的 **前处理（Preprocess）**、**推理（Inference）** 和 **后处理（Postprocess）** 的耗时统计。  
- 测试所用的目标图片存放在 `data/images` 目录下。  
- 通过执行 `./bin/benchmark` 进行测试。  
- 测试结果存储在 `output/benchmark` 目录下：
	- 测试结果 CSV 列说明：
		- Filename：图片文件名
		- Infer-Backend：推理后端类型
		- Intra-Thread / Inter-Thread：算子内部/间并发线程数
		- AvgPre / AvgInfer / AvgPost / AvgTotal：平均前处理 / 推理 / 后处理 / 总耗时 (ms)
		- P90Total / P99Total：总耗时 P90 / P99

## 其他说明

- 当前版本未启用 oneDNN，加上后推理速度会更快。
- 当前版本不支持 GPU 推理。