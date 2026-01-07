CONFIG  := ./config/Makefile.config
include $(CONFIG)

# =========================
# Device Config
# =========================
ifeq ($(INFTER_BACKEND), ORT_CPU)
	INFTER_BACKEND_ID := 0
endif

ifeq ($(INFTER_BACKEND),ORT_CUDA)
	INFTER_BACKEND_ID := 1
endif

ifeq ($(INFTER_BACKEND),TRT)
	INFTER_BACKEND_ID := 2
endif

# =========================
# Core Source Files (NO main)
# =========================
CXX_SRC     := $(filter-out $(SRC_PATH)/main.cpp, $(wildcard $(SRC_PATH)/*.cpp))
KERNELS_SRC := $(wildcard $(SRC_PATH)/*.cu)

APP_OBJS    := $(patsubst $(SRC_PATH)/%, $(BUILD_PATH)/%, $(CXX_SRC:.cpp=.cpp.o))
APP_OBJS    += $(patsubst $(SRC_PATH)/%, $(BUILD_PATH)/%, $(KERNELS_SRC:.cu=.cu.o))

APP_MKS     := $(APP_OBJS:.o=.mk)

APP_DEPS    := $(CXX_SRC) \
               $(KERNELS_SRC) \
               $(wildcard $(SRC_PATH)/*.h)

# =========================
# Executable Entry Sources
# =========================
MAIN_SRC        := $(SRC_PATH)/main.cpp
MAIN_OBJ        := $(BUILD_PATH)/main.o

BENCH_SRC       := benchmark/benchmark.cpp
BENCH_OBJ       := $(BUILD_PATH)/benchmark.o
BENCH_APP       := benchmark

# =========================
# Compiler
# =========================
CUCC      := $(CUDA_DIR)/bin/nvcc
CXXFLAGS  := -std=c++14 -pthread -fPIC -DINFTER_BACKEND_ID=$(INFTER_BACKEND_ID)
CUDAFLAGS := -Xcompiler -fPIC

# =========================
# Include / Lib
# =========================
INCS := -I $(SRC_PATH) \
        -I $(INC_PATH) \
        -I $(OPENCV_DIR) \
        -I $(ONNXRUNTIME_DIR)/include

LIBS := -lstdc++fs \
        -L "$(ONNXRUNTIME_DIR)/lib" -lonnxruntime \
        `pkg-config --libs opencv4`

ifeq ($(INFTER_BACKEND),TRT)
	INCS += -I $(CUDA_DIR)/include \
	        -I $(TENSORRT_DIR)/include
	LIBS += -L "$(CUDA_DIR)/lib64" -lcudart -lcublas -lcudnn
endif

ifeq ($(INFTER_BACKEND),ORT_CUDA)
	INCS += -I $(CUDA_DIR)/include
	LIBS += -L "$(TENSORRT_DIR)/lib" -lnvinfer -lnvonnxparser \
	        -L "$(CUDA_DIR)/lib64" -lcudart -lcublas -lcudnn
endif

# =========================
# Debug / Release
# =========================
ifeq ($(DEBUG),1)
	CXXFLAGS  += -g -O0
	CUDAFLAGS += -g -O0
else
	CXXFLAGS  += -O3
	CUDAFLAGS += -O3
endif

# =========================
# Warning
# =========================
ifeq ($(SHOW_WARNING),1)
	CXXFLAGS  += -Wall -Wunused-function -Wunused-variable -Wfatal-errors
	CUDAFLAGS += -Wall -Wunused-function -Wunused-variable -Wfatal-errors
else
	CXXFLAGS  += -w
	CUDAFLAGS += -w
endif

# =========================
# Default Target
# =========================
all: \
	$(BIN_DIR)/$(APP) \
	$(BIN_DIR)/$(BENCH_APP) \
	$(LIB_DIR)/lib$(LIB_NAME).a \
	$(LIB_DIR)/lib$(LIB_NAME).so 
	@mkdir -p output
	@echo "Build finished!!!"

# =========================
# Executables
# =========================
$(BIN_DIR)/$(APP): $(MAIN_OBJ) $(APP_OBJS)
	@mkdir -p $(BIN_DIR)
	@echo "Link EXE $@"
	@$(CXX) $^ -o $@ $(LIBS)

$(BIN_DIR)/$(BENCH_APP): $(BENCH_OBJ) $(APP_OBJS)
	@mkdir -p $(BIN_DIR)
	@echo "Link BENCHMARK $@"
	@$(CXX) $^ -o $@ $(LIBS)

# =========================
# Libraries
# =========================
$(LIB_DIR)/lib$(LIB_NAME).a: $(APP_OBJS)
	@mkdir -p $(LIB_DIR)
	@echo "Archive STATIC $@"
	@ar rcs $@ $^

$(LIB_DIR)/lib$(LIB_NAME).so: $(APP_OBJS)
	@mkdir -p $(LIB_DIR)
	@echo "Link SHARED $@"
	@$(CXX) -shared -o $@ $^ $(LIBS)

# =========================
# Compile Rules
# =========================
$(BUILD_PATH)/%.cpp.o: $(SRC_PATH)/%.cpp
	@mkdir -p $(BUILD_PATH)
	@echo "Compile CXX $<"
	@$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCS)

$(BUILD_PATH)/main.o: $(MAIN_SRC)
	@mkdir -p $(BUILD_PATH)
	@echo "Compile MAIN $<"
	@$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCS)

$(BUILD_PATH)/benchmark.o: $(BENCH_SRC)
	@mkdir -p $(BUILD_PATH)
	@echo "Compile BENCHMARK MAIN $<"
	@$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCS)

$(BUILD_PATH)/%.cu.o: $(SRC_PATH)/%.cu
	@mkdir -p $(BUILD_PATH)
	@echo "Compile CUDA $<"
	@$(CUCC) -c $< -o $@ $(CUDAFLAGS) $(INCS)

# =========================
# Utils
# =========================
run: $(BIN_DIR)/$(APP)
	@./$(BIN_DIR)/$(APP)

run_bench: $(BIN_DIR)/$(BENCH_APP)
	@./$(BIN_DIR)/$(BENCH_APP)

clean:
	rm -rf $(BUILD_PATH) $(BIN_DIR) $(LIB_DIR)
	rm -rf output
	rm -rf config/compile_commands.json

ifneq ($(MAKECMDGOALS),clean)
-include $(APP_MKS)
endif

.PHONY: all run run_bench clean
