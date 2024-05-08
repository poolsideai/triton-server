# Triton Inference Server - Poolside fork

[Original README](https://github.com/triton-inference-server/server/blob/main/README.md)

## Installation with TRT-LLM without containers for local development

Assuming Ubuntu 22.04 with CUDA 12.2 - those are the requirements for building TRT-LLM.
There are two steps: building TRT-LLM and then build Triton.

### Building TRT-LLM

Packages:

```
sudo apt install openmpi-bin libopenmpi-dev cuda-command-line-tools-12-2 cuda-nvcc-12-2 cuda-nvtx-12-2 libcublas-dev-12-2 libcurand-dev-12-2 libcufft-dev-12-2 libcusolver-dev-12-2 cuda-nvrtc-dev-12-2 libcusparse-dev-12-2 cuda-profiler-api-12-2 git-lfs
```

TensorRT:

```
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.3.0/tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-12.2.tar.gz
tar -xf tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-12.2.tar.gz
sudo mv TensorRT-9.3.0.1 /opt/TensorRT-9.3.0.1
rm tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-12.2.tar.gz
```

Clone TRT-LLM:

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/poolsideai/TensorRT-LLM
cd TensorRT-LLM
git submodule update --init
git config remote.origin.lfsurl https://github.com/nvidia/TensorRT-LLM.git/info/lfs
git lfs pull
```

Build:

```
./scripts/build_wheel.py  --skip_building_wheel --cuda_architectures 90-real --trt_root /opt/TensorRT-9.3.0.1/ --build_type RelWithDebInfo --extra-cmake-vars 'USE_CXX11_ABI=1' --cpp_only
```

### Building Triton

Packages:

```
sudo apt install zlib1g-dev libarchive-dev libxml2-dev libnuma-dev libre2-dev libssl-dev libgoogle-perftools-dev libb64-dev libcurl4-openssl-dev rapidjson-dev datacenter-gpu-manager=1:3.2.6 libcudnn8-dev
```

Symlink TRT-LLM source directory inside Triton build directory:

```
mkdir -p build/tensorrtllm
ln -s path/to/TensorRT-LLM/ build/tensorrtllm/tensorrt_llm
```

Build Triton:

```
./build.py -v --no-container-build --build-dir=$(pwd)/build --enable-logging --enable-stats --enable-metrics --enable-cpu-metrics --enable-gpu-metrics --enable-gpu --backend tensorrtllm  --endpoint http
```