#! /bin/bash
# Wheel build environment (cibuildwheel CIBW_BEFORE_ALL, manylinux_2_28 x86_64): the CUDA
# 12.8 compiler and headers, then the library built the way the upstream ctranslate2
# wheels are: no cuDNN (conv1d runs CTranslate2's own kernel), CUDA loaded at runtime
# (cudart static, cuBLAS through the stub), so outputs match the upstream wheel. Native
# code for Ada (sm_89) and Blackwell (sm_120), plus compute_120 PTX.

set -e
set -x

pip install "cmake==3.22.*"

dnf install -y dnf-plugins-core
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
# error mirrorlist.centos.org doesn't exists anymore.
sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
dnf install --setopt=obsoletes=0 -y \
    cuda-nvcc-12-8-12.8.93-1 \
    cuda-cudart-devel-12-8-12.8.90-1 \
    libcurand-devel-12-8-10.3.9.90-1 \
    libcublas-devel-12-8-12.8.4.1-1
ln -s cuda-12.8 /usr/local/cuda

mkdir build-release && cd build-release
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DCUDA_DYNAMIC_LOADING=ON \
    -DCUDA_NVCC_FLAGS="-Xfatbin=-compress-all" \
    "-DCUDA_ARCH_FLAGS_OVERRIDE=-gencode;arch=compute_89,code=sm_89;-gencode;arch=compute_120,code=sm_120;-gencode;arch=compute_120,code=compute_120" \
    ..
VERBOSE=1 make -j$(nproc) install
cd ..
rm -r build-release

cp README.md python/
