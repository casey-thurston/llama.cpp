# llama.cpp + voxtral-stream build environment for Jetson Orin (SM87, aarch64).
#
# Extends our fully-baked orin-ml-base image with cmake / build-essential / git
# so we can run `cmake -B build -DGGML_CUDA=ON ...` inside it. This image is
# build-only — it's not the runtime image for voxtral-stream.

FROM orin-ml-base:r36.5.tegra-aarch64-cu126-22.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        git \
        pkg-config \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

WORKDIR /src
