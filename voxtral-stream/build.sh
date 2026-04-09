#!/usr/bin/env bash
#
# voxtral-stream/build.sh - one-stop build wrapper for the Jetson Orin build
# image. All llama.cpp / ggml / voxtral-stream compilation runs inside the
# `orin-ml-base:r36.5.tegra-aarch64-cu126-22.04-llamacpp` container so the host
# bare-metal stays clean.
#
# Subcommands:
#   image    Build (or rebuild) the docker build image.
#   build    Configure + build voxtral-stream (default).
#   clean    Remove the cmake build directory.
#   shell    Drop into a bash shell in the build container with the repo mounted.
#
# Usage:
#   ./build.sh             # same as `./build.sh build`
#   ./build.sh image
#   ./build.sh shell

set -euo pipefail

IMAGE_TAG="orin-ml-base:r36.5.tegra-aarch64-cu126-22.04-llamacpp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

cmd="${1:-build}"

ensure_image() {
    if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
        echo ">>> building docker image ${IMAGE_TAG}"
        docker build \
            -t "${IMAGE_TAG}" \
            -f "${SCRIPT_DIR}/build.Dockerfile" \
            "${SCRIPT_DIR}"
    fi
}

run_in_container() {
    docker run --rm -t \
        -v "${REPO_ROOT}:/src" \
        -w /src/voxtral-stream \
        "${IMAGE_TAG}" \
        "$@"
}

case "${cmd}" in
    image)
        # Force-rebuild the image (useful after editing build.Dockerfile)
        docker build \
            -t "${IMAGE_TAG}" \
            -f "${SCRIPT_DIR}/build.Dockerfile" \
            "${SCRIPT_DIR}"
        ;;

    build)
        ensure_image
        run_in_container bash -c '
            set -e
            mkdir -p build
            cmake -B build -S . \
                -DCMAKE_BUILD_TYPE=Release \
                -DGGML_CUDA=ON \
                -DCMAKE_CUDA_ARCHITECTURES=87
            cmake --build build -j$(nproc)
        '
        ;;

    clean)
        rm -rf "${BUILD_DIR}"
        echo "removed ${BUILD_DIR}"
        ;;

    shell)
        ensure_image
        docker run --rm -it \
            -v "${REPO_ROOT}:/src" \
            -w /src/voxtral-stream \
            "${IMAGE_TAG}" \
            bash
        ;;

    *)
        echo "unknown subcommand: ${cmd}" >&2
        echo "usage: $0 [image|build|clean|shell]" >&2
        exit 2
        ;;
esac
