#!/bin/bash
# Build a patched Mesa dzn (Dozen) Vulkan driver for WSL2 GPU inference.
#
# The dzn driver translates Vulkan calls to D3D12, which WSL2 routes to
# the host GPU. Stock Mesa dzn lacks extensions wgpu-native requires
# (VK_EXT_robustness2, etc.) and reports conservative buffer limits.
# This script applies patches/mesa-dzn-wgpu-compat.patch to fix both.
#
# Prerequisites:
#   sudo apt-get install -y meson libdrm-dev libelf-dev llvm-dev \
#     libexpat1-dev directx-headers-dev ninja-build python3-mako
#   pip3 install --user meson  # need meson >= 1.4
#
# Usage:
#   ./patches/build-dzn.sh
#
# Output:
#   /tmp/mesa-dzn/build/src/microsoft/vulkan/dzn_devenv_icd.x86_64.json
#
# Then run bitnet with:
#   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
#   VK_ICD_FILENAMES=/tmp/mesa-dzn/build/src/microsoft/vulkan/dzn_devenv_icd.x86_64.json \
#   ./bitnet model.gguf --gpu --maxseq 4096 -p "Hello" -n 64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MESA_DIR="/tmp/mesa-dzn"
PATCH="$REPO_DIR/patches/mesa-dzn-wgpu-compat.patch"

# Find meson (prefer pip-installed version)
MESON="${HOME}/.local/bin/meson"
if [ ! -x "$MESON" ]; then
    MESON="$(command -v meson)"
fi

echo "=== Checking prerequisites ==="
for cmd in "$MESON" ninja cc; do
    command -v "$cmd" >/dev/null || { echo "Missing: $cmd"; exit 1; }
done
"$MESON" --version | awk -F. '{if ($1 < 1 || ($1 == 1 && $2 < 4)) { print "meson >= 1.4 required (have " $0 ")"; exit 1 }}'

echo "=== Cloning Mesa (shallow) ==="
if [ -d "$MESA_DIR/.git" ]; then
    echo "Mesa source already present at $MESA_DIR"
else
    rm -rf "$MESA_DIR"
    git clone --depth 1 https://gitlab.freedesktop.org/mesa/mesa.git "$MESA_DIR"
fi

echo "=== Applying patch ==="
cd "$MESA_DIR"
git checkout -- . 2>/dev/null || true
git apply "$PATCH"

echo "=== Configuring (dzn driver only) ==="
rm -rf build
"$MESON" setup build \
    -Dvulkan-drivers=microsoft-experimental \
    -Dgallium-drivers= \
    -Dllvm=enabled \
    -Dplatforms= \
    -Dglx=disabled \
    -Degl=disabled \
    -Dgbm=disabled \
    -Dprefix="$MESA_DIR/install"

echo "=== Building ==="
ninja -C build -j"$(nproc)"

echo ""
echo "=== Done ==="
echo "ICD JSON: $MESA_DIR/build/src/microsoft/vulkan/dzn_devenv_icd.x86_64.json"
echo ""
echo "Run bitnet with GPU on WSL2:"
echo "  LD_LIBRARY_PATH=/usr/lib/wsl/lib \\"
echo "  VK_ICD_FILENAMES=$MESA_DIR/build/src/microsoft/vulkan/dzn_devenv_icd.x86_64.json \\"
echo "  ./bitnet model.gguf --gpu --maxseq 4096 -p \"Hello\" -n 64"
