# OmniDrive ONNX Export Environment Setup (Python 3.9, CUDA 11.7)

This guide records a validated environment for OmniDrive vision ONNX export.
It is based on actual installation and runtime checks in this workspace.

## 1. Prerequisites

- Linux with NVIDIA GPU and CUDA driver available
- Conda installed
- Repository root at /OmniDrive

## 2. Create and Activate Environment

```bash
cd /OmniDrive
source ~/.bashrc
conda create -n omnidrive_onnx python=3.9 -y
conda activate omnidrive_onnx
python -m pip install --upgrade pip
```

## 3. Install PyTorch (CUDA 11.7)

Use the CUDA-enabled wheels (not CPU-only):

```bash
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
```

## 4. Install OpenMMLab Core Stack

```bash
pip install --no-build-isolation mmcv-full==1.6.2
pip install mmdet==2.28.2 mmsegmentation==0.30.0
pip install --no-build-isolation -e /OmniDrive/mmdetection3d
```

Notes:
- The editable install target is /OmniDrive/mmdetection3d.
- Do not use /OmniDrive/projects/mmdet3d_plugin as editable target (not an installable package root).

## 5. Install ONNX Toolchain and Transformers

```bash
pip install transformers==4.31.0
pip install onnx==1.14.0 onnxruntime-gpu==1.15.1 onnxsim==0.4.33
pip install --upgrade onnx-graphsurgeon
```

## 6. Apply Compatibility Pins (Important)

Some transitive dependencies can break runtime compatibility with torch 1.13.1.
Apply these pins after the stack installation:

```bash
pip install "numpy<2" "opencv-python<4.10"
pip install numba==0.59.1 llvmlite==0.42.0 networkx==2.8.8
```

Why:
- torch 1.13.1 + torchvision 0.14.1 is not stable with NumPy 2.x ABI.
- Latest opencv-python may force NumPy 2.x.
- mmdetection3d transitive old numba/networkx pins are not fully compatible with this Python/runtime combination.

## 7. Optional OpenLane Package

If your code path imports openlanev2 package symbols:

```bash
pip install --no-deps -e /OmniDrive/OpenLane-V2
```

## 8. Validate Installation

```bash
python - <<'PY'
import torch, torchvision, torchaudio, onnxruntime as ort
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())
print('torchvision:', torchvision.__version__)
print('torchaudio:', torchaudio.__version__)
print('onnxruntime providers:', ort.get_available_providers())
PY
```

Expected key points:
- torch reports 1.13.1+cu117
- cuda available is True
- providers include CUDAExecutionProvider

## 9. Export Script CLI Check

```bash
PYTHONPATH=/OmniDrive python /OmniDrive/deploy/export_vision.py -h
```

If help text prints successfully, the CLI entry path is healthy.

## 10. Known Optional Dependency

- flash-attn is not required for this ONNX export flow.
- If needed later, verify compatibility separately with your exact torch/CUDA build.
