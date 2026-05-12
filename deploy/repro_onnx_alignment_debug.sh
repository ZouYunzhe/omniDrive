#!/usr/bin/env bash
set -euo pipefail

# Reproduce ONNX vs PyTorch debug workflow (224 resolution) with extra debug outputs.
# Usage:
#   bash deploy/repro_onnx_alignment_debug.sh

ROOT_DIR="/omniDrive"
ENV_NAME="omnidrive_onnx"
CONFIG="projects/configs/OmniDrive/mask_eva_lane_det_vlm.py"
CKPT="/omniDrive/eva02_petr_proj.pth"
ONNX="onnxs/mask_eva_lane_det_vlm_eva02_gpu_224.onnx"

cd "${ROOT_DIR}"

echo "[1/4] Export ONNX with debug outputs"
PYTHONPATH=. conda run -n "${ENV_NAME}" python tools/test.py \
  "${CONFIG}" "${CKPT}" \
  --export-onnx --onnx-file "${ONNX}" \
  --launcher none --export-cams 6 --export-height 224 --export-width 224 \
  --cfg-options model.img_backbone.img_size=224

echo "[2/4] Verify bbox_pre_cls alignment (CPU baseline + CUDA run)"
PYTHONPATH=. conda run -n "${ENV_NAME}" python deploy/compare_bbox_precls_onnx.py \
  "${CONFIG}" "${CKPT}" "${ONNX}" \
  --cams 6 --height 224 --width 224 \
  --cfg-options model.img_backbone.img_size=224

echo "[3/4] Run full output validation (CUDA)"
PYTHONPATH=. conda run -n "${ENV_NAME}" python deploy/validate_vision_onnx.py \
  "${CONFIG}" "${CKPT}" "${ONNX}" \
  --device cuda --ort-provider CUDAExecutionProvider \
  --cams 6 --height 224 --width 224 \
  --cfg-options model.img_backbone.img_size=224

echo "[4/4] Focus on memory/TopK debug outputs"
PYTHONPATH=. conda run -n "${ENV_NAME}" python deploy/validate_vision_onnx.py \
  "${CONFIG}" "${CKPT}" "${ONNX}" \
  --device cuda --ort-provider CUDAExecutionProvider \
  --cams 6 --height 224 --width 224 \
  --cfg-options model.img_backbone.img_size=224 | \
  grep -E "bbox_memory_ref_|map_memory_ref_|bbox_rec_score|bbox_topk_indexes|bbox_rec_reference_points|map_rec_score|map_topk_indexes|map_rec_reference_points|Mismatched outputs|Passed|All outputs are within tolerance" || true

echo "Done. Use the full log to locate first divergence stage:"
echo "- pre_update diverges: before topk/cat"
echo "- post_cat diverges: topk indexes / gather / concat path"
echo "- post_transform diverges: transform_reference_points* path"
