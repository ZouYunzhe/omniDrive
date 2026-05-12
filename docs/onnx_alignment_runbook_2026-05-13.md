# ONNX Alignment Runbook (2026-05-13)

This document records the stabilized debug version and the exact steps to reproduce today's conclusions.

## Stabilized Version Scope

Code paths updated for this version:
- `deploy/export_vision.py`
- `tools/test.py`
- `deploy/validate_vision_onnx.py`
- `deploy/compare_bbox_precls_onnx.py`

Debug outputs included in ONNX:
- Intermediate classifier input:
  - `bbox_pre_cls`
- Memory-reference tracing (bbox/map):
  - `bbox_memory_ref_pre_update`
  - `bbox_memory_ref_post_cat`
  - `bbox_memory_ref_post_transform`
  - `map_memory_ref_pre_update`
  - `map_memory_ref_post_cat`
  - `map_memory_ref_post_transform`
- TopK tracing (bbox/map):
  - `bbox_rec_score`
  - `bbox_topk_indexes`
  - `bbox_rec_reference_points`
  - `map_rec_score`
  - `map_topk_indexes`
  - `map_rec_reference_points`

## One-Command Reproduction

```bash
cd /omniDrive
bash deploy/repro_onnx_alignment_debug.sh
```

## Manual Reproduction Steps

### 1) Export ONNX with debug outputs

```bash
cd /omniDrive
PYTHONPATH=. conda run -n omnidrive_onnx python tools/test.py \
  projects/configs/OmniDrive/mask_eva_lane_det_vlm.py /omniDrive/eva02_petr_proj.pth \
  --export-onnx --onnx-file onnxs/mask_eva_lane_det_vlm_eva02_gpu_224.onnx \
  --launcher none --export-cams 6 --export-height 224 --export-width 224 \
  --cfg-options model.img_backbone.img_size=224
```

### 2) Check bbox_pre_cls alignment (critical sanity check)

```bash
cd /omniDrive
PYTHONPATH=. conda run -n omnidrive_onnx python deploy/compare_bbox_precls_onnx.py \
  projects/configs/OmniDrive/mask_eva_lane_det_vlm.py /omniDrive/eva02_petr_proj.pth \
  onnxs/mask_eva_lane_det_vlm_eva02_gpu_224.onnx \
  --cams 6 --height 224 --width 224 \
  --cfg-options model.img_backbone.img_size=224
```

Expected key lines:
- Warning about shared cls branches may appear (this is expected and informative).
- Drift verdict lines should be printed:
  - `cpu_baseline_ok: ...`
  - `cuda_run_ok: ...`
  - `overall_acceptable: ...`

### 3) Run full output validation

```bash
cd /omniDrive
PYTHONPATH=. conda run -n omnidrive_onnx python deploy/validate_vision_onnx.py \
  projects/configs/OmniDrive/mask_eva_lane_det_vlm.py /omniDrive/eva02_petr_proj.pth \
  onnxs/mask_eva_lane_det_vlm_eva02_gpu_224.onnx \
  --device cuda --ort-provider CUDAExecutionProvider \
  --cams 6 --height 224 --width 224 \
  --cfg-options model.img_backbone.img_size=224
```

## Today's Conclusions

1. Previous large `layer0` mismatch was partly from hook-capture ambiguity when cls branches are shared.
2. Direct `bbox_pre_cls` comparison is the reliable method.
3. CPU baseline can be very tight (near machine precision).
4. Full end-to-end outputs are still not all aligned under CUDA settings; further tracing should use the debug outputs above.
5. To localize divergence stage quickly:
   - If `*_pre_update` diverges first: issue is before topk/cat.
   - If `*_post_cat` diverges first: likely topk indexes / gather / concat.
   - If `*_post_transform` diverges first: likely `transform_reference_points*` path.

## Known Pitfalls

- Disk usage: ONNX file is large (~1.4G at this setup). Keep free space before export.
- Seed consistency: build PyTorch models with fixed seeds before each run to avoid false drift from randomly initialized missing checkpoint keys.
- Hook caveat: per-layer cls hook may be overwritten for shared branch modules; prefer direct output tensors for alignment.

## Recommended Next Start Point

Run Step 2 first. If drift verdict passes but full validation still fails, use Step 3 output and focus only on new memory/TopK debug tensors to find first failing stage.
