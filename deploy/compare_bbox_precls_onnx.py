import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch
from mmcv import DictAction

from deploy.validate_vision_onnx import (
    INPUT_NAMES,
    build_numpy_inputs,
    build_proxy,
    prepare_cfg,
    resolve_torch_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ONNX bbox pre-cls transformer outputs against PyTorch bbox_pre_cls outputs"
    )
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("onnx_file")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--ort-provider", default="CUDAExecutionProvider")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cams", type=int, default=6)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold-max-abs", type=float, default=1e-3)
    parser.add_argument(
        "--threshold-max-abs-cpu",
        type=float,
        default=1e-3,
        help="acceptance threshold for CPU baseline run",
    )
    parser.add_argument(
        "--threshold-max-abs-cuda",
        type=float,
        default=5e-3,
        help="acceptance threshold for CUDA run",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config settings using key=value pairs",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _rel_l2(a, b):
    num = np.linalg.norm(a.reshape(-1) - b.reshape(-1))
    den = np.linalg.norm(a.reshape(-1)) + 1e-12
    return float(num / den)


def run_pytorch_precls(proxy, torch_inputs):
    # Use direct proxy output to avoid hook ambiguity when cls branches are shared.
    with torch.no_grad():
        outputs = proxy(*torch_inputs)
    return _to_numpy(outputs[3])


def run_onnx_precls(onnx_file, numpy_inputs, provider):
    providers = [provider]
    if provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(onnx_file, providers=providers)

    output_names = [o.name for o in session.get_outputs()]
    if "bbox_pre_cls" not in output_names:
        raise RuntimeError(
            "ONNX model does not contain output 'bbox_pre_cls'. Please re-export with updated proxy."
        )

    feed = {i.name: numpy_inputs[i.name] for i in session.get_inputs()}
    return session.run(["bbox_pre_cls"], feed)[0]


def detect_shared_cls_branches(proxy) -> List[List[int]]:
    branches = list(proxy.mod.pts_bbox_head.cls_branches)
    groups: Dict[int, List[int]] = {}
    for idx, module in enumerate(branches):
        groups.setdefault(id(module), []).append(idx)
    return [idxs for idxs in groups.values() if len(idxs) > 1]


def compare_and_print(pt_precls, onnx_precls, threshold, label):
    if pt_precls.shape != onnx_precls.shape:
        raise RuntimeError(f"Shape mismatch: pt={pt_precls.shape}, onnx={onnx_precls.shape}")

    print(f"\n[bbox_pre_cls alignment] {label}")
    print(f"shape={pt_precls.shape} threshold_max_abs={threshold}")

    first_bad = None
    for layer in range(pt_precls.shape[0]):
        a = pt_precls[layer]
        b = onnx_precls[layer]
        d = np.abs(a - b)
        max_abs = float(d.max())
        mean_abs = float(d.mean())
        rel_l2 = _rel_l2(a, b)
        max_idx = np.unravel_index(np.argmax(d), d.shape)
        print(
            f"layer={layer}: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rel_l2={rel_l2:.6e} max_idx={max_idx} "
            f"pt={float(a[max_idx]):.6e} onnx={float(b[max_idx]):.6e}"
        )
        if first_bad is None and max_abs > threshold:
            first_bad = layer

    if first_bad is None:
        print("first_diverged_layer: none (all layers within threshold)")
        return True

    print(f"first_diverged_layer: {first_bad}")
    return False


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = prepare_cfg(args)
    numpy_inputs = build_numpy_inputs(args)

    # Re-seed before each model build to keep missing checkpoint params consistent.
    set_seed(args.seed)
    cpu_device = torch.device("cpu")
    cpu_proxy = build_proxy(cfg, args.checkpoint, cpu_device)

    shared_groups = detect_shared_cls_branches(cpu_proxy)
    if shared_groups:
        print("[warning] Detected shared cls branches in bbox head:")
        for group in shared_groups:
            print(f"  shared branch indices: {group}")
        print("[warning] Hook-based per-layer capture on cls branches may be overwritten. Use bbox_pre_cls direct output.")

    cpu_torch_inputs = [torch.from_numpy(numpy_inputs[name]).to(cpu_device) for name in INPUT_NAMES]
    pt_precls_cpu = run_pytorch_precls(cpu_proxy, cpu_torch_inputs)
    onnx_precls_cpu = run_onnx_precls(args.onnx_file, numpy_inputs, "CPUExecutionProvider")
    cpu_ok = compare_and_print(
        pt_precls_cpu,
        onnx_precls_cpu,
        args.threshold_max_abs_cpu,
        "CPU baseline (PyTorch CPU vs ORT CPU)",
    )

    have_cuda = torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers()
    cuda_ok = None
    if have_cuda:
        set_seed(args.seed)
        cuda_device = torch.device("cuda")
        cuda_proxy = build_proxy(cfg, args.checkpoint, cuda_device)
        cuda_torch_inputs = [torch.from_numpy(numpy_inputs[name]).to(cuda_device) for name in INPUT_NAMES]
        pt_precls_cuda = run_pytorch_precls(cuda_proxy, cuda_torch_inputs)
        onnx_precls_cuda = run_onnx_precls(args.onnx_file, numpy_inputs, "CUDAExecutionProvider")
        cuda_ok = compare_and_print(
            pt_precls_cuda,
            onnx_precls_cuda,
            args.threshold_max_abs_cuda,
            "CUDA run (PyTorch CUDA vs ORT CUDA)",
        )
    else:
        print("\n[info] CUDA not available for both PyTorch and ORT; skipped CUDA run.")

    print("\n[drift verdict]")
    print(f"cpu_baseline_ok: {cpu_ok}")
    if cuda_ok is None:
        print("cuda_run_ok: skipped")
        print(f"overall_acceptable: {cpu_ok}")
    else:
        print(f"cuda_run_ok: {cuda_ok}")
        print(f"overall_acceptable: {cpu_ok and cuda_ok}")


if __name__ == "__main__":
    main()
