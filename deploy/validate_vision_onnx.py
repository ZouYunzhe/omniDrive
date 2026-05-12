import argparse
import importlib
import os
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d.models import build_model

from deploy.export_vision import OmniDriveVisionTrtProxy

INPUT_NAMES = [
    'img',
    'intrinsics',
    'img2lidars',
    'command',
    'can_bus',
    'is_first_frame',
    'ego_pose',
    'timestamp',
    'ego_pose_inv',
    'memory_embedding_bbox_in',
    'memory_reference_point_bbox_in',
    'memory_timestamp_bbox_in',
    'memory_egopose_bbox_in',
    'memory_canbus_bbox_in',
    'sample_time_bbox_in',
    'memory_timestamp_map_in',
    'sample_time_map_in',
    'memory_egopose_map_in',
    'memory_embedding_map_in',
    'memory_reference_point_map_in',
]

OUTPUT_NAMES = [
    'vision_embeded',
    'all_cls_scores',
    'all_bbox_preds',
    'bbox_pre_cls',
    'all_lane_cls_one2one',
    'all_lane_preds_one2one',
    'all_lane_cls_one2many',
    'all_lane_preds_one2many',
    'outs_dec_one2one',
    'outs_dec_one2many',
    'memory_embedding_bbox_out',
    'memory_reference_point_bbox_out',
    'memory_timestamp_bbox_out',
    'memory_egopose_bbox_out',
    'memory_canbus_bbox_out',
    'sample_time_bbox_out',
    'memory_embedding_map_out',
    'memory_timestamp_map_out',
    'memory_egopose_map_out',
    'memory_reference_point_map_out',
    'sample_time_map_out',
    'bbox_memory_ref_pre_update',
    'bbox_memory_ref_post_cat',
    'bbox_memory_ref_post_transform',
    'map_memory_ref_pre_update',
    'map_memory_ref_post_cat',
    'map_memory_ref_post_transform',
    'bbox_rec_score',
    'bbox_topk_indexes',
    'bbox_rec_reference_points',
    'map_rec_score',
    'map_topk_indexes',
    'map_rec_reference_points',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare OmniDrive PyTorch proxy outputs against ONNX outputs.'
    )
    parser.add_argument('config', help='model config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('onnx_file', help='exported ONNX file path')
    parser.add_argument('--seed', type=int, default=0, help='random seed for input generation')
    parser.add_argument('--cams', type=int, default=6, help='number of camera views')
    parser.add_argument('--height', type=int, default=224, help='dummy image height')
    parser.add_argument('--width', type=int, default=224, help='dummy image width')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for validation')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='PyTorch execution device')
    parser.add_argument(
        '--ort-provider',
        choices=['auto', 'CPUExecutionProvider', 'CUDAExecutionProvider'],
        default='auto',
        help='onnxruntime execution provider',
    )
    parser.add_argument('--atol', type=float, default=1e-4, help='absolute tolerance for np.allclose')
    parser.add_argument('--rtol', type=float, default=1e-3, help='relative tolerance for np.allclose')
    parser.add_argument(
        '--fail-on-mismatch',
        action='store_true',
        help='exit with code 1 when any output exceeds tolerance',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config settings using key=value pairs',
    )
    return parser.parse_args()


def import_plugin_modules(cfg: Config, config_path: str) -> None:
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg['custom_imports'])

    if not getattr(cfg, 'plugin', False):
        return

    if hasattr(cfg, 'plugin_dir'):
        module_dir = os.path.dirname(cfg.plugin_dir)
    else:
        module_dir = os.path.dirname(config_path)
    module_path = '.'.join(module_dir.split('/'))
    importlib.import_module(module_path)


def prepare_cfg(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    import_plugin_modules(cfg, args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.export_onnx = True
    cfg.model.lm_head = None
    cfg.model.tokenizer = None

    if 'img_backbone' in cfg.model:
        if 'flash_attn' in cfg.model.img_backbone:
            cfg.model.img_backbone['flash_attn'] = False
        if 'with_cp' in cfg.model.img_backbone:
            cfg.model.img_backbone['with_cp'] = False
    if 'pts_bbox_head' in cfg.model and 'transformer' in cfg.model.pts_bbox_head:
        if 'flash_attn' in cfg.model.pts_bbox_head.transformer:
            cfg.model.pts_bbox_head.transformer['flash_attn'] = False
        if 'with_cp' in cfg.model.pts_bbox_head.transformer:
            cfg.model.pts_bbox_head.transformer['with_cp'] = False
    if 'map_head' in cfg.model and 'transformer' in cfg.model.map_head:
        if 'flash_attn' in cfg.model.map_head.transformer:
            cfg.model.map_head.transformer['flash_attn'] = False
        if 'with_cp' in cfg.model.map_head.transformer:
            cfg.model.map_head.transformer['with_cp'] = False

    return cfg


def build_proxy(cfg: Config, checkpoint_path: str, device: torch.device) -> OmniDriveVisionTrtProxy:
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = model.float().to(device)
    model.eval()
    model.training = False
    return OmniDriveVisionTrtProxy(model).to(device)


def resolve_torch_device(requested: str) -> torch.device:
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')
    return torch.device(requested)


def resolve_ort_providers(requested: str) -> List[str]:
    available = ort.get_available_providers()
    if requested == 'auto':
        if 'CUDAExecutionProvider' in available:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']
    if requested not in available:
        raise RuntimeError(f'ONNXRuntime provider {requested} is not available: {available}')
    if requested == 'CUDAExecutionProvider' and 'CPUExecutionProvider' in available:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return [requested]


def build_numpy_inputs(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    b = args.batch_size
    cams = args.cams
    h = args.height
    w = args.width

    def normal(shape: Tuple[int, ...], scale: float = 1.0) -> np.ndarray:
        return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(np.float32)

    def positive(shape: Tuple[int, ...], low: float = 0.1, high: float = 10.0) -> np.ndarray:
        return rng.uniform(low, high, size=shape).astype(np.float32)

    intrinsics = np.tile(np.eye(4, dtype=np.float32), (b, cams, 1, 1))
    intrinsics[..., 0, 0] = positive((b, cams), 500.0, 1500.0)
    intrinsics[..., 1, 1] = positive((b, cams), 500.0, 1500.0)
    intrinsics[..., 0, 2] = positive((b, cams), 100.0, max(101.0, float(w)))
    intrinsics[..., 1, 2] = positive((b, cams), 100.0, max(101.0, float(h)))

    img2lidars = np.tile(np.eye(4, dtype=np.float32), (b, cams, 1, 1))
    img2lidars[..., :3, 3] = normal((b, cams, 3), scale=0.5)

    ego_pose = np.tile(np.eye(4, dtype=np.float32), (b, 1, 1))
    ego_pose[:, :3, 3] = normal((b, 3), scale=0.5)
    ego_pose_inv = np.linalg.inv(ego_pose).astype(np.float32)

    memory_egopose_bbox = np.tile(np.eye(4, dtype=np.float32), (b, 900, 1, 1))
    memory_egopose_bbox[..., :3, 3] = normal((b, 900, 3), scale=0.2)
    memory_egopose_map = np.tile(np.eye(4, dtype=np.float32), (b, 900, 1, 1))
    memory_egopose_map[..., :3, 3] = normal((b, 900, 3), scale=0.2)

    return {
        'img': normal((b, cams, 3, h, w), scale=1.0),
        'intrinsics': intrinsics,
        'img2lidars': img2lidars,
        'command': rng.integers(low=0, high=4, size=(b,), dtype=np.int32).astype(np.float32),
        'can_bus': normal((b, 13), scale=0.1),
        'is_first_frame': np.zeros((b,), dtype=np.float32),
        'ego_pose': ego_pose,
        'timestamp': positive((b,), 0.0, 1.0),
        'ego_pose_inv': ego_pose_inv,
        'memory_embedding_bbox_in': normal((b, 900, 256), scale=0.02),
        'memory_reference_point_bbox_in': normal((b, 900, 3), scale=1.0),
        'memory_timestamp_bbox_in': normal((b, 900, 1), scale=0.1),
        'memory_egopose_bbox_in': memory_egopose_bbox,
        'memory_canbus_bbox_in': normal((b, 3, 14), scale=0.1),
        'sample_time_bbox_in': positive((b,), 0.0, 1.0),
        'memory_timestamp_map_in': normal((b, 900, 1), scale=0.1),
        'sample_time_map_in': positive((b,), 0.0, 1.0),
        'memory_egopose_map_in': memory_egopose_map,
        'memory_embedding_map_in': normal((b, 900, 256), scale=0.02),
        'memory_reference_point_map_in': normal((b, 900, 11, 3), scale=1.0),
    }


def run_pytorch(
    proxy: OmniDriveVisionTrtProxy,
    inputs: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    torch_inputs = [torch.from_numpy(inputs[name]).to(device) for name in INPUT_NAMES]
    with torch.no_grad():
        outputs = proxy(*torch_inputs)
    return {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(OUTPUT_NAMES, outputs)
    }


def run_onnx(onnx_file: str, inputs: Dict[str, np.ndarray], providers: List[str]) -> Dict[str, np.ndarray]:
    session = ort.InferenceSession(onnx_file, providers=providers)
    session_input_names = [item.name for item in session.get_inputs()]
    session_output_names = [item.name for item in session.get_outputs()]
    missing = [name for name in session_input_names if name not in inputs]
    if missing:
        raise KeyError(f'Missing ONNX inputs in generated feed: {missing}')
    feed = {name: inputs[name] for name in session_input_names}
    outputs = session.run(session_output_names, feed)
    return {name: array for name, array in zip(session_output_names, outputs)}


def summarize_diff(name: str, ref: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    diff = np.abs(ref - pred)
    denom = np.maximum(np.abs(ref), 1e-6)
    rel = diff / denom
    return {
        'max_abs': float(diff.max(initial=0.0)),
        'mean_abs': float(diff.mean()),
        'max_rel': float(rel.max(initial=0.0)),
        'mean_rel': float(rel.mean()),
    }


def _format_sample(array: np.ndarray, limit: int = 8) -> str:
    flat = array.reshape(-1)
    sample = flat[:limit]
    return np.array2string(sample, precision=6, separator=', ')


def build_failure_detail(
    name: str,
    ref: np.ndarray,
    pred: np.ndarray,
    metrics: Dict[str, float],
) -> Dict[str, object]:
    diff = np.abs(ref - pred)
    flat_index = int(diff.argmax()) if diff.size else 0
    max_index = tuple(int(item) for item in np.unravel_index(flat_index, diff.shape)) if diff.size else ()
    ref_value = float(ref[max_index]) if diff.size else 0.0
    pred_value = float(pred[max_index]) if diff.size else 0.0
    abs_diff = float(diff[max_index]) if diff.size else 0.0
    denom = max(abs(ref_value), 1e-6)
    rel_diff = float(abs_diff / denom)
    return {
        'name': name,
        'shape': tuple(int(dim) for dim in ref.shape),
        'torch_sample': _format_sample(ref),
        'onnx_sample': _format_sample(pred),
        'max_index': max_index,
        'torch_value': ref_value,
        'onnx_value': pred_value,
        'abs_diff': abs_diff,
        'rel_diff': rel_diff,
        'metrics': metrics,
    }


def compare_outputs(
    torch_outputs: Dict[str, np.ndarray],
    onnx_outputs: Dict[str, np.ndarray],
    atol: float,
    rtol: float,
) -> List[Tuple[str, bool, Dict[str, float]]]:
    results = []
    for name in OUTPUT_NAMES:
        if name not in torch_outputs or name not in onnx_outputs:
            continue
        torch_out = torch_outputs[name]
        onnx_out = onnx_outputs[name]
        if torch_out.shape != onnx_out.shape:
            raise ValueError(
                f'Output {name} shape mismatch: PyTorch {torch_out.shape} vs ONNX {onnx_out.shape}'
            )
        metrics = summarize_diff(name, torch_out, onnx_out)
        passed = np.allclose(torch_out, onnx_out, atol=atol, rtol=rtol, equal_nan=True)
        results.append((name, passed, metrics))
    return results


def print_summary(results: List[Tuple[str, bool, Dict[str, float]]], atol: float, rtol: float) -> None:
    print(f'Comparison tolerance: atol={atol} rtol={rtol}')
    print('-' * 98)
    print(
        f"{'output':32} {'status':8} {'max_abs':>12} {'mean_abs':>12} {'max_rel':>12} {'mean_rel':>12}"
    )
    print('-' * 98)
    for name, passed, metrics in results:
        status = 'PASS' if passed else 'FAIL'
        print(
            f"{name:32} {status:8} {metrics['max_abs']:12.6e} {metrics['mean_abs']:12.6e} "
            f"{metrics['max_rel']:12.6e} {metrics['mean_rel']:12.6e}"
        )
    print('-' * 98)
    passed_count = sum(1 for _, passed, _ in results if passed)
    print(f'Passed {passed_count}/{len(results)} outputs')


def print_failure_details(failure_details: List[Dict[str, object]]) -> None:
    if not failure_details:
        return
    print('\nDetailed mismatch report')
    print('=' * 98)
    for item in failure_details:
        print(f"output: {item['name']}")
        print(f"shape: {item['shape']}")
        print(f"torch_sample: {item['torch_sample']}")
        print(f"onnx_sample:  {item['onnx_sample']}")
        print(f"max_abs_index: {item['max_index']}")
        print(
            'max_abs_values: '
            f"torch={item['torch_value']:.6e} onnx={item['onnx_value']:.6e} "
            f"abs_diff={item['abs_diff']:.6e} rel_diff={item['rel_diff']:.6e}"
        )
        metrics = item['metrics']
        print(
            'summary: '
            f"max_abs={metrics['max_abs']:.6e} mean_abs={metrics['mean_abs']:.6e} "
            f"max_rel={metrics['max_rel']:.6e} mean_rel={metrics['mean_rel']:.6e}"
        )
        print('-' * 98)


def main() -> int:
    args = parse_args()

    onnx.checker.check_model(onnx.load(args.onnx_file))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_torch_device(args.device)
    providers = resolve_ort_providers(args.ort_provider)
    cfg = prepare_cfg(args)
    proxy = build_proxy(cfg, args.checkpoint, device)

    inputs = build_numpy_inputs(args)
    torch_outputs = run_pytorch(proxy, inputs, device)
    onnx_outputs = run_onnx(args.onnx_file, inputs, providers)

    results = compare_outputs(torch_outputs, onnx_outputs, args.atol, args.rtol)
    failure_details = []
    for name, passed, metrics in results:
        if not passed:
            failure_details.append(
                build_failure_detail(name, torch_outputs[name], onnx_outputs[name], metrics)
            )
    print(f'PyTorch device: {device}')
    print(f'ONNXRuntime providers: {providers}')
    print(f'ONNX file: {args.onnx_file}')
    print_summary(results, args.atol, args.rtol)
    print_failure_details(failure_details)

    failed = [name for name, passed, _ in results if not passed]
    if failed:
        print('Mismatched outputs:', ', '.join(failed))
        return 1 if args.fail_on_mismatch else 0
    print('All outputs are within tolerance.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
