# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import mmcv
import os
import torch
import warnings
import numpy as np
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.core.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config',help='test config file path')
    parser.add_argument('checkpoint', nargs='?', default=None, help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--export-onnx',
        action='store_true',
        help='export model to ONNX with random-initialized weights when checkpoint is not provided')
    parser.add_argument(
        '--onnx-file',
        default='onnxs/test_random.onnx',
        help='output ONNX path when --export-onnx is set')
    parser.add_argument(
        '--opset-version',
        type=int,
        default=17,
        help='ONNX opset version for export')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.export_onnx or args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results / export onnx) with the argument '
         '"--out", "--eval", "--format-only", "--show", "--show-dir" '
         'or "--export-onnx"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    if not args.export_onnx:
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    if args.export_onnx:
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

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    else:
        print('No checkpoint is provided. Using random initialized weights.')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if not args.export_onnx:
        if checkpoint is not None and 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        # palette for visualization in segmentation tasks
        if checkpoint is not None and 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        elif hasattr(dataset, 'PALETTE'):
            # segmentation dataset has `PALETTE` attribute
            model.PALETTE = dataset.PALETTE

    if not distributed:
        if args.export_onnx:
            from deploy.export_vision import OmniDriveVisionTrtProxy

            model = model.float().cpu()
            model.eval()
            model.training = False

            mmcv.mkdir_or_exist(osp.dirname(args.onnx_file) or '.')

            input_precision = np.float32
            onnx_device = 'cpu'

            img = np.ones([1, 6, 3, 640, 640], dtype=np.float32)
            intrinsics = np.ones([1, 6, 4, 4], dtype=np.float32)
            img2lidars = np.ones([1, 6, 4, 4], dtype=np.float32)
            command = np.ones([1], dtype=np.float32)
            can_bus = np.ones([1, 13], dtype=np.float32)
            is_first_frame = np.ones([1], dtype=np.float32)
            ego_pose = np.ones([1, 4, 4], dtype=np.float32)
            timestamp = np.ones([1], dtype=np.float32)
            ego_pose_inv = np.ones([1, 4, 4], dtype=np.float32)
            memory_embedding_bbox_in = np.ones([1, 900, 256], dtype=np.float32)
            memory_reference_point_bbox_in = np.ones([1, 900, 3], dtype=np.float32)
            memory_timestamp_bbox_in = np.ones([1, 900, 1], dtype=np.float32)
            memory_egopose_bbox_in = np.ones([1, 900, 4, 4], dtype=np.float32)
            memory_canbus_bbox_in = np.ones([1, 3, 14], dtype=np.float32)
            sample_time_bbox_in = np.ones([1], dtype=np.float32)
            memory_timestamp_map_in = np.ones([1, 900, 1], dtype=np.float32)
            sample_time_map_in = np.ones([1], dtype=np.float32)
            memory_egopose_map_in = np.ones([1, 900, 4, 4], dtype=np.float32)
            memory_embedding_map_in = np.ones([1, 900, 256], dtype=np.float32)
            memory_reference_point_map_in = np.ones([1, 900, 11, 3], dtype=np.float32)

            proxy = OmniDriveVisionTrtProxy(model)
            onnx_args = [
                torch.from_numpy(img.astype(input_precision)).to(onnx_device),
                torch.from_numpy(intrinsics.astype(input_precision)).to(onnx_device),
                torch.from_numpy(img2lidars.astype(input_precision)).to(onnx_device),
                torch.from_numpy(command.astype(input_precision)).to(onnx_device),
                torch.from_numpy(can_bus.astype(input_precision)).to(onnx_device),
                torch.from_numpy(is_first_frame.astype(input_precision)).to(onnx_device),
                torch.from_numpy(ego_pose.astype(input_precision)).to(onnx_device),
                torch.from_numpy(timestamp.astype(input_precision)).to(onnx_device),
                torch.from_numpy(ego_pose_inv.astype(input_precision)).to(onnx_device),
                torch.from_numpy(memory_embedding_bbox_in.astype(input_precision)).to(onnx_device),
                torch.from_numpy(memory_reference_point_bbox_in.astype(input_precision)).to(onnx_device),
                torch.from_numpy(memory_timestamp_bbox_in.astype(input_precision)).to(onnx_device),
                torch.from_numpy(memory_egopose_bbox_in.astype(input_precision)).to(onnx_device),
                torch.from_numpy(memory_canbus_bbox_in.astype(input_precision)).to(onnx_device),
                torch.from_numpy(sample_time_bbox_in.astype(input_precision)).to(onnx_device),
                torch.from_numpy(memory_timestamp_map_in.astype(input_precision)).to(onnx_device),
                torch.from_numpy(sample_time_map_in.astype(input_precision)).to(onnx_device),
                torch.from_numpy(memory_egopose_map_in.astype(input_precision)).to(onnx_device),
                torch.from_numpy(memory_embedding_map_in.astype(input_precision)).to(onnx_device),
                torch.from_numpy(memory_reference_point_map_in.astype(input_precision)).to(onnx_device),
            ]

            input_names = [
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

            output_names = [
                'vision_embeded',
                'all_cls_scores',
                'all_bbox_preds',
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
            ]

            torch.onnx.export(
                proxy,
                tuple(onnx_args),
                args.onnx_file,
                input_names=input_names,
                output_names=output_names,
                do_constant_folding=True,
                opset_version=args.opset_version,
                verbose=False)
            print(f'ONNX model is exported to {args.onnx_file}')
            return

        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
