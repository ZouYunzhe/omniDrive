# OmniDrive ONNX Vision 输入 Shape 参考

本文档总结 `tools/test.py` 中 ONNX 导出示例输入（dummy input）的来源与确定方式，便于后续修改配置或适配部署时快速核对。

## 1. 输入 Tensor 是怎么确定的

ONNX 的输入集合由 `OmniDriveVisionTrtProxy.forward(...)` 的函数签名决定，`tools/test.py` 里的 `input_names` 只是按该顺序做名称映射。

- Proxy forward 定义：`deploy/export_vision.py`
- 测试导出入口：`tools/test.py`

## 2. Shape 是怎么确定的

输入 shape 由以下两类因素共同决定：

1. 模型/数据配置（例如 `memory_len`、`topk_proposals`、`embed_dims`、`n_control`、`img_scale`）
2. 时序 memory 的更新逻辑（先截断再拼接 top-k）

其中 `tools/test.py` 里全部使用 `np.ones(...)` 只是为了 ONNX trace；数值本身不重要，维度才重要。

## 3. 关键输入及来源对照

### 3.1 图像与几何输入

- `img`: `[B, Ncam, 3, H, W]`，示例为 `[1, 6, 3, 640, 640]`
  - `H/W=640` 来自多视角 resize 配置 `img_scale=(640, 640)`
  - `Ncam=6` 与当前多相机输入设置一致，导出 proxy 中也按 6 路构造
- `intrinsics`, `img2lidars`: `[B, Ncam, 4, 4]`

### 3.2 时序 memory 主干维度

- `memory_len=600`
- `topk_proposals=300`
- 因此导出侧 memory 输入长度使用 `900 = 600 + 300`

这解释了如下输入中第二维为 `900`：

- `memory_embedding_bbox_in`: `[B, 900, 256]`
- `memory_reference_point_bbox_in`: `[B, 900, 3]`
- `memory_timestamp_bbox_in`: `[B, 900, 1]`
- `memory_egopose_bbox_in`: `[B, 900, 4, 4]`
- `memory_timestamp_map_in`: `[B, 900, 1]`
- `memory_egopose_map_in`: `[B, 900, 4, 4]`
- `memory_embedding_map_in`: `[B, 900, 256]`
- `memory_reference_point_map_in`: `[B, 900, 11, 3]`

### 3.3 为什么是 256 和 11

- `256` 来自 transformer `embed_dims=256`
- `11` 来自 map 头的 `n_control=11`（每条 lane 11 个控制点，每点 3 维坐标）

### 3.4 `memory_canbus_bbox_in` 为什么是 `[B, 3, 14]`

- `14 = command(1) + can_bus(13)`
- 代码里会按 `head.can_bus_len` 对历史 can bus 先切片，再拼回当前帧，因此导出示例给到 3 这个长度用于与推理循环对齐（out->in）

## 4. 代码逻辑要点

时序 memory 更新的核心流程：

1. 先对历史 memory 按 `memory_len`（或 can bus 的 `can_bus_len`）进行 refresh/截断
2. 计算当前帧 top-k proposal
3. 将当前帧的 top-k 与历史 memory 进行拼接

因此导出时常采用 `memory_len + topk_proposals` 作为导出输入长度，便于在流式推理时稳定衔接。

## 5. 维护建议

当你修改以下配置项时，需要同步检查 ONNX 导出 dummy input 的 shape：

- `memory_len`
- `topk_proposals`
- `embed_dims`
- `n_control`
- 图像尺寸（如 `img_scale`）
- 相机数量（若数据管线/模型改成非 6 路）

建议优先保证以下三处一致：

1. `deploy/export_vision.py` 的 `forward` 签名
2. `tools/test.py` 中导出用 dummy input 的 shape
3. 实际 TensorRT/部署侧的循环绑定（out->in）逻辑

## 6. 快速结论

- 输入列表：由 proxy forward 签名固定。
- 输入维度：由配置项 + memory 更新规则共同决定。
- dummy 输入值：可为全 1，不影响导出结构；但 shape 必须与模型逻辑严格一致。
