# 训练与评估动作专家

**中文** | [English](train_action_expert_en.md)

本文介绍如何用本地数据评估一个已发布的 UnifoLM-WLA 动作专家 checkpoint，
以及如何基于 UnifoLM-ER 系列基础视觉语言模型从头训练一个动作专家。
以下命令均假设当前工作目录为项目根目录。

## 安装

本项目使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理。

1. 安装 `uv`：

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. 安装本项目（在项目根目录下执行）：

   ```bash
   uv sync
   ```

3. 安装 `flash-attn`：

   ```bash
   uv pip install flash-attn --no-build-isolation
   ```

   未安装 `flash-attn` 也不影响运行：VLM 主干
   （`unifolm_wla/model/modules/vlm/QWen3.py`）和动作专家（DiT）主干
   （`unifolm_wla/model/modules/action_model/DiT_modules/mmdit.py`）在检测不到
   `flash_attn` 时都会自动回退到 `sdpa`（diffusers 的 `NATIVE` attention
   backend），只是速度会慢一些。

4. `uv sync` 会创建独立的 `.venv`，并不会修改系统 Python 环境。激活一次即可：

   ```bash
   source .venv/bin/activate
   ```

   之后直接使用 `python` / `hf` / `accelerate` 等命令即可，本文档后续命令均假设
   已激活该虚拟环境。

## 评估 Checkpoint

[`examples/unifolm_wla/eval_files/unitree/eval_local_episode.py`](../examples/unifolm_wla/eval_files/unitree/eval_local_episode.py)
会在本地 Unitree 数据集的一整段 episode 上，按 chunk 逐段运行训练好（或已发布）的
checkpoint 推理，并将每个动作维度的预测值与真值绘制成图，同时给出末端执行器绝对位姿
表示和模型原生的相对动作表示两种视图。

### 1. 下载模型

从 [UnifoLM-WLA-1.0 模型合集](https://huggingface.co/collections/unitreerobotics/unifolm-wla-10)
下载已发布的 checkpoint 到本机目录，例如使用 `hf`：

```bash
hf download unitreerobotics/UnifoLM-WLA-1.0-Base \
    --local-dir playground/Pretrained_models/UnifoLM-WLA-1.0-Base
```

下载后的目录结构应如下所示：

```text
UnifoLM-WLA-1.0-Base/
├── checkpoints/
│   └── model.safetensors
├── config.yaml
├── dataset_statistics.json
└── tokenizer/
```

### 2. 下载并配置评估数据

从 [UnifoLM-WLA-1.0 数据集合集](https://huggingface.co/collections/unitreerobotics/unifolm-wla-10)
下载评估数据。每个任务都是一个独立的数据集仓库，分别下载到对应 Dex1/WBT 类型目录下的
子目录中，例如使用 `hf`：

```bash
export DATA_ROOT=/path/to/unifolm_data

hf download unitreerobotics/G1_Dex1_MountCamera_Dataset --repo-type dataset \
    --local-dir $DATA_ROOT/UnifoLM_G1_Dex1_Dataset/G1_Dex1_MountCamera_Dataset

hf download unitreerobotics/G1_WBT_Brainco_Supermarket_Shelf_Organizing --repo-type dataset \
    --local-dir $DATA_ROOT/UnifoLM_WBT_Dataset/G1_WBT_Brainco_Supermarket_Shelf_Organizing
```

数据需要按照 Dex1 和 WBT 两种类型分别组织；同一种类型的目录下可以包含多个任务。
推荐的目录结构如下：

```text
$DATA_ROOT/
├── UnifoLM_G1_Dex1_Dataset/
│   ├── G1_Dex1_MountCamera_Dataset/
│   └── G1_Dex1_Stack_Block/
└── UnifoLM_WBT_Dataset/
    ├── G1_WBT_Brainco_Supermarket_Shelf_Organizing/
    └── G1_WBT_Brainco_Make_The_Bed/
```

随后编辑
[`unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml`](../unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml)：

1. 将 `data_base` 设置为 `$DATA_ROOT`——即同时包含 Dex1 和 WBT 目录的公共父目录。
2. Dex1 数据继承 `*unitree_base`，并将 `data_path` 设置为 `UnifoLM_G1_Dex1_Dataset`。
3. WBT 数据继承 `*unitree_fullbody_base`，并将 `data_path` 设置为 `UnifoLM_WBT_Dataset`。
4. 将 `cache_dir` 设置为具有足够空间的本地缓存目录。

配置示例：

```yaml
data_base: "/path/to/unifolm_data"  # $DATA_ROOT
cache_dir: "/path/to/unifolm_cache"

datasets:
  - <<: *unitree_base
    name: "unifolm_g1_dex1"
    data_path: "UnifoLM_G1_Dex1_Dataset"
    image_keys: *unitree_img_with_stereo

  - <<: *unitree_fullbody_base
    name: "unifolm_wbt"
    data_path: "UnifoLM_WBT_Dataset"
    image_keys: *unitree_img_wo_stereo
```

### 3. 运行评估

```bash
python -m examples.unifolm_wla.eval_files.unitree.eval_local_episode \
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
    --data_config_path unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml \
    --episode_idx 0 \
    --save_dir results/eval_local_episode
```

请在项目根目录下运行该命令，以便 `unifolm_wla` 能作为包被正确导入。`--ckpt_path`
可以是 `.safetensors` 或 `.pt` checkpoint 文件；脚本假设标准的运行目录结构——
`config.yaml` 与 `dataset_statistics.json` 与 `checkpoints/` 目录同级
（即 `<run_dir>/checkpoints/<name>.safetensors`、`<run_dir>/config.yaml`、
`<run_dir>/dataset_statistics.json`）。若缺少 `dataset_statistics.json`，
脚本会根据 `--data_config_path` 自动重新生成。

## 模型服务

[`model_server/action_server_wbc_msgpack_unitree.py`](../model_server/action_server_wbc_msgpack_unitree.py)
会将训练好的 checkpoint 运行在一个 websocket/msgpack 动作服务后面，供 Unitree G1
全身控制（WBC）客户端调用。
[`model_server/eval_local_episode_wbc_msgpack_server_only.py`](../model_server/eval_local_episode_wbc_msgpack_server_only.py)
会驱动一个已经在运行的服务，在本地一整段 episode 上进行推理，并与真值对比。

> **仅支持 Dex1。**
> [`model_server/unifolm_wla_action_adapter.py`](../model_server/unifolm_wla_action_adapter.py)
> 中的 `_ACTIVE_SLOT_SPECS` 对应的是 `unitree.yaml` 里 `unitree_base`（Dex1）的
> `action_keys`，没有 `left_fig6d` / `right_fig6d`（五指灵巧手关节角度）和
> `base_pose`（全身移动相对位姿）这几项——这些字段只存在于 `unitree_fullbody_base`
> （WBT）中。因此由它构造出的 action mask，以及
> `action_server_wbc_msgpack_unitree.py` 里硬编码的客户端 obs/action 协议
> （`_DATA_KEYS`、`_build_state_unnorm`、`_encode_action`），目前都只覆盖二指
> 夹爪的 Dex1 动作空间。若要服务 `UnifoLM_WBT` checkpoint，需要：（1）在
> `_ACTIVE_SLOT_SPECS` 中补充 `left_fig6d`/`right_fig6d`/`base`（对应
> `base_rotvec`）这几项；（2）扩展服务端的 obs/action 协议（新增手指角度、
> 全身位姿对应的 obs key，以及 `_encode_action` 里相应的 `action.*` 返回
> 字段）——当前协议尚未支持。

### 1. 启动服务

```bash
python -m model_server.action_server_wbc_msgpack_unitree \
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
    --host 0.0.0.0 --port 8600 --instruction "pick up the object"
```

### 2. 评估正在运行的服务

在服务保持运行的情况下，另开一个终端执行：

```bash
python -m model_server.eval_local_episode_wbc_msgpack_server_only \
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
    --data_config_path unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml \
    --host 127.0.0.1 --port 8600 --episode_idx 0 \
    --save_dir results/eval_local_episode_wbc_msgpack
```

## 微调 UnifoLM-WLA-1.0-Base 动作专家

[`examples/unifolm_wla/train_files/run_finetune_mmdit_frozen_vlm.sh`](../examples/unifolm_wla/train_files/run_finetune_mmdit_frozen_vlm.sh)
会在新数据上微调一个已发布的 `UnifoLM-WLA-1.0-Base` checkpoint（例如
[`unitreerobotics/UnifoLM-WLA-1.0-Base`](https://huggingface.co/unitreerobotics/UnifoLM-WLA-1.0-Base)）。
VLM 主干被冻结（`trainer.freeze_modules: qwen_vl_interface`），只训练动作专家
（DiT）头和 robot-state projector。默认配置面向单卡 24GB 显存 GPU；若显存更小，
可在 [`mmdit_finetune_frozen_vlm.yaml`](../unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml)
中调小 batch size。

### 1. 下载基础 Checkpoint

下载要微调的已发布 checkpoint，方式与上文 [下载模型](#1-下载模型) 相同：

```bash
hf download unitreerobotics/UnifoLM-WLA-1.0-Base \
    --local-dir playground/Pretrained_models/UnifoLM-WLA-1.0-Base
```

如果已经下载到该目录，跳过此步骤即可。

### 2. 配置数据

按照上文 [下载并配置评估数据](#2-下载并配置评估数据) 一节下载并配置数据。

### 3. 启动训练

```bash
base_model_dir=playground/Pretrained_models/UnifoLM-WLA-1.0-Base \
bash examples/unifolm_wla/train_files/run_finetune_mmdit_frozen_vlm.sh
```

`base_model_dir` 是第一步下载的目录。脚本会将 `framework.qwenvl.base_vlm`
指向 `${base_model_dir}/tokenizer`（只提供架构配置与 tokenizer——基础
checkpoint 自身的 VLM 权重通过 `trainer.pretrained_checkpoint` 加载，因此
不需要再单独下载 VLM），并将 `trainer.pretrained_checkpoint` 指向
`${base_model_dir}/checkpoints/model.safetensors`。

可编辑 [`unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml`](../unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml)
调整数据配置路径、学习率、batch size 和训练步数。若想改为联合训练 VLM
而非冻结它（需要更多显存），清空 `trainer.freeze_modules` 即可。

微调输出默认写入 `run_root_dir/run_id`（默认为
`playground/Checkpoints/finetune_wla_base_frozen_vlm`），目录结构与上文
[评估 Checkpoint](#评估-checkpoint) 和 [模型服务](#模型服务) 使用的运行目录结构相同。

## LoRA 微调

除了全参数微调外，也可以通过 `trainer.lora` 配置块对 `qwen_vl_interface`
（VLM）和/或 `action_model`（DiT 动作头）两个主干分别使用
[LoRA](https://arxiv.org/abs/2106.09685) 进行适配。这只训练一小部分低秩
adapter 权重，该主干的其余部分保持冻结——适用于联合训练 VLM（出于显存考虑
通常会冻结）的场景，或者动作头全量微调的容量超出目标数据实际需要的场景。

```yaml
trainer:
  lora:
    enabled: true
    qwen_vl_interface:
      enabled: true
      r: 16
      lora_alpha: 32
      lora_dropout: 0.05
      target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
      bias: "none"
    action_model:
      enabled: true
      r: 16
      lora_alpha: 32
      lora_dropout: 0.05
      target_modules: ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
      bias: "none"
```

两个主干的子配置相互独立——可以只启用一个，也可以两个都启用。启用 LoRA
的模块**不需要**再出现在 `trainer.freeze_modules` 中：注入 LoRA 时已经会
冻结该主干的基础权重，只留下 adapter（`lora_A`/`lora_B`）参数可训练。

[`unifolm_wla/config/training/mmdit_lora_frozen_vlm.yaml`](../unifolm_wla/config/training/mmdit_lora_frozen_vlm.yaml)
是一份开箱即用的示例配置，基于 `mmdit_finetune_frozen_vlm.yaml` 构建：VLM
保持冻结，动作头改为 LoRA 适配而非全量微调。
[`examples/unifolm_wla/train_files/run_lora_finetune_mmdit_frozen_vlm.sh`](../examples/unifolm_wla/train_files/run_lora_finetune_mmdit_frozen_vlm.sh)
的启动方式与上文
[微调 UnifoLM-WLA-1.0-Base 动作专家](#微调-unifolm-wla-10-base-动作专家)
相同：

```bash
base_model_dir=playground/Pretrained_models/UnifoLM-WLA-1.0-Base \
bash examples/unifolm_wla/train_files/run_lora_finetune_mmdit_frozen_vlm.sh
```

每次 `save_interval`（以及训练结束时），除了常规的完整模型 checkpoint 外，
还会在同一目录下额外写入一个体积小很多的 adapter-only checkpoint
（`steps_<n>_adapter.safetensors` / `adapter.safetensors`，只包含
`lora_*` 张量），通常比完整 checkpoint 小几个数量级，是用于分发推理的产物。
断点续训（`trainer.is_resume: true`）始终从常规的完整 checkpoint 恢复，
其中已经同时包含冻结的基础权重和 LoRA 权重。

在启动正式训练之前，可以用下面这个独立的 smoke test 来验证 LoRA 注入、
冻结逻辑和分模块学习率是否配置正确：

```bash
python unifolm_wla/scripts/smoke_test_lora_injection.py \
    --config_yaml unifolm_wla/config/training/mmdit_debug_lora_frozen_vlm.yaml
```

## 从零训练

### 1. 下载基础视觉语言模型

选择并下载以下任一基础模型，例如使用 `hf`：

```bash
hf download unitreerobotics/UnifoLM-ER-1 \
    --local-dir playground/Pretrained_models/UnifoLM-ER-1

# 或者
hf download unitreerobotics/UnifoLM-ER-Flow \
    --local-dir playground/Pretrained_models/UnifoLM-ER-Flow
```

下载完成后，打开
[`examples/unifolm_wla/train_files/run_multi_source_train_mmdit_from_scratch.sh`](../examples/unifolm_wla/train_files/run_multi_source_train_mmdit_from_scratch.sh)，
将 `base_vlm` 设置为模型在本机的完整路径。例如：

```bash
base_vlm=/path/to/UnifoLM-ER-1
```

也可以将其改为 UnifoLM-ER-Flow 的本地目录：

```bash
base_vlm=/path/to/UnifoLM-ER-Flow
```

### 2. 下载并配置训练数据

从 [UnifoLM-WLA-1.0 数据集合集](https://huggingface.co/collections/unitreerobotics/unifolm-wla-10)
下载训练数据。数据组织方式与目录结构、`unitree.yaml` 的配置步骤与上文评估部分相同。

`multi_task: true` 时，加载器会把 `data_path` 下的各任务目录作为子数据集加载。
所有设置为 `enabled: true` 的数据源都会通过 `ConcatDataset` 拼接，因此不会遗漏后面的数据源；
采样比例由各数据源的数据量决定，当前不会应用配置中的 `weight`。

首次加载 LeRobot 数据时，加载器会在 `cache_dir/arrow_cache` 下生成 Arrow 缓存。
后续加载会复用该缓存，因此建议将 `cache_dir` 放在容量充足、读写速度较快的磁盘上。

### 3. 单节点训练

确认脚本中的 `base_vlm`、`run_root_dir` 和数据配置路径正确后，在项目根目录执行：

```bash
bash examples/unifolm_wla/train_files/run_multi_source_train_mmdit_from_scratch.sh
```

脚本默认使用当前节点上 `nvidia-smi -L` 检测到的全部 GPU。若只希望启动指定数量的进程，
可以通过 `NUM_PROCESSES` 覆盖，例如：

```bash
NUM_PROCESSES=4 bash examples/unifolm_wla/train_files/run_multi_source_train_mmdit_from_scratch.sh
```

训练输出默认写入 `run_root_dir/run_id`。批大小、训练步数、保存间隔和其他训练参数可在启动脚本中调整。

