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
UnifoLM-WLA-1.0/
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

hf download unitreerobotics/G1_WBT_Brainco_Pickup_Pillow --repo-type dataset \
    --local-dir $DATA_ROOT/UnifoLM_WBT_Dataset/G1_WBT_Brainco_Pickup_Pillow
```

数据需要按照 Dex1 和 WBT 两种类型分别组织；同一种类型的目录下可以包含多个任务。
推荐的目录结构如下：

```text
$DATA_ROOT/
├── UnifoLM_G1_Dex1_Dataset/
│   ├── G1_Dex1_MountCamera_Dataset/
│   └── G1_Dex1_Stack_Block/
└── UnifoLM_WBT_Dataset/
    ├── G1_WBT_Brainco_Pickup_Pillow/
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
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0/checkpoints/model.safetensors \
    --host 0.0.0.0 --port 8600 --instruction "pick up the object"
```

### 2. 评估正在运行的服务

在服务保持运行的情况下，另开一个终端执行：

```bash
python -m model_server.eval_local_episode_wbc_msgpack_server_only \
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0/checkpoints/model.safetensors \
    --data_config_path unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml \
    --host 127.0.0.1 --port 8600 --episode_idx 0 \
    --save_dir results/eval_local_episode_wbc_msgpack
```

## 微调 UnifoLM-WLA-1.0-Base 动作专家

[`examples/unifolm_wla/train_files/run_finetune_mmdit_frozen_vlm.sh`](../examples/unifolm_wla/train_files/run_finetune_mmdit_frozen_vlm.sh)
会在新数据上微调一个已发布的 `UnifoLM-WLA-1.0-Base` checkpoint（例如
[`unitreerobotics/UnifoLM-WLA-1.0-Base`](https://huggingface.co/unitreerobotics/UnifoLM-WLA-1.0-Base)）。
VLM 主干被冻结（`trainer.freeze_modules: qwen_vl_interface`），只训练动作专家
（DiT）头和 robot-state projector。

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

## 从零训练

### 1. 下载基础视觉语言模型

选择并下载以下任一基础模型：

- [UnifoLM-ER-1](https://huggingface.co/unitreerobotics/UnifoLM-ER-1)
- [UnifoLM-ER-Flow](https://huggingface.co/unitreerobotics/UnifoLM-ER-Flow)

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

