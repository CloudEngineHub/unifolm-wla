# Training and Evaluating an Action Expert

[Chinese](train_action_expert.md) | **English**

This guide explains how to evaluate a released UnifoLM-WLA action expert
checkpoint against a local episode, and how to train one from scratch using a
pretrained UnifoLM-ER vision-language model. All commands assume that the
current working directory is the project root.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1. Install `uv`:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install the project (from the project root):

   ```bash
   uv sync
   ```

3. Install `flash-attn`:

   ```bash
   uv pip install flash-attn --no-build-isolation
   ```

4. `uv sync` creates an isolated `.venv` and does not touch the system Python.
   Activate it once:

   ```bash
   source .venv/bin/activate
   ```

   After that, use `python` / `hf` / `accelerate` directly. The rest of this
   document assumes the virtualenv is active.

## Evaluating a Checkpoint

[`examples/unifolm_wla/eval_files/unitree/eval_local_episode.py`](../examples/unifolm_wla/eval_files/unitree/eval_local_episode.py)
runs a trained (or released) checkpoint chunk-by-chunk across one full episode
of a local Unitree dataset and plots predicted vs. ground-truth actions for
every dimension, both in absolute end-effector pose and in the model's native
relative representation.

### 1. Download the Model

Download the released checkpoint from the
[UnifoLM-WLA-1.0 model collection](https://huggingface.co/collections/unitreerobotics/unifolm-wla-10)
to a local directory, for example with `hf`:

```bash
hf download unitreerobotics/UnifoLM-WLA-1.0-Base \
    --local-dir playground/Pretrained_models/UnifoLM-WLA-1.0-Base
```

The downloaded directory should have the following layout:

```text
UnifoLM-WLA-1.0-Base/
├── checkpoints/
│   └── model.safetensors
├── config.yaml
├── dataset_statistics.json
└── tokenizer/
```

### 2. Download and Configure the Evaluation Data

Download the evaluation data from the
[UnifoLM-WLA-1.0 dataset collection](https://huggingface.co/collections/unitreerobotics/unifolm-wla-10).
Each task is a separate dataset repo; download each into a subdirectory under
its Dex1/WBT type directory, for example with `hf`:

```bash
export DATA_ROOT=/path/to/unifolm_data

hf download unitreerobotics/G1_Dex1_MountCamera_Dataset --repo-type dataset \
    --local-dir $DATA_ROOT/UnifoLM_G1_Dex1_Dataset/G1_Dex1_MountCamera_Dataset

hf download unitreerobotics/G1_WBT_Brainco_Supermarket_Shelf_Organizing --repo-type dataset \
    --local-dir $DATA_ROOT/UnifoLM_WBT_Dataset/G1_WBT_Brainco_Supermarket_Shelf_Organizing
```

Organize the data by the Dex1 and WBT dataset types. Each type may contain
multiple task directories. The recommended directory structure is:

```text
$DATA_ROOT/
├── UnifoLM_G1_Dex1_Dataset/
│   ├── G1_Dex1_MountCamera_Dataset/
│   └── G1_Dex1_Stack_Block/
└── UnifoLM_WBT_Dataset/
    ├── G1_WBT_Brainco_Supermarket_Shelf_Organizing/
    └── G1_WBT_Brainco_Make_The_Bed/
```

Then edit
[`unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml`](../unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml):

1. Set `data_base` to `$DATA_ROOT` — the common parent directory containing the Dex1 and WBT directories.
2. For Dex1 data, inherit from `*unitree_base` and set `data_path` to `UnifoLM_G1_Dex1_Dataset`.
3. For WBT data, inherit from `*unitree_fullbody_base` and set `data_path` to `UnifoLM_WBT_Dataset`.
4. Set `cache_dir` to a local cache directory with sufficient free space.

Example configuration:

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

### 3. Run Evaluation

```bash
python -m examples.unifolm_wla.eval_files.unitree.eval_local_episode \
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
    --data_config_path unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml \
    --episode_idx 0 \
    --save_dir results/eval_local_episode
```

Run this from the project root so `unifolm_wla` resolves as a package.
`--ckpt_path` accepts either a `.safetensors` or `.pt` checkpoint file; the
script expects the standard run layout — `config.yaml` and
`dataset_statistics.json` as siblings of the `checkpoints/` directory
(`<run_dir>/checkpoints/<name>.safetensors`, `<run_dir>/config.yaml`,
`<run_dir>/dataset_statistics.json`). If `dataset_statistics.json` is
missing, the script regenerates it from `--data_config_path` automatically.

## Model Server

[`model_server/action_server_wbc_msgpack_unitree.py`](../model_server/action_server_wbc_msgpack_unitree.py)
runs a trained checkpoint behind a websocket/msgpack action server for the
Unitree G1 whole-body-control (WBC) client.
[`model_server/eval_local_episode_wbc_msgpack_server_only.py`](../model_server/eval_local_episode_wbc_msgpack_server_only.py)
drives an already-running server over one full local episode and compares its
predictions against ground truth.

> **Dex1 only.** `_ACTIVE_SLOT_SPECS` in
> [`model_server/unifolm_wla_action_adapter.py`](../model_server/unifolm_wla_action_adapter.py)
> mirrors `unitree.yaml`'s `unitree_base` (Dex1) `action_keys` — it has no
> entry for `left_fig6d` / `right_fig6d` (dexterous-hand finger angles) or
> `base_pose` (whole-body relative base motion), which only exist under
> `unitree_fullbody_base` (WBT). Consequently the action mask built from it,
> and the client obs/action protocol hardcoded in
> `action_server_wbc_msgpack_unitree.py` (`_DATA_KEYS`, `_build_state_unnorm`,
> `_encode_action`), only cover the two-finger-gripper Dex1 action space. To
> serve a `UnifoLM_WBT` checkpoint you would need to: (1) add
> `left_fig6d`/`right_fig6d`/`base` (`base_rotvec`) entries to
> `_ACTIVE_SLOT_SPECS`, and (2) extend the server's obs/action protocol (obs
> keys for finger angles + base pose, and the corresponding `action.*` keys in
> `_encode_action`) to carry those fields — the current protocol does not.

### 1. Start the Server

```bash
python -m model_server.action_server_wbc_msgpack_unitree \
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
    --host 0.0.0.0 --port 8600 --instruction "pick up the object"
```

### 2. Evaluate the Running Server

With the server from step 1 still running, in another terminal:

```bash
python -m model_server.eval_local_episode_wbc_msgpack_server_only \
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
    --data_config_path unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml \
    --host 127.0.0.1 --port 8600 --episode_idx 0 \
    --save_dir results/eval_local_episode_wbc_msgpack
```

## Fine-tuning the UnifoLM-WLA-1.0-Base Action Expert

[`examples/unifolm_wla/train_files/run_finetune_mmdit_frozen_vlm.sh`](../examples/unifolm_wla/train_files/run_finetune_mmdit_frozen_vlm.sh)
fine-tunes a released `UnifoLM-WLA-1.0-Base` checkpoint (e.g.
[`unitreerobotics/UnifoLM-WLA-1.0-Base`](https://huggingface.co/unitreerobotics/UnifoLM-WLA-1.0-Base))
on new data. The VLM backbone is frozen (`trainer.freeze_modules:
qwen_vl_interface`), so only the action-expert (DiT) head and the robot-state
projector train. The default config targets a single 24GB GPU; lower the
batch size in
[`mmdit_finetune_frozen_vlm.yaml`](../unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml)
if you have less VRAM.

### 1. Download the Base Checkpoint

Download the released checkpoint to fine-tune from, the same way as in
[Download the Model](#1-download-the-model) above:

```bash
hf download unitreerobotics/UnifoLM-WLA-1.0-Base \
    --local-dir playground/Pretrained_models/UnifoLM-WLA-1.0-Base
```

Skip this step if you already downloaded it there.

### 2. Configure the Data

Download and configure the data exactly as described in
[Download and Configure the Evaluation Data](#2-download-and-configure-the-evaluation-data)
above.

### 3. Start Training

```bash
base_model_dir=playground/Pretrained_models/UnifoLM-WLA-1.0-Base \
bash examples/unifolm_wla/train_files/run_finetune_mmdit_frozen_vlm.sh
```

`base_model_dir` is the directory downloaded in step 1. The script points
`framework.qwenvl.base_vlm` at `${base_model_dir}/tokenizer` (architecture
config + tokenizer only — the base checkpoint's own VLM weights are loaded via
`trainer.pretrained_checkpoint` instead, so no separate VLM download is
needed) and `trainer.pretrained_checkpoint` at
`${base_model_dir}/checkpoints/model.safetensors`.

Edit [`unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml`](../unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml)
to adjust the dataset config path, learning rate, batch size, and training
steps. To co-train the VLM instead of freezing it (requires more VRAM), clear
`trainer.freeze_modules`.

Fine-tuning outputs are written to `run_root_dir/run_id` (defaults to
`playground/Checkpoints/finetune_wla_base_frozen_vlm`), in the same run
layout consumed by [Evaluating a Checkpoint](#evaluating-a-checkpoint) and
[Model Server](#model-server) above.

## LoRA Fine-tuning

Instead of full-parameter fine-tuning, either backbone — `qwen_vl_interface`
(the VLM) and/or `action_model` (the DiT action head) — can be adapted with
[LoRA](https://arxiv.org/abs/2106.09685) via a `trainer.lora` config block.
This trains a small set of low-rank adapter weights while the rest of that
backbone stays frozen, which is useful when co-training the VLM (normally
frozen for VRAM reasons) or when a full action-model fine-tune is more
capacity than the target data needs.

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

Each backbone's sub-block is independent — enable one, both, or neither.
A module with LoRA enabled does **not** need to also appear in
`trainer.freeze_modules`: injecting LoRA already freezes that backbone's base
weights and leaves only the adapter (`lora_A`/`lora_B`) parameters trainable.

[`unifolm_wla/config/training/mmdit_lora_frozen_vlm.yaml`](../unifolm_wla/config/training/mmdit_lora_frozen_vlm.yaml)
is a ready-to-use example, built on top of `mmdit_finetune_frozen_vlm.yaml`:
the VLM stays frozen and the action-model head is LoRA-adapted instead of
fully fine-tuned.
[`examples/unifolm_wla/train_files/run_lora_finetune_mmdit_frozen_vlm.sh`](../examples/unifolm_wla/train_files/run_lora_finetune_mmdit_frozen_vlm.sh)
launches it the same way as
[Fine-tuning the UnifoLM-WLA-1.0-Base Action Expert](#fine-tuning-the-unifolm-wla-10-base-action-expert)
above:

```bash
base_model_dir=playground/Pretrained_models/UnifoLM-WLA-1.0-Base \
bash examples/unifolm_wla/train_files/run_lora_finetune_mmdit_frozen_vlm.sh
```

At each `save_interval` (and at training end), a small adapter-only checkpoint
(`steps_<n>_adapter.safetensors` / `adapter.safetensors`, containing only the
`lora_*` tensors) is written next to the regular full-model checkpoint — this
is typically orders of magnitude smaller and is the artifact to distribute for
inference. Resuming (`trainer.is_resume: true`) always continues from the
regular full checkpoint, which already contains both the frozen base weights
and the LoRA weights.

To sanity-check that LoRA injection, freezing, and per-module learning rates
are wired correctly before launching a real run, use the standalone smoke
test:

```bash
python unifolm_wla/scripts/smoke_test_lora_injection.py \
    --config_yaml unifolm_wla/config/training/mmdit_debug_lora_frozen_vlm.yaml
```

## Training from Scratch

### 1. Download the Base Vision-Language Model

Download either of the following base models, for example with `hf`:

```bash
hf download unitreerobotics/UnifoLM-ER-1 \
    --local-dir playground/Pretrained_models/UnifoLM-ER-1

# or
hf download unitreerobotics/UnifoLM-ER-Flow \
    --local-dir playground/Pretrained_models/UnifoLM-ER-Flow
```

After downloading the model, open
[`examples/unifolm_wla/train_files/run_multi_source_train_mmdit_from_scratch.sh`](../examples/unifolm_wla/train_files/run_multi_source_train_mmdit_from_scratch.sh)
and set `base_vlm` to the full path of the local model directory. For example:

```bash
base_vlm=/path/to/UnifoLM-ER-1
```

To use UnifoLM-ER-Flow instead, set the path to its local directory:

```bash
base_vlm=/path/to/UnifoLM-ER-Flow
```

### 2. Download and Configure the Training Data

Download the training data from the
[UnifoLM-WLA-1.0 dataset collection](https://huggingface.co/collections/unitreerobotics/unifolm-wla-10).
Organize the data by the Dex1 and WBT dataset types, following the same
layout and `unitree.yaml` configuration steps described above.

When `multi_task: true`, the loader treats each task directory under
`data_path` as a sub-dataset. All sources with `enabled: true` are combined
using `ConcatDataset`, so later sources are not skipped. Sampling is
proportional to the number of samples in each source; the current implementation
does not apply the configured `weight` value.

When LeRobot data is loaded for the first time, the loader creates an Arrow
cache under `cache_dir/arrow_cache`. Subsequent runs reuse this cache. Place
`cache_dir` on a disk with sufficient capacity and good read/write performance.

### 3. Train on a Single Node

After verifying `base_vlm`, `run_root_dir`, and the dataset configuration path
in the launch script, run the following command from the project root:

```bash
bash examples/unifolm_wla/train_files/run_multi_source_train_mmdit_from_scratch.sh
```

By default, the script launches one process for each GPU detected by
`nvidia-smi -L`. Set `NUM_PROCESSES` to use a specific number of processes:

```bash
NUM_PROCESSES=4 bash examples/unifolm_wla/train_files/run_multi_source_train_mmdit_from_scratch.sh
```

Training outputs are written to `run_root_dir/run_id`. Batch size, training
steps, checkpoint intervals, and other training parameters can be adjusted in
the launch script.


