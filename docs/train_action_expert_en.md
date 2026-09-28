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

## Evaluating a Checkpoint

[`examples/pretrain/eval_files/unitree/eval_local_episode.py`](../examples/pretrain/eval_files/unitree/eval_local_episode.py)
runs a trained (or released) checkpoint chunk-by-chunk across one full episode
of a local Unitree dataset and plots predicted vs. ground-truth actions for
every dimension, both in absolute end-effector pose and in the model's native
relative representation.

### 1. Download the Model

Download the released checkpoint from the
[UnifoLM-WLA-1.0 model collection](https://huggingface.co/collections/unitreerobotics/unifolm-wla-10)
to a local directory, for example with `hf`:

```bash
hf download unitreerobotics/UnifoLM-WLA-1.0 \
    --local-dir playground/Pretrained_models/UnifoLM-WLA-1.0
```

The downloaded directory should have the following layout:

```text
UnifoLM-WLA-1.0/
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

hf download unitreerobotics/G1_WBT_Brainco_Pickup_Pillow --repo-type dataset \
    --local-dir $DATA_ROOT/UnifoLM_WBT_Dataset/G1_WBT_Brainco_Pickup_Pillow
```

Organize the data by the Dex1 and WBT dataset types. Each type may contain
multiple task directories. The recommended directory structure is:

```text
$DATA_ROOT/
├── UnifoLM_G1_Dex1_Dataset/
│   ├── G1_Dex1_MountCamera_Dataset/
│   └── G1_Dex1_Stack_Block/
└── UnifoLM_WBT_Dataset/
    ├── G1_WBT_Brainco_Pickup_Pillow/
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
python -m examples.pretrain.eval_files.unitree.eval_local_episode \
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0/checkpoints/model.safetensors \
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
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0/checkpoints/model.safetensors \
    --host 0.0.0.0 --port 8600 --instruction "pick up the object"
```

### 2. Evaluate the Running Server

With the server from step 1 still running, in another terminal:

```bash
python -m model_server.eval_local_episode_wbc_msgpack_server_only \
    --ckpt_path playground/Pretrained_models/UnifoLM-WLA-1.0/checkpoints/model.safetensors \
    --data_config_path unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml \
    --host 127.0.0.1 --port 8600 --episode_idx 0 \
    --save_dir results/eval_local_episode_wbc_msgpack
```

## Training from Scratch

### 1. Download the Base Vision-Language Model

Download either of the following base models:

- [UnifoLM-ER-1](https://huggingface.co/unitreerobotics/UnifoLM-ER-1)
- [UnifoLM-ER-Flow](https://huggingface.co/unitreerobotics/UnifoLM-ER-Flow)

After downloading the model, open
[`examples/pretrain/train_files/run_multi_source_train_mmdit.sh`](../examples/pretrain/train_files/run_multi_source_train_mmdit.sh)
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
bash examples/pretrain/train_files/run_multi_source_train_mmdit.sh
```

By default, the script launches one process for each GPU detected by
`nvidia-smi -L`. Set `NUM_PROCESSES` to use a specific number of processes:

```bash
NUM_PROCESSES=4 bash examples/pretrain/train_files/run_multi_source_train_mmdit.sh
```

Training outputs are written to `run_root_dir/run_id`. Batch size, training
steps, checkpoint intervals, and other training parameters can be adjusted in
the launch script.
