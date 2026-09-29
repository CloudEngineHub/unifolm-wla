"""Full-episode inference test on a local Unitree dataset.

Evaluates a trained VLA checkpoint chunk-by-chunk across one full episode
(non-overlapping chunks by default) and visualises the stitched ground-truth
vs predicted trajectory for every action dimension in a single figure.

Differences vs ``eval_local_dataset.py`` (which evaluates one chunk per
randomly sampled frame):
  - Iterates ``frame = from_idx, from_idx + stride, ...`` over a single
    episode (``stride = chunk_size`` by default → no overlap).
  - Stitches each chunk's first ``stride`` predictions into one episode-long
    trajectory.
  - Renders a single image per representation (absolute EE / relative EE)
    with vertical guides at chunk boundaries.

Usage:
    python -m examples.unifolm_wla.eval_files.unitree.eval_local_episode \\
        --ckpt_path playground/Checkpoints/<run_id>/checkpoints/steps_60000_model.safetensors \\
        --data_config_path unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml \\
        --episode_idx 0 \\
        --save_dir results/eval_local_episode
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

_WORKSPACE_ROOT = str(Path(__file__).resolve().parents[4])
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from unifolm_wla.dataloader.multi_source_dataset.config import load_config
from unifolm_wla.dataloader.multi_source_dataset.single_source_dataset import create_single_source_dataset
from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES, STATE_SLICES
from unifolm_wla.dataloader.multi_source_dataset.se3_utils import pose_to_se3, rotvec_to_matrix, matrix_to_rotvec
from scipy.spatial.transform import Rotation

from unifolm_wla.model.framework.base_framework import baseframework, build_framework
from unifolm_wla.model.framework.share_tools import dict_to_namespace, read_mode_config

# Absolute EE dims: left xyz(3) + rotvec(3) + grip(1) + right xyz(3) + rotvec(3) + grip(1) = 14
ABS_DIM_LABELS = [
    "L_x", "L_y", "L_z", "L_rv0", "L_rv1", "L_rv2", "L_grip",
    "R_x", "R_y", "R_z", "R_rv0", "R_rv1", "R_rv2", "R_grip",
]
ABS_DIM_COLOURS = ["steelblue"] * 6 + ["steelblue"] + ["darkorange"] * 6 + ["darkorange"]

# Relative EE dims: left xyz(3) + rotvec(3) + grip(1) + right xyz(3) + rotvec(3) + grip(1) = 14
REL_DIM_LABELS = [
    "L_dx", "L_dy", "L_dz", "L_rv0", "L_rv1", "L_rv2", "L_grip",
    "R_dx", "R_dy", "R_dz", "R_rv0", "R_rv1", "R_rv2", "R_grip",
]
REL_DIM_COLOURS = ["mediumseagreen"] * 6 + ["mediumseagreen"] + ["mediumpurple"] * 6 + ["mediumpurple"]


def _build_example_from_sample(sample: dict) -> dict:
    """Convert a multi_source_dataset sample to predict_action input format.

    Mirrors `_adapt_multi_source_batch` in QwenGR00T.py exactly so the prompt
    built at eval matches training: keep the dict (= config image_keys) order
    — do NOT sort — and drop masked-out cameras from BOTH images and roles so
    they stay positionally aligned (build_qwenvl_inputs zips roles with images).
    """
    roles = [r for r in sample["images"].keys() if sample["image_mask"].get(r, False)]
    images = [TF.to_pil_image(sample["images"][r]) for r in roles]
    return {
        "image": images,
        "image_roles": roles,
        "lang": sample["task"],
        "action_mask": _to_numpy(sample["action_mask"]),
        "state": _to_numpy(sample["state"]),
        "state_mask": _to_numpy(sample["state_mask"]),
        "arm_type": sample["arm_type"],
        "robot_type": sample["robot_type"],
    }


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().float().numpy()
    return np.array(x, dtype=np.float32)


def unnormalize(normalized: np.ndarray, offset: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return normalized * scale + offset


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation back to rotation matrix.

    rot6d: (..., 6) — first two columns of R in row-major: [R00,R10,R20, R01,R11,R21]
    Returns: (..., 3, 3) rotation matrices
    """
    shape = rot6d.shape[:-1]
    r = rot6d.reshape(-1, 6)
    # Columns of R
    a1 = r[:, 0:3]  # first column
    a2 = r[:, 3:6]  # second column
    # Gram-Schmidt: normalize a1, orthogonalize a2, cross for a3
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=1)
    R = np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3) columns are b1, b2, b3
    return R.reshape(*shape, 3, 3)

def ensure_dataset_statistics(ckpt_path: Path, data_config_path: str):
    """Generate dataset_statistics.json if missing in the run directory."""
    run_dir = ckpt_path.parents[1]
    stats_path = run_dir / "dataset_statistics.json"
    if stats_path.exists():
        return

    print(f"dataset_statistics.json not found in {run_dir}, generating from {data_config_path} ...")
    config = load_config(data_config_path)

    stats = {}
    for ds_cfg in config.datasets:
        if not ds_cfg.enabled:
            continue
        src = create_single_source_dataset(ds_cfg, config)
        stats[ds_cfg.name] = {
            "action": {
                "offset": src._action_norm_offset.tolist(),
                "scale": src._action_norm_scale.tolist(),
            },
            "state": {
                "offset": src._state_norm_offset.tolist(),
                "scale": src._state_norm_scale.tolist(),
            },
        }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved dataset_statistics.json at {stats_path}")

def load_model(ckpt_path: Path, base_vlm: str | None = None) -> baseframework:
    """Load a checkpoint via `baseframework.from_pretrained`'s own logic, with an
    optional override for `framework.qwenvl.base_vlm` (checkpoints trained on a
    cluster record a base_vlm path that doesn't exist on this machine).
    """
    model_config, norm_stats = read_mode_config(ckpt_path)
    if base_vlm:
        model_config["framework"]["qwenvl"]["base_vlm"] = base_vlm
    cfg = dict_to_namespace(model_config)
    cfg.trainer.pretrained_checkpoint = None

    model = build_framework(cfg=cfg)
    model.norm_stats = norm_stats

    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt_path))
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model

def relative_to_absolute_ee(state_unnorm: np.ndarray, rel_actions_unnorm: np.ndarray) -> np.ndarray:
    """Convert relative EE actions to absolute EE poses.

    T_abs = T_curr @ T_rel for each timestep.

    Args:
        state_unnorm: (60,) unnormalized state with absolute EE as xyz+rot6d
        rel_actions_unnorm: (T, 54) unnormalized relative actions
    Returns:
        abs_ee: (T, 14) — [L_xyz(3), L_rotvec(3), L_grip(1), R_xyz(3), R_rotvec(3), R_grip(1)]
    """
    T = rel_actions_unnorm.shape[0]

    # Extract current EE poses from state (xyz + rot6d)
    left_state = state_unnorm[STATE_SLICES["left_xyz_rot6d"]]   # (9,): xyz(3) + rot6d(6)
    right_state = state_unnorm[STATE_SLICES["right_xyz_rot6d"]]  # (9,): xyz(3) + rot6d(6)

    left_xyz_curr = left_state[:3]
    left_R_curr = rot6d_to_matrix(left_state[3:9].reshape(1, 6))[0]  # (3,3)
    right_xyz_curr = right_state[:3]
    right_R_curr = rot6d_to_matrix(right_state[3:9].reshape(1, 6))[0]  # (3,3)

    T_left_curr = pose_to_se3(left_xyz_curr, left_R_curr)    # (4,4)
    T_right_curr = pose_to_se3(right_xyz_curr, right_R_curr)  # (4,4)

    # Extract relative actions (xyz + rotvec)
    left_rel = rel_actions_unnorm[:, SLICES["left_xyz_rotvec"]]    # (T, 6)
    right_rel = rel_actions_unnorm[:, SLICES["right_xyz_rotvec"]]  # (T, 6)
    left_grip = rel_actions_unnorm[:, SLICES["left_gripper"]]      # (T, 1)
    right_grip = rel_actions_unnorm[:, SLICES["right_gripper"]]    # (T, 1)

    # Build T_rel for each timestep and compute T_abs = T_curr @ T_rel
    left_R_rel = rotvec_to_matrix(left_rel[:, 3:6])   # (T, 3, 3)
    left_T_rel = pose_to_se3(left_rel[:, :3], left_R_rel)  # (T, 4, 4)
    left_T_abs = T_left_curr @ left_T_rel  # (T, 4, 4) broadcast

    right_R_rel = rotvec_to_matrix(right_rel[:, 3:6])  # (T, 3, 3)
    right_T_rel = pose_to_se3(right_rel[:, :3], right_R_rel)  # (T, 4, 4)
    right_T_abs = T_right_curr @ right_T_rel  # (T, 4, 4) broadcast

    # Extract absolute xyz
    left_xyz_abs = left_T_abs[:, :3, 3]   # (T, 3)
    right_xyz_abs = right_T_abs[:, :3, 3]  # (T, 3)

    # Extract absolute rotation as rotvec
    left_rotvec = Rotation.from_matrix(left_T_abs[:, :3, :3]).as_rotvec()   # (T, 3)
    right_rotvec = Rotation.from_matrix(right_T_abs[:, :3, :3]).as_rotvec()  # (T, 3)

    # Combine: [L_xyz(3), L_rotvec(3), L_grip(1), R_xyz(3), R_rotvec(3), R_grip(1)] = 14 dims
    abs_ee = np.concatenate([
        left_xyz_abs, left_rotvec, left_grip,
        right_xyz_abs, right_rotvec, right_grip,
    ], axis=1)
    return abs_ee


def extract_rel_ee(unnorm: np.ndarray) -> np.ndarray:
    """Extract the 14 relative EE dims from (T, 54) unnormalized relative actions.

    Returns (T, 14): [L_xyz(3), L_rotvec(3), L_grip(1), R_xyz(3), R_rotvec(3), R_grip(1)]
    """
    left_xyz_rv = unnorm[:, SLICES["left_xyz_rotvec"]]   # (T, 6)
    right_xyz_rv = unnorm[:, SLICES["right_xyz_rotvec"]]  # (T, 6)
    left_grip = unnorm[:, SLICES["left_gripper"]]          # (T, 1)
    right_grip = unnorm[:, SLICES["right_gripper"]]        # (T, 1)
    return np.concatenate([left_xyz_rv, left_grip, right_xyz_rv, right_grip], axis=1)  # (T, 14)



def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VLA on a full local episode (chunk-by-chunk).")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--data_config_path", type=str,
                   default="unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml")
    p.add_argument("--source_idx", type=int, default=0,
                   help="Index into the enabled dataset sources in data_config_path.")
    p.add_argument("--base_vlm", type=str, default=None,
                   help="Override framework.qwenvl.base_vlm (checkpoint's own path "
                        "may not exist on this machine).")
    p.add_argument("--task_idx", type=int, default=0,
                   help="Sub-dataset (task) index inside the chosen source.")
    p.add_argument("--episode_idx", type=int, default=0,
                   help="Episode index inside the chosen sub-dataset.")
    p.add_argument("--source_name", type=str, default=None,
                   help="Override source name used to read norm stats (auto if omitted).")
    p.add_argument("--stride", type=int, default=None,
                   help="Step in frames between chunk inferences. Default = chunk_size (non-overlapping).")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Optional cap on the number of episode frames evaluated.")
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--save_dir", type=str, default="results/eval_local_episode")
    return p.parse_args()


def _episode_frame_range(src, task_idx: int, episode_idx: int) -> tuple[int, int, int]:
    """Resolve (task_offset, from_idx, to_idx) for an episode inside ``src``.

    ``from_idx``/``to_idx`` are absolute frame indices into the chosen sub-dataset.
    ``task_offset`` is the source-local start of that sub-dataset.
    """
    if task_idx < 0 or task_idx >= len(src._datasets):
        raise IndexError(
            f"task_idx={task_idx} out of range; source has {len(src._datasets)} sub-datasets."
        )
    sub_ds = src._datasets[task_idx]
    episodes = sub_ds.meta.episodes
    if episode_idx < 0 or episode_idx >= len(episodes):
        raise IndexError(
            f"episode_idx={episode_idx} out of range; sub-dataset has {len(episodes)} episodes."
        )
    ep = episodes[episode_idx]
    # `dataset_from_index` / `dataset_to_index` are absolute over the whole
    # sub-dataset. The sub-dataset's source-local offset is the cumulative
    # length of preceding sub-datasets.
    prev_cum = src._cum_lengths[task_idx - 1] if task_idx > 0 else 0
    return int(prev_cum), int(ep["dataset_from_index"]), int(ep["dataset_to_index"])


def _plot_episode_grid(gt: np.ndarray, pred: np.ndarray, chunk_starts: list[int],
                       dim_labels: list, dim_colours: list,
                       title: str, save_path: Path) -> None:
    """Plot all dims for a single episode in one figure.

    Args:
        gt, pred: (L, D) arrays. L = episode length.
        chunk_starts: frame indices (relative to start of episode) where a new
            chunk begins — used for thin vertical guides.
        dim_labels / dim_colours: per-dim metadata.
    """
    L, D = gt.shape
    n_cols = 2
    n_rows = (D + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6.5, n_rows * 1.8),
                             constrained_layout=True, sharex=True)
    axes = np.array(axes).reshape(n_rows, n_cols)
    fig.suptitle(title, fontsize=10)

    steps = np.arange(L)
    for d in range(D):
        ax = axes[d // n_cols][d % n_cols]
        l1_d = float(np.mean(np.abs(pred[:, d] - gt[:, d])))
        ax.plot(steps, gt[:, d], color="steelblue", linewidth=1.0, label="GT")
        ax.plot(steps, pred[:, d], color="darkorange", linewidth=1.0,
                linestyle="--", label="Pred")
        for s in chunk_starts[1:]:  # skip 0 to avoid a line at the left edge
            ax.axvline(s, color="lightgray", linewidth=0.5, alpha=0.7)
        ax.set_title(f"{dim_labels[d]}   L1={l1_d:.4f}", fontsize=8,
                     color=dim_colours[d])
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
    for d in range(D, n_rows * n_cols):
        axes[d // n_cols][d % n_cols].set_visible(False)
    axes[-1][0].set_xlabel("frame in episode", fontsize=8)
    if n_cols > 1:
        axes[-1][-1].set_xlabel("frame in episode", fontsize=8)
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt_path)

    # 0. Self-heal dataset_statistics.json next to the checkpoint.
    ensure_dataset_statistics(ckpt_path, args.data_config_path)

    # 1. Model.
    print(f"Loading model from {ckpt_path} ...")
    model = load_model(ckpt_path, base_vlm=args.base_vlm)
    if args.use_bf16:
        model = model.to(torch.bfloat16)
    model = model.to("cuda").eval()
    print("Model loaded.")

    # 2. Dataset.
    print(f"Building dataset from {args.data_config_path} ...")
    config = load_config(args.data_config_path)
    enabled = [d for d in config.datasets if d.enabled]
    if args.source_idx < 0 or args.source_idx >= len(enabled):
        raise IndexError(
            f"source_idx={args.source_idx} out of range; "
            f"{len(enabled)} enabled dataset source(s) in {args.data_config_path}."
        )
    ds_cfg = enabled[args.source_idx]
    src = create_single_source_dataset(ds_cfg, config)
    src_name = ds_cfg.name
    chunk_size = src.chunk_size
    print(f"Using source[{args.source_idx}]='{src_name}', chunk_size={chunk_size}")

    # 3. Episode bounds.
    task_offset, from_idx, to_idx = _episode_frame_range(src, args.task_idx, args.episode_idx)
    ep_len_raw = to_idx - from_idx
    if args.max_frames is not None:
        ep_len = min(ep_len_raw, args.max_frames)
    else:
        ep_len = ep_len_raw
    print(f"Episode {args.episode_idx} in task {args.task_idx}: "
          f"frames [{from_idx}, {from_idx + ep_len}) (raw episode length {ep_len_raw})")

    stride = args.stride if args.stride is not None else chunk_size
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if stride > chunk_size:
        print(f"WARN: stride={stride} > chunk_size={chunk_size}; "
              f"there will be gaps between consecutive chunks.")

    # 4. Norm stats.
    source_name = args.source_name or src_name
    if source_name not in model.norm_stats:
        # Fall back to whatever's there.
        source_name = next(iter(model.norm_stats.keys()))
        print(f"Source '{src_name}' missing in norm_stats; falling back to '{source_name}'")
    action_stats = model.norm_stats[source_name]["action"]
    offset = np.array(action_stats["offset"], dtype=np.float32)
    scale = np.array(action_stats["scale"], dtype=np.float32)
    print(f"Using norm stats for '{source_name}'")

    # 5. Chunk-by-chunk inference.
    gt_abs_all = np.zeros((ep_len, 14), dtype=np.float32)
    pred_abs_all = np.zeros((ep_len, 14), dtype=np.float32)
    gt_rel_all = np.zeros((ep_len, 14), dtype=np.float32)
    pred_rel_all = np.zeros((ep_len, 14), dtype=np.float32)
    fill_mask = np.zeros(ep_len, dtype=bool)
    chunk_starts: list[int] = []
    task_label = None

    chunk_frame_starts = list(range(0, ep_len, stride))
    for chunk_local_start in tqdm(chunk_frame_starts, desc="Chunks"):
        frame_in_sub = from_idx + chunk_local_start
        src_local_idx = task_offset + frame_in_sub
        sample = src[src_local_idx]
        if task_label is None:
            task_label = sample.get("task", "")

        gt_action_norm = _to_numpy(sample["action"])         # (T, 54)
        state_unnorm = _to_numpy(sample["state_unnorm"])    # (60,)
        example = _build_example_from_sample(sample)

        output = model.predict_action([example])
        pred_norm = output["normalized_actions"][0]          # (T, 54)

        T = min(len(pred_norm), len(gt_action_norm))
        # Number of frames this chunk should fill in the episode buffer.
        write_len = min(stride, ep_len - chunk_local_start, T)
        if write_len <= 0:
            continue

        pred_n = pred_norm[:T]
        gt_n = gt_action_norm[:T]
        pred_unnorm = unnormalize(pred_n, offset, scale)
        gt_unnorm = unnormalize(gt_n, offset, scale)

        gt_rel = extract_rel_ee(gt_unnorm)                   # (T, 14)
        pred_rel = extract_rel_ee(pred_unnorm)               # (T, 14)
        gt_abs = relative_to_absolute_ee(state_unnorm, gt_unnorm)     # (T, 14)
        pred_abs = relative_to_absolute_ee(state_unnorm, pred_unnorm) # (T, 14)

        sl = slice(chunk_local_start, chunk_local_start + write_len)
        gt_rel_all[sl] = gt_rel[:write_len]
        pred_rel_all[sl] = pred_rel[:write_len]
        gt_abs_all[sl] = gt_abs[:write_len]
        pred_abs_all[sl] = pred_abs[:write_len]
        fill_mask[sl] = True
        chunk_starts.append(chunk_local_start)

    if not fill_mask.any():
        raise RuntimeError("No frames were evaluated; check episode bounds and stride.")
    if not fill_mask.all():
        # Trim trailing gap so plots don't show flat zeros at the tail.
        last_filled = int(np.where(fill_mask)[0].max()) + 1
        gt_abs_all = gt_abs_all[:last_filled]
        pred_abs_all = pred_abs_all[:last_filled]
        gt_rel_all = gt_rel_all[:last_filled]
        pred_rel_all = pred_rel_all[:last_filled]
        ep_len = last_filled
        chunk_starts = [s for s in chunk_starts if s < last_filled]

    # 6. Plots.
    task_str = (task_label or "").replace("\n", " ")[:80]
    title_suffix = (
        f"source='{src_name}', task={args.task_idx}, ep={args.episode_idx}, "
        f"len={ep_len}, stride={stride}, chunk={chunk_size} | {task_str}"
    )

    abs_png = save_dir / f"episode{args.episode_idx:04d}_abs.png"
    print(f"Saving absolute-EE episode plot → {abs_png}")
    _plot_episode_grid(
        gt_abs_all, pred_abs_all, chunk_starts,
        ABS_DIM_LABELS, ABS_DIM_COLOURS,
        f"[ABS] {title_suffix}", abs_png,
    )

    rel_png = save_dir / f"episode{args.episode_idx:04d}_rel.png"
    print(f"Saving relative-EE episode plot → {rel_png}")
    _plot_episode_grid(
        gt_rel_all, pred_rel_all, chunk_starts,
        REL_DIM_LABELS, REL_DIM_COLOURS,
        f"[REL] {title_suffix}", rel_png,
    )

    # 7. Summary.
    abs_l1_per_dim = np.mean(np.abs(pred_abs_all - gt_abs_all), axis=0)
    rel_l1_per_dim = np.mean(np.abs(pred_rel_all - gt_rel_all), axis=0)
    print("\n" + "=" * 60)
    print(f"Episode {args.episode_idx} | {ep_len} frames | {len(chunk_starts)} chunks")
    print("=" * 60)
    print("Per-dim mean L1 (absolute EE):")
    for d, (lbl, val) in enumerate(zip(ABS_DIM_LABELS, abs_l1_per_dim)):
        print(f"    dim {d:2d}  {lbl:8s}  {val:.4f}")
    print(f"  Overall abs EE L1: {abs_l1_per_dim.mean():.4f}")
    print("\nPer-dim mean L1 (relative EE):")
    for d, (lbl, val) in enumerate(zip(REL_DIM_LABELS, rel_l1_per_dim)):
        print(f"    dim {d:2d}  {lbl:8s}  {val:.4f}")
    print(f"  Overall rel EE L1: {rel_l1_per_dim.mean():.4f}")
    print(f"\nPlots saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
