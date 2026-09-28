import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_WORKSPACE_ROOT = str(Path(__file__).resolve().parents[1])
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES, STATE_SLICES, map_state  # noqa: E402
from unifolm_wla.dataloader.multi_source_dataset.se3_utils import (  # noqa: E402
    matrix_to_rot6d, matrix_to_rotvec, pose_to_se3, se3_inverse, rotvec_to_matrix,
)

_ACTIVE_SLOT_SPECS: List[Tuple[str, "str | Tuple[str, ...]", Optional[str], bool]] = [
    ("left_ee", "left_xyz_rotvec", "left_xyz_rot6d", True),
    ("left_gripper", "left_gripper", None, False),
    ("right_ee", "right_xyz_rotvec", "right_xyz_rot6d", True),
    ("right_gripper", "right_gripper", None, False),
    ("waist_joint", "waist_joint", None, False),
    ("base_command", ("base_vx_vy", "base_vw", "height"), None, False),
    ("left_leg_joint", "left_leg_joint", None, False),
    ("right_leg_joint", "right_leg_joint", None, False),
]


def _raw_width(raw_key: "str | Tuple[str, ...]") -> int:
    keys = raw_key if isinstance(raw_key, tuple) else (raw_key,)
    return sum(SLICES[k].stop - SLICES[k].start for k in keys)


def _gather_raw(unnorm: np.ndarray, raw_key: "str | Tuple[str, ...]") -> np.ndarray:
    """Concatenate one or more (possibly non-contiguous) SLICES columns of a
    (T, 54) unified action chunk into one (T, width) block, in `raw_key` order."""
    keys = raw_key if isinstance(raw_key, tuple) else (raw_key,)
    return np.concatenate([unnorm[:, SLICES[k]] for k in keys], axis=1)


def build_active_rot6d_layout() -> List[Tuple[str, int, int, bool]]:
    """Column layout of the active-rot6d intermediate representation.

    Returns [(name, c_start, c_end, is_rot9d), ...].
    """
    layout = []
    col = 0
    for name, raw_key, _state_key, is_rot9d in _ACTIVE_SLOT_SPECS:
        width = 9 if is_rot9d else _raw_width(raw_key)
        layout.append((name, col, col + width, is_rot9d))
        col += width
    return layout


ACTIVE_DIM = build_active_rot6d_layout()[-1][2]
_ACTIVE_LAYOUT = build_active_rot6d_layout()
_ACTIVE_LAYOUT_MAP = {name: (c0, c1) for name, c0, c1, _ in _ACTIVE_LAYOUT}


def build_unitree_fullbody_action_mask() -> np.ndarray:
    mask = np.zeros(SLICES["right_leg_joint"].stop, dtype=bool)
    for _name, raw_key, _state_key, _is_rot9d in _ACTIVE_SLOT_SPECS:
        for k in (raw_key if isinstance(raw_key, tuple) else (raw_key,)):
            mask[SLICES[k]] = True
    return mask


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(...,6) rot6d -> (...,3,3) via Gram-Schmidt (same convention as
    `se3_utils.matrix_to_rot6d`'s inverse)."""
    shape = rot6d.shape[:-1]
    r = rot6d.reshape(-1, 6)
    a1, a2 = r[:, 0:3], r[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=1)
    R = np.stack([b1, b2, b3], axis=-1)
    return R.reshape(*shape, 3, 3)


def _compose_ee9(anchor_9: np.ndarray, rel_6_chunk: np.ndarray) -> np.ndarray:
    """Compose a (T,6) xyz+rotvec relative action chunk onto a (9,) xyz+rot6d
    absolute state anchor (EE convention: STATE_SLICES["*_xyz_rot6d"]).

    Returns (T,9) absolute xyz+rot6d.
    """
    xyz_curr = anchor_9[:3]
    R_curr = _rot6d_to_matrix(anchor_9[3:9].reshape(1, 6))[0]
    T_curr = pose_to_se3(xyz_curr, R_curr)  # (4,4)

    R_rel = rotvec_to_matrix(rel_6_chunk[:, 3:6])  # (T,3,3)
    T_rel = pose_to_se3(rel_6_chunk[:, :3], R_rel)  # (T,4,4)
    T_abs = T_curr @ T_rel  # (T,4,4) broadcast

    xyz_abs = T_abs[:, :3, 3]
    rot6d_abs = matrix_to_rot6d(T_abs[:, :3, :3])
    return np.concatenate([xyz_abs, rot6d_abs], axis=1).astype(np.float32)  # (T,9)


def _compose_base9(anchor_6: np.ndarray, rel_6_chunk: np.ndarray) -> np.ndarray:
    """Same as `_compose_ee9`, but the state anchor is xyz+rotvec (6,) — unlike
    the EE slots, `map_state` stores the base pose as xyz+rotvec (via
    `pose_to_xyz_rotvec_from_format(..., "xyz_quat")`), NOT xyz+rot6d.

    Returns (T,9) absolute xyz+rot6d.
    """
    xyz_curr = anchor_6[:3]
    R_curr = rotvec_to_matrix(anchor_6[3:6].reshape(1, 3))[0]
    T_curr = pose_to_se3(xyz_curr, R_curr)

    R_rel = rotvec_to_matrix(rel_6_chunk[:, 3:6])
    T_rel = pose_to_se3(rel_6_chunk[:, :3], R_rel)
    T_abs = T_curr @ T_rel

    xyz_abs = T_abs[:, :3, 3]
    rot6d_abs = matrix_to_rot6d(T_abs[:, :3, :3])
    return np.concatenate([xyz_abs, rot6d_abs], axis=1).astype(np.float32)


def normalized_chunk_to_active_rot6d(
    pred_norm: np.ndarray,      # (T, 54) normalized unified action
    state_unnorm: np.ndarray,   # (60,) unnormalized unified state
    offset: np.ndarray,         # (54,) action norm offset
    scale: np.ndarray,          # (54,) action norm scale
) -> np.ndarray:
    """Unnormalize -> relative-to-absolute compose (EE/base) -> active-rot6d layout.

    Returns (T, ACTIVE_DIM) float32.
    """
    unnorm = (pred_norm * scale + offset).astype(np.float32)  # (T, 54)
    T = unnorm.shape[0]
    layout = _ACTIVE_LAYOUT
    out = np.zeros((T, ACTIVE_DIM), dtype=np.float32)

    for (name, raw_key, state_key, is_rot9d), (_, c0, c1, _) in zip(_ACTIVE_SLOT_SPECS, layout):
        chunk = _gather_raw(unnorm, raw_key)
        if not is_rot9d:
            out[:, c0:c1] = chunk
            continue
        anchor = state_unnorm[STATE_SLICES[state_key]]
        out[:, c0:c1] = _compose_base9(anchor, chunk) if name == "base" else _compose_ee9(anchor, chunk)
    return out


def active_rot6d_to_wbc50(active: np.ndarray) -> np.ndarray:
    layout = _ACTIVE_LAYOUT_MAP
    T = active.shape[0]
    out = np.zeros((T, 50), dtype=np.float32)

    def _ee9_to_xyzrpy(c0: int, c1: int) -> np.ndarray:
        blk = active[:, c0:c1]
        xyz = blk[:, :3]
        rpy = Rotation.from_matrix(_rot6d_to_matrix(blk[:, 3:9])).as_euler("xyz")
        return np.concatenate([xyz, rpy], axis=1)

    c0, c1 = layout["left_ee"]
    out[:, 0:6] = _ee9_to_xyzrpy(c0, c1)
    c0, c1 = layout["left_fig6d"]
    out[:, 6:12] = active[:, c0:c1]
    c0, c1 = layout["right_ee"]
    out[:, 12:18] = _ee9_to_xyzrpy(c0, c1)
    c0, c1 = layout["right_fig6d"]
    out[:, 18:24] = active[:, c0:c1]
    c0, c1 = layout["waist_joint"]
    out[:, 24:27] = active[:, c0:c1]
    # [27:31] left as zero padding — client does not read these dims.
    c0, c1 = layout["base"]
    blk = active[:, c0:c1]
    out[:, 31:34] = blk[:, :3]
    out[:, 34:38] = Rotation.from_matrix(_rot6d_to_matrix(blk[:, 3:9])).as_quat()  # xyzw
    c0, c1 = layout["left_leg_joint"]
    out[:, 38:44] = active[:, c0:c1]
    c0, c1 = layout["right_leg_joint"]
    out[:, 44:50] = active[:, c0:c1]
    return out

# ---------------------------------------------------------------------------
# Observation adapter: wbc client `state_dict` payload -> unified 60-dim state.
# ---------------------------------------------------------------------------

@dataclass
class _MinimalSourceConfig:
    """Just enough of `DatasetSourceConfig` for `map_state` (dual_with_legs branch)."""
    arm_type: str = "dual_with_legs"
    ee_format: str = "xyz_rpy"


_WBC_STATE_KEY_MAP: Dict[str, str] = {
    "observation.state.left_ee_pose_gripper_base": "left_ee_pose",
    "observation.state.right_ee_pose_gripper_base": "right_ee_pose",
    "observation.state.state_base_pose": "base_pose",
    "observation.state.left_fig6d": "left_fig6d",
    "observation.state.right_fig6d": "right_fig6d",
    "observation.state.waist_state_joint": "waist_joint",
    "observation.state.left_leg": "left_leg",
    "observation.state.right_leg": "right_leg",
}

WBC_IMAGE_ROLES = ["head_left", "cam_wrist_left", "cam_wrist_right"]


def build_state_dict_from_wbc_obs(payload: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Map the wbc client's `state_dict` payload to the unified 60-dim state.

    Returns (state_unnorm, state_mask), both (60,) — NOT normalized; caller is
    responsible for normalization via the checkpoint's own norm stats.
    """
    state_dict_raw = payload["state_dict"]
    mapped = {}
    for wbc_key, field_name in _WBC_STATE_KEY_MAP.items():
        if wbc_key in state_dict_raw:
            mapped[field_name] = np.asarray(state_dict_raw[wbc_key], dtype=np.float32)
    state_unnorm, state_mask = map_state(_MinimalSourceConfig(), mapped)
    return state_unnorm, state_mask
