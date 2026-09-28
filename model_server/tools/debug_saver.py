import logging
import os
import queue
import threading
import traceback
from typing import List, Optional, Sequence

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

_LEFT_COLOR = "steelblue"
_RIGHT_COLOR = "darkorange"
_PREFIX_COLOR = "crimson"
_PREFIX_SHADE = "mistyrose"


class StepDebugSaver:
    """Saves one composed (camera views + action-chunk plot) image per
    inference step, off the inference thread.

    `save_dir=None` makes every `save_step()` call a no-op, so callers never
    need to guard the call site with `if debug_saver is not None`.
    """

    def __init__(self, save_dir: Optional[str], queue_maxsize: int = 8):
        self._save_dir = save_dir
        if save_dir is None:
            self._queue = None
            self._thread = None
            return

        os.makedirs(save_dir, exist_ok=True)
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_maxsize)
        self._frame_idx = 0
        self._thread = threading.Thread(target=self._worker_loop, daemon=True,
                                         name="debug-saver")
        self._thread.start()
        logging.info("[DebugSaver] saving to %s", save_dir)

    @property
    def enabled(self) -> bool:
        return self._save_dir is not None

    def save_step(
        self,
        images: Sequence[np.ndarray],
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        prefix_len: int = 0,
        chunk_id: int = 0,
        state_unnorm: Optional[np.ndarray] = None,
        pred_norm: Optional[np.ndarray] = None,
        unnorm_key: Optional[str] = None,
    ) -> None:
        """`left_pose`/`right_pose` are (T,6) xyz+rpy -- the same layout as
        `action.left_ee_rpy`/`action.right_ee_rpy`.

        `state_unnorm` ((60,) unnormalized state anchor), `pred_norm`
        ((T,54) raw normalized model output) and `unnorm_key` are optional
        and, if given, are dumped to a companion `.npz` alongside the
        rendered `.jpg` -- enough to reconstruct the full wire action
        offline without re-running inference.
        """
        if not self.enabled:
            return
        frame_idx = self._frame_idx
        self._frame_idx += 1
        item = (
            frame_idx, list(images), np.asarray(left_pose), np.asarray(right_pose),
            int(prefix_len), int(chunk_id),
            None if state_unnorm is None else np.asarray(state_unnorm),
            None if pred_norm is None else np.asarray(pred_norm),
            unnorm_key,
        )
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logging.warning("[DebugSaver] queue full, dropping frame %d", frame_idx)

    def close(self) -> None:
        if not self.enabled:
            return
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    # ── worker thread ───────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._render_and_save(item)
            except Exception:
                logging.error("[DebugSaver] render/save failed:\n%s", traceback.format_exc())

    def _render_and_save(self, item) -> None:
        (frame_idx, images, left_pose, right_pose, prefix_len, chunk_id,
         state_unnorm, pred_norm, unnorm_key) = item

        cam_row = self._build_camera_row(images)
        plot_img = self._build_action_plot(left_pose, right_pose, prefix_len)

        target_h = max(cam_row.shape[0], plot_img.shape[0])
        cam_row = self._pad_to_height(cam_row, target_h)
        plot_img = self._pad_to_height(plot_img, target_h)
        composed = cv2.hconcat([cam_row, plot_img])
        self._stamp_label(composed, f"chunk_id={chunk_id} prefix={prefix_len}")

        path = os.path.join(self._save_dir, f"step_{frame_idx:06d}.jpg")
        cv2.imwrite(path, composed)

        self._dump_npz(frame_idx, images, left_pose, right_pose, prefix_len, chunk_id,
                        state_unnorm, pred_norm, unnorm_key)

    def _dump_npz(self, frame_idx, images, left_pose, right_pose, prefix_len, chunk_id,
                  state_unnorm, pred_norm, unnorm_key) -> None:
        data = {
            "images": self._stack_images(images),
            "left_pose": left_pose,
            "right_pose": right_pose,
            "prefix_len": prefix_len,
            "chunk_id": chunk_id,
        }
        if state_unnorm is not None:
            data["state_unnorm"] = state_unnorm
        if pred_norm is not None:
            data["pred_norm"] = pred_norm
        if unnorm_key is not None:
            data["unnorm_key"] = unnorm_key
        path = os.path.join(self._save_dir, f"step_{frame_idx:06d}.npz")
        np.savez(path, **data)

    @staticmethod
    def _stack_images(images: List[np.ndarray]) -> np.ndarray:
        """(V,H,W,3) if every view shares a resolution, else an object array of
        the per-view arrays -- camera views need not be uniform (e.g.
        `--image_size 0 0` disables the servers' resize-to-common-size step)."""
        if not images:
            return np.empty(0)
        arrs = [np.asarray(img) for img in images]
        if len({a.shape for a in arrs}) == 1:
            return np.stack(arrs)
        out = np.empty(len(arrs), dtype=object)
        out[:] = arrs
        return out

    @staticmethod
    def _build_camera_row(images: List[np.ndarray], target_h: int = 240) -> np.ndarray:
        """Concatenate the multi-view RGB frames into one row, converting back
        to BGR since that's what `cv2.imwrite` expects."""
        if not images:
            return np.full((target_h, 1, 3), 255, dtype=np.uint8)
        resized = []
        for img in images:
            img_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img_bgr.shape[:2]
            new_w = max(1, int(round(w * target_h / h)))
            resized.append(cv2.resize(img_bgr, (new_w, target_h), interpolation=cv2.INTER_LINEAR))
        return cv2.hconcat(resized)

    @staticmethod
    def _build_action_plot(left_pose: np.ndarray, right_pose: np.ndarray, prefix_len: int) -> np.ndarray:
        """(T,6) left/right EE xyz+rpy -> a 2x3 (x/y/z, roll/pitch/yaw vs step)
        plot, BGR uint8.

        The [0:prefix_len] head -- the RTC-pinned prefix carried over from the
        previous chunk -- is shaded and re-drawn in `crimson` on top of the
        normal per-arm color, so a pinned-prefix discontinuity is visible at a
        glance against the freshly generated tail.
        """
        T = left_pose.shape[0]
        steps = np.arange(T)
        prefix_len = max(0, min(prefix_len, T))
        dim_labels = ("x", "y", "z", "roll", "pitch", "yaw")

        fig = Figure(figsize=(6.0, 4.0), dpi=100)
        canvas = FigureCanvasAgg(fig)
        axes = fig.subplots(2, 3).ravel()
        for i, (ax, label) in enumerate(zip(axes, dim_labels)):
            if prefix_len > 0:
                ax.axvspan(-0.5, prefix_len - 0.5, color=_PREFIX_SHADE, alpha=0.5, zorder=0)
            ax.plot(steps, left_pose[:, i], color=_LEFT_COLOR, linewidth=1.1,
                    label="L" if i == 0 else None, zorder=2)
            ax.plot(steps, right_pose[:, i], color=_RIGHT_COLOR, linewidth=1.1, linestyle="--",
                    label="R" if i == 0 else None, zorder=2)
            if prefix_len > 0:
                ax.plot(steps[:prefix_len], left_pose[:prefix_len, i], color=_PREFIX_COLOR,
                        linewidth=1.6, zorder=4)
                ax.plot(steps[:prefix_len], right_pose[:prefix_len, i], color=_PREFIX_COLOR,
                        linewidth=1.6, linestyle="--", zorder=4)
            ax.set_title(label, fontsize=8)
            ax.tick_params(labelsize=6)
        axes[0].legend(fontsize=6, loc="best")
        fig.tight_layout()

        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        return cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)

    @staticmethod
    def _pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
        h = img.shape[0]
        if h == target_h:
            return img
        pad = target_h - h
        top, bottom = pad // 2, pad - pad // 2
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    @staticmethod
    def _stamp_label(img: np.ndarray, text: str) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.5, 1
        left, top = 5, 20
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(img, (left - 2, top - th - 4), (left + tw + 2, top + 4), (0, 0, 0), -1)
        cv2.putText(img, text, (left, top), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
