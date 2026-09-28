"""Distributed helpers ported from ms-swift's swift.utils.torch_utils.

Provides `safe_ddp_context`, a context manager that serializes cache-producing
operations across ranks: global master runs first, then each node's local
master, then all remaining ranks. Falls back to a file lock when DDP is not
initialized.

The implementation follows ms-swift exactly; only the `get_cache_dir` helper
is reimplemented locally so this module does not depend on modelscope.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from typing import Optional

import torch.distributed as dist
from datasets.utils.filelock import FileLock


def get_dist_setting() -> tuple[int, int, int, int]:
    """Return rank, local_rank, world_size, local_world_size from launcher env."""
    rank = int(os.getenv("RANK", "-1"))
    local_rank = int(os.getenv("LOCAL_RANK", "-1"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", os.getenv("LOCAL_SIZE", "1")))
    return rank, local_rank, world_size, local_world_size


def is_dist() -> bool:
    rank, local_rank, _, _ = get_dist_setting()
    return rank >= 0 and local_rank >= 0


def is_master() -> bool:
    rank, _, _, _ = get_dist_setting()
    return rank in {-1, 0}


def is_local_master() -> bool:
    _, local_rank, _, _ = get_dist_setting()
    return local_rank in {-1, 0}


def get_cache_dir() -> str:
    """Cache directory for non-DDP file locks. Mirrors modelscope's default."""
    cache_dir = os.getenv("UNIFOLM_WLA_CACHE_DIR") or os.path.expanduser("~/.cache/unifolm_wla")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


_DISABLE_USE_BARRIER = False


@contextmanager
def disable_safe_ddp_context_use_barrier():
    global _DISABLE_USE_BARRIER
    _DISABLE_USE_BARRIER = True
    try:
        yield
    finally:
        _DISABLE_USE_BARRIER = False


@contextmanager
def safe_ddp_context(hash_id: Optional[str], use_barrier: bool = True):
    """Serialize a cache-producing block across ranks.

    When DDP is initialized and use_barrier is True, runs the block first on
    global master, then on each node's local master, then on remaining ranks.
    When DDP is not initialized but hash_id is provided, falls back to a
    fcntl-based file lock keyed by hash_id.
    """
    if _DISABLE_USE_BARRIER:
        use_barrier = False
    if use_barrier and dist.is_initialized():
        if is_dist():
            if not is_master():
                dist.barrier()
            if not is_local_master():
                # Compatible with multi-machine scenarios,
                # where each machine uses different storage hardware.
                dist.barrier()
        yield
        if is_dist():
            if is_master():
                dist.barrier()
            if is_local_master():
                dist.barrier()
    elif hash_id is not None:
        lock_dir = os.path.join(get_cache_dir(), 'lockers')
        os.makedirs(lock_dir, exist_ok=True)
        file_path = hashlib.sha256(hash_id.encode('utf-8')).hexdigest() + '.lock'
        file_path = os.path.join(lock_dir, file_path)
        with FileLock(file_path):
            yield
    else:
        yield
