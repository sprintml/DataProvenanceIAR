"""Minimal stand-in for the VAR repo's distributed utility.

The vendored VAR architecture only calls ``get_device`` / ``initialized`` at
inference time; the full multi-GPU helper is unnecessary here.
"""
import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_world_size() -> int:
    return torch.distributed.get_world_size() if initialized() else 1


def get_rank() -> int:
    return torch.distributed.get_rank() if initialized() else 0


def is_master() -> bool:
    return get_rank() == 0
