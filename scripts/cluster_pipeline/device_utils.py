from __future__ import annotations


def select_device() -> str:
    """Select best available torch device in order: cuda -> mps -> cpu."""
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
