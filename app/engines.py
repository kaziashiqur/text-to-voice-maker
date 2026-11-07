"""Audio helpers shared by the voice services."""
import re

import numpy as np

SR = 24000  # output sample rate for the whole app

BN_CHAR = re.compile(r"[ঀ-৿]")


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if src == dst or len(x) == 0:
        return x
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
