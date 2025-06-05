# atari/utils.py

import numpy as np

def discretize_frame(obs_np):
    """
    Convert a (4, 84, 84) uint8 frame-stack into a 7x7 grid of 11-level buckets.
    Steps:
      1. Average over the 4 stacked frames -> (84, 84) float array.
      2. Downsample by averaging each non-overlapping 12x12 block -> (7, 7).
      3. Quantize each block’s averaged value into one of 11 bins (0..10).
      4. Flatten to a tuple of length 7x7 to serve as a dictionary key.
    """
    # 1) Average over the 4 stacked frames → shape (84,84)
    gray = obs_np.astype(np.float32).mean(axis=0)  # → (84,84)

    # 2) Downsample 84×84 → 7×7 by averaging each non‐overlapping 12×12 block
    h, w = gray.shape
    assert h % 12 == 0 and w % 12 == 0, "Frame dimensions not divisible by 12 for 7x7 grid"
    downsampled = gray.reshape(7, 12, 7, 12).mean(axis=(1, 3))  # → shape (7,7)

    # 3) Quantize into 11 bins. Each downsampled pixel ∈ [0..255], so floor_divide by 24 → values 0..10
    # (Previous: floor_divide by 32 for 8 bins; floor_divide by 16 for 16 bins)
    quantized = np.floor_divide(downsampled, 24).astype(np.uint8) # Now 11 bins

    # 4) Flatten into a tuple of length 7×7 for use as a dictionary key
    return tuple(quantized.flatten().tolist())