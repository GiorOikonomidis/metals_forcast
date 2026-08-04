"""
Runtime environment: the detected compute device and process-wide setup
(RNG seeding, SSL relaxation). Detected, not configured — which is why DEVICE
lives here rather than in consts.py.
"""

import random
import ssl

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_runtime(seed: int) -> None:
    """
    Seed every RNG the run touches and relax SSL verification so the Chronos /
    HuggingFace downloads work behind an intercepting proxy. Call once, before
    anything builds a model or draws a number.
    """
    ssl._create_default_https_context = ssl._create_unverified_context

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
