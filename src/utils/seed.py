"""
Bu dosya, rastgelelik kullanan işlemlerin her çalıştırmada mümkün olduğunca aynı sonucu vermesini sağlar.
"""

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except AttributeError:
                pass

    except ImportError:
        pass


if __name__ == "__main__":
    set_seed(seed=42)
    print("random seed set to 17")