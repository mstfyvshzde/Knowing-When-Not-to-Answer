"""
To make random operations produce the same results each time, so AI/ML experiments are more reproducible.
"""


import os # os is a Python module for interacting with the operating system.
import random

import numpy as np



def set_seed(
    seed: int,
    deterministic: bool = True # This parameter decides whether PyTorch should use more deterministic (repeatable) operations. True -> try to produce the same results every run. False -> allow faster algorithms, but results may vary slightly between runs.
) -> None:
    if seed < 0:
        raise ValueError("Seed must be a non-negative integer.")

    # It sets Python’s hash seed as an environment variable, helping hash-based operations behave more consistently across runs. str(seed) is needed because environment-variable values must be strings.
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    try: 
        import torch
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # These two lines control how PyTorch uses cuDNN on an NVIDIA GPU:
        if deterministic:
            torch.backends.cudnn.deterministic = True # “Prefer algorithms that produce the same result every time.”
            torch.backends.cudnn.benchmark = False # Normally, cuDNN may test several algorithms and choose the fastest one. The chosen algorithm can vary depending on the input or hardware.

            try:
                # "Use only deterministic algorithms whenever possible."
                torch.use_deterministic_algorithms(
                    True,
                    warn_only=True
                )

            except AttributeError:
                pass

    except ImportError:
        pass


if __name__ == "__main__":
    set_seed(
        seed=17,
        deterministic=True,
    )

    print("Python random:", random.random())
    print("NumPy random:", np.random.random())
    print("Random seed set to 17.")
