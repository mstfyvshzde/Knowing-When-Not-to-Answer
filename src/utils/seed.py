"""
Set reproducibility controls for random operations used by the project.

A random seed (rastgelelik tohumu) makes pseudo-random operations produce the
same sequence when an experiment is repeated with the same software and setup.

This function seeds Python, NumPy, and PyTorch. When deterministic=True, it
also asks PyTorch to prefer deterministic operations where possible.

Deterministic execution improves reproducibility (tekrarlanabilirlik), although
some GPU operations may still depend on hardware, library versions, or
operations for which no fully deterministic implementation exists.
"""

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed the project's main random-number generators.

    deterministic=True asks PyTorch to prefer deterministic algorithms
    (aynı girdide mümkün olduğunca aynı sonucu üreten işlemler). This can
    reduce performance slightly but makes experimental runs easier to reproduce.
    """

    if seed < 0:
        raise ValueError("Seed must be a non-negative integer.")

    # Record the desired Python hash seed in the environment.
    # Important: PYTHONHASHSEED is normally read when the Python interpreter
    # starts, so setting it here does not retroactively change hash randomization
    # in the current process. It can still be inherited by child Python processes.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Seed Python's built-in random generator.
    random.seed(seed)

    # Seed NumPy's global random generator.
    np.random.seed(seed)

    # PyTorch is optional for this utility. If it is installed, seed both CPU
    # and available CUDA generators so model-related randomness is repeatable.
    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
        # cuDNN (NVIDIA'nın GPU neural-network kütüphanesi) may choose among
        # several implementations of the same operation. Deterministic mode
        # prefers repeatable implementations instead of benchmarking for speed.
            torch.backends.cudnn.deterministic = (
                True  # “Prefer algorithms that produce the same result every time.”
            )
            torch.backends.cudnn.benchmark = False  # Normally, cuDNN may test several algorithms and choose the fastest one. The chosen algorithm can vary depending on the input or hardware.

            try:
            # Ask PyTorch to use deterministic implementations when available.
            # warn_only=True reports nondeterministic operations instead of stopping
            # the entire experiment.
                torch.use_deterministic_algorithms(True, warn_only=True)

            except AttributeError:
            # Older PyTorch versions may not provide this API.
                pass

    except ImportError:
    # Allow non-PyTorch utilities to use this module even if PyTorch is absent.
        pass
