"""
Random seed utilities for reproducible experiments.

Bu dosyanın görevi:
- Python'ın rastgelelik sistemini sabitlemek
- NumPy rastgeleliğini sabitlemek
- PyTorch rastgeleliğini sabitlemek
- Deneylerin tekrar üretilebilirliğini artırmak
"""

import os
import random

import numpy as np


def set_seed(
    seed: int,
    deterministic: bool = True,
) -> None:
    """
    Set random seeds for supported libraries.

    Args:
        seed:
            Kullanılacak sabit rastgelelik değeri.

        deterministic:
            True olduğunda PyTorch işlemlerini mümkün olduğunca
            deterministic(aynı girdi verildiğinde her zaman aynı çıktıyı üreten sistem veya işlemd) hâle getirir.

    Örnek:
        set_seed(17)

    Aynı seed değeri, rastgele işlemlerin aynı sırayı
    üretmesine yardımcı olur.
    """

    # Negatif seed kullanmayı engelliyoruz.
    if seed < 0:
        raise ValueError(
            "Seed must be a non-negative integer."
        )

    # Python hash işlemlerinin daha tutarlı olması için (Hash işlemleri, bir veriyi sabit uzunlukta bir değere dönüştürür.)
    # environment variable ayarlıyoruz.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python'ın yerleşik random modülünü sabitliyoruz.
    random.seed(seed)

    # NumPy rastgele sayı üreticisini sabitliyoruz.
    np.random.seed(seed)

    try:
        # PyTorch yüklüyse onun seed ayarlarını da yapıyoruz.
        import torch

        # CPU üzerinde çalışan PyTorch işlemleri için seed.
        torch.manual_seed(seed)

        # CUDA destekli NVIDIA GPU varsa GPU seed ayarları.
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Aynı işlemin mümkün olduğunca aynı sonucu
            # üretmesini sağlamaya çalışır.
            torch.backends.cudnn.deterministic = True

            # PyTorch'un hız için farklı algoritmalar seçmesini
            # kapatıyoruz. Bu bazen sonucu değiştirebilir.
            torch.backends.cudnn.benchmark = False

            try:
                # Deterministic algoritmaları tercih eder.
                # warn_only=True sayesinde desteklenmeyen bir işlemde
                # program tamamen durmak yerine uyarı verebilir.
                torch.use_deterministic_algorithms(
                    True,
                    warn_only=True,
                )

            except AttributeError:
                # Eski PyTorch sürümlerinde bu fonksiyon
                # bulunmayabilir.
                pass

    except ImportError:
        # Projenin erken aşamasında PyTorch kurulu olmayabilir.
        # Bu durumda Python ve NumPy seed ayarları yine uygulanır.
        pass


if __name__ == "__main__":
    """
    Bu bölüm yalnızca dosya doğrudan çalıştırıldığında çalışır.

    Terminal komutu:

        python -m src.utils.seed
    """

    set_seed(
        seed=17,
        deterministic=True,
    )

    # Seed'in gerçekten aynı rastgele değerleri üretip
    # üretmediğini görmek için küçük bir test.
    print("Python random:", random.random())
    print("NumPy random:", np.random.random())
    print("Random seed set to 17.")