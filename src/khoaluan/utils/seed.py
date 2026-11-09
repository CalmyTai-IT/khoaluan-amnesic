"""Cố định seed (SPEC_REPRODUCE_VDL §2.3). Gọi TRƯỚC khi tạo model/dataloader."""
import os
import random


def set_all_seeds(seed: int, deterministic: bool = True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        from transformers import set_seed
        set_seed(seed)
    except ImportError:
        pass
