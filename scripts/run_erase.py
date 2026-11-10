"""Erasure: L_out (stage lout) và [CLS] tầng giữa (stage lmid). Xem src/khoaluan/erase/pilot.py và layers.py.
    CUDA_VISIBLE_DEVICES=0 python scripts/run_erase.py --data-root /kaggle/input --runs-root /kaggle/input \
        --out /kaggle/working/erase/q0 --stages lout,lmid --run-ids bertL-mnliB-16k-s111 bertL-mnliB-16k-s33"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from khoaluan.erase.pilot import main  # noqa: E402

if __name__ == "__main__":
    main()
