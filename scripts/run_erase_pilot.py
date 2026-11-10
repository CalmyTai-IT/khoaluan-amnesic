"""Wrapper: erasure tại L_out (pilot / bản đủ). Xem src/khoaluan/erase/pilot.py.
    CUDA_VISIBLE_DEVICES=0 python scripts/run_erase_pilot.py --data-root /kaggle/input --out /kaggle/working/erase \
        --run-ids bertL-mnliB-16k-s111 bertL-mnliB-16k-s22 bertL-mnliB-16k-s33"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from khoaluan.erase.pilot import main  # noqa: E402

if __name__ == "__main__":
    main()
