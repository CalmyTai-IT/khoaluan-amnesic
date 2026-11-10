"""Wrapper: CUDA_VISIBLE_DEVICES=0 python scripts/run_queue.py --data-root ... --out-root ... --specs mnliB-16000-11 ...
Xem src/khoaluan/p2_queue.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from khoaluan.p2_queue import main  # noqa: E402

if __name__ == "__main__":
    main()
