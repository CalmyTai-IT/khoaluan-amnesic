"""Wrapper: chạy từ thư mục gốc repo. Mỗi tiến trình chỉ được thấy 1 GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_p2.py --data-root /kaggle/input --out-root /kaggle/working/out \
        --data mnliB --size 16000 --seed 11 --account TV-B
Xem src/khoaluan/p2.py và SPEC_REPRODUCE_VDL v1.1."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from khoaluan.p2 import main  # noqa: E402

if __name__ == "__main__":
    main()
