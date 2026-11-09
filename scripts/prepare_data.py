"""Wrapper: chạy từ thư mục gốc repo.  python scripts/prepare_data.py --out /kaggle/working/data"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from khoaluan.data.prepare_data import main  # noqa: E402

if __name__ == "__main__":
    main()
