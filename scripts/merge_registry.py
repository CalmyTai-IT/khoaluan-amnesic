"""Gộp run_record.json vào results/registry.csv.
    python scripts/merge_registry.py path/to/run_record.json [...]
Sau đó: git add results/registry.csv && git commit (người còn lại review)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from khoaluan.utils.registry import merge  # noqa: E402

if __name__ == "__main__":
    reg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "registry.csv")
    print("thêm", merge(sys.argv[1:], reg), "dòng vào", os.path.normpath(reg))
