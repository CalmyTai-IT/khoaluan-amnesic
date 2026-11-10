"""Gộp kết quả P2 + đánh giá G2 theo quy tắc viết trước (src/khoaluan/analysis/p2_report.py).
    python scripts/analyze_p2.py --in-root <thư mục chứa các run (giải nén p2_small_S*.zip)> --out p2_report
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from khoaluan.analysis.p2_report import build_report  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out", default="p2_report")
    a = ap.parse_args()
    text, _, _ = build_report(a.in_root, a.out)
    print(text)
