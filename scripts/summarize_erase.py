"""Gộp kết quả erasure từ nhiều phiên (ví dụ: phiên 1 với khoaluan-p2-s1 Version 1, phiên 2 với Version 2).
    python scripts/summarize_erase.py --in erase_v1 erase_v2 --out erase_all
Mỗi thư mục --in là một thư mục --out của run_erase_pilot.py (giải nén erase_small.zip). Tập fit phải giống hệt nhau."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd  # noqa: E402
from khoaluan.erase.pilot import summarize  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ins", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-null", type=int, default=1000)
    a = ap.parse_args()
    models, cells, reports = [], [], []
    for d in a.ins:
        reports.append(json.load(open(os.path.join(d, "sets", "sets_report.json"))))
        for rid in sorted(os.listdir(d)):
            f = os.path.join(d, rid, "erase_model.json")
            if os.path.exists(f):
                models.append(json.load(open(f)))
                cells.append(pd.read_csv(os.path.join(d, rid, "erase_cells.csv")))
    if any(json.dumps(r["mnli"]["c_LO_total"]) != json.dumps(reports[0]["mnli"]["c_LO_total"]) for r in reports):
        sys.exit("[DỪNG] tập fit khác nhau giữa các phiên")
    os.makedirs(a.out, exist_ok=True)
    cells = pd.concat(cells, ignore_index=True)
    cells.to_csv(os.path.join(a.out, "erase_cells_all.csv"), index=False)
    text = summarize(models, cells, reports[0], a.n_null, "leace")
    open(os.path.join(a.out, "erase_summary.md"), "w").write(text)
    print(text)
