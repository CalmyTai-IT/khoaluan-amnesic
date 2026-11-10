"""Gộp kết quả erasure từ nhiều phiên / nhiều tiến trình (mỗi thư mục --in là một thư mục --out của run_erase.py).
    python scripts/summarize_erase.py --in erase/q0 erase/q1 --out erase/all
Tập fit phải giống hệt nhau (so danh sách uid của mọi tập đã lưu). Gộp cả L_out (erase_*) và L_ℓ (lmid_*) nếu có."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd  # noqa: E402
from khoaluan.erase.layers import summarize_layers  # noqa: E402
from khoaluan.erase.pilot import summarize  # noqa: E402


def sets_signature(d):
    sig = {}
    sd = os.path.join(d, "sets")
    for fn in sorted(os.listdir(sd)):
        if fn.startswith("set_") and fn.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(sd, fn))
            sig[fn] = (len(df), hash(tuple(df.uid)), hash(tuple(df.split)))
    return sig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ins", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-null", type=int, default=1000)
    a = ap.parse_args(argv)
    models, cells, lmodels, lcells, reports, sigs = [], [], [], [], [], []
    for d in a.ins:
        reports.append(json.load(open(os.path.join(d, "sets", "sets_report.json"))))
        sigs.append(sets_signature(d))
        for rid in sorted(os.listdir(d)):
            f = os.path.join(d, rid, "erase_model.json")
            if os.path.exists(f):
                models.append(json.load(open(f)))
                cells.append(pd.read_csv(os.path.join(d, rid, "erase_cells.csv")))
            f = os.path.join(d, rid, "lmid_model.json")
            if os.path.exists(f):
                lmodels.append(json.load(open(f)))
                lcells.append(pd.read_csv(os.path.join(d, rid, "lmid_cells.csv")))
    if any(s != sigs[0] for s in sigs):
        sys.exit("[DỪNG] tập fit khác nhau giữa các thư mục --in")
    ids = [m["run_id"] for m in models] + [m["run_id"] for m in lmodels]
    if len(set(m["run_id"] for m in models)) != len(models) or len(set(m["run_id"] for m in lmodels)) != len(lmodels):
        sys.exit(f"[DỪNG] một mô hình xuất hiện ở nhiều thư mục: {ids}")
    os.makedirs(a.out, exist_ok=True)
    texts = []
    if models:
        c = pd.concat(cells, ignore_index=True)
        c.to_csv(os.path.join(a.out, "erase_cells_all.csv"), index=False)
        t = summarize(models, c, reports[0], a.n_null, "leace")
        open(os.path.join(a.out, "erase_summary.md"), "w").write(t)
        texts.append(t)
    if lmodels:
        c = pd.concat(lcells, ignore_index=True)
        c.to_csv(os.path.join(a.out, "lmid_cells_all.csv"), index=False)
        t = summarize_layers(lmodels, c)
        open(os.path.join(a.out, "lmid_summary.md"), "w").write(t)
        texts.append(t)
    json.dump(reports[0], open(os.path.join(a.out, "sets_report.json"), "w"), indent=2, ensure_ascii=False)
    print("\n\n".join(texts))


if __name__ == "__main__":
    main()
