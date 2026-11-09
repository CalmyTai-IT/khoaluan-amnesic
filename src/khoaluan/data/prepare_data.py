"""
prepare_data.py — S1a/S1b + A-06: dựng các tập dữ liệu, bảng 2x2, mẫu kiểm tay.
Tương thích: SPEC_DULIEU_KHAINIEM.md v1.1 (§2, §3.5, §4, §5) và SPEC_REPRODUCE_VDL.md v1.0 (§3).

Chạy trên Kaggle (CPU là đủ, BẬT Internet):
    python scripts/prepare_data.py --out /kaggle/working/data --data-seed 1234

Đầu ra (mọi bản ghi có `uid` = "<nguồn>:<id gốc>", vì HANS-train và HANS-eval dùng TRÙNG pairID ex0..ex29999):
    hans_eval.jsonl, hans_train_A.jsonl (8K, fine-tune M-MIX), hans_train_B.jsonl (22K, fit khái niệm)
    mnli_bin_train_all.parquet  (MNLI train nhị phân + đặc trưng + ft_rank + in_pool)
    mnli_bin_dev_m.parquet, mnli_bin_dev_mm.parquet
    ft_uids_{25,100,1000,8000,16000}.txt (lồng nhau), mix_uids.txt
    handcheck_200_blind.csv + handcheck_200_key.csv  (A-14; file blind KHÔNG có nhãn máy)
    manifest.json   (đầy đủ, máy đọc)
    A06_summary.txt (tóm tắt để dán lại cho Claude / vào root §6)
Changelog:
  v1.1 (05/11/2026): uid có tiền tố nguồn; kiểm khớp parse–văn bản (MNLI); tỉ lệ parse lỗi; vân tay dữ liệu;
                     SMD trước ghép; mẫu kiểm tay A-14; bỏ cột *_binary_parse; A06_summary.txt.
  v1.0 (04/11/2026): bản đầu.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd

from khoaluan.data.feats import (c4, covariates, hans_subject_number, hyp_len_bin, is_con_anynode,
                                 norm_str, parse_alignment)

HANS_COMMIT = "7299f6f657089ce06a0f98e7e81f8d0f5b7741ce"  # tommccoy1/hans, kiểm 04/11/2026
HANS_SHA256 = {
    "heuristics_evaluation_set.jsonl": "15f2dbc5439d15138d8e7af6a10b202a421a5e0632e7fd4e0126b4295a6bafce",
    "heuristics_train_set.jsonl": "440040b6790c6359b56496375b87e6cedfa13a20f52bc3223dad31bd28b77faf",
}
FT_SIZES = [25, 100, 1000, 8000, 16000]
RARE_CELL_MIN = 300  # SPEC_DULIEU §6.1 bước 8
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(values):
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ----------------------------------------------------------------- HANS
def load_hans(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for fn, sha in HANS_SHA256.items():
        path = os.path.join(cache_dir, fn)
        if not os.path.exists(path):
            url = f"https://raw.githubusercontent.com/tommccoy1/hans/{HANS_COMMIT}/{fn}"
            log(f"tải {url}")
            urllib.request.urlretrieve(url, path)
        got = sha256_file(path)
        if got != sha:
            sys.exit(f"[DỪNG] SHA256 lệch cho {fn}: {got} (mong đợi {sha})")
        out[fn] = pd.read_json(path, lines=True)
    return out["heuristics_evaluation_set.jsonl"], out["heuristics_train_set.jsonl"]


def add_hans_cols(df, source):
    df = df.copy()
    df["uid"] = source + ":" + df["pairID"].astype(str)
    df["y"] = (df["gold_label"] != "entailment").astype(int)  # 0 = E, 1 = NE
    df["c4"] = [c4(p, h, pp) for p, h, pp in zip(df.sentence1, df.sentence2, df.sentence1_parse)]
    exp = df["heuristic"].map({"lexical_overlap": "LO_only", "subsequence": "SUB_only", "constituent": "CON"})
    agree = float((df["c4"] == exp).mean())
    df["subj_num"] = [hans_subject_number(x) for x in df.sentence2_parse]
    df = df.drop(columns=[c for c in df.columns if c.endswith("binary_parse")])
    return df, agree


def split_hans_train(ht, n_a, seed):
    """HT-A: n_a ví dụ cân bằng 30 subcase (fine-tune M-MIX). HT-B: phần còn lại (fit khái niệm)."""
    rng = np.random.default_rng(seed)
    subcases = sorted(ht.subcase.unique())
    base, extra = divmod(n_a, len(subcases))
    extra_sc = set(rng.choice(subcases, size=extra, replace=False))
    pick = []
    for sc in subcases:
        idx = ht.index[ht.subcase == sc].to_numpy()
        k = base + (1 if sc in extra_sc else 0)
        pick += list(rng.choice(idx, size=k, replace=False))
    a = ht.loc[sorted(pick)]
    return a, ht.drop(index=a.index)


# ----------------------------------------------------------------- MNLI
def load_mnli(revision=None):
    import datasets
    from datasets import load_dataset
    kw = {"revision": revision} if revision else {}
    ds = load_dataset("nyu-mll/multi_nli", **kw)
    meta = {"datasets_version": datasets.__version__, "hf_revision_arg": revision,
            "raw_split_sizes": {s: len(ds[s]) for s in ds}}
    out = {}
    for split in ["train", "validation_matched", "validation_mismatched"]:
        df = ds[split].to_pandas()
        meta[f"fingerprint_raw_{split}"] = fingerprint(zip(df.pairID, df.label))
        df = df[df.label.isin([0, 2])].copy()           # HF: 0 E, 1 N, 2 C, -1 không nhãn
        df["y"] = (df.label == 2).astype(int)           # 0 = E, 1 = C (non-entailment)
        df = df.drop(columns=[c for c in df.columns if c.endswith("binary_parse")])
        out[split] = df.reset_index(drop=True)
    return out, meta


def add_mnli_cols(df, source):
    df = df.copy()
    df["uid"] = source + ":" + df["pairID"].astype(str)
    df["c4"] = [c4(p, h, pp) for p, h, pp in zip(df.premise, df.hypothesis, df.premise_parse)]
    df["con_anynode"] = [is_con_anynode(pp, h) for pp, h in zip(df.premise_parse, df.hypothesis)]
    df["parse_aligned"] = [parse_alignment(p, pp) for p, pp in zip(df.premise, df.premise_parse)]
    cov = pd.DataFrame([covariates(p, h) for p, h in zip(df.premise, df.hypothesis)])
    df = pd.concat([df.reset_index(drop=True), cov], axis=1)
    df["lo"] = (df.c4 != "NONE").astype(int)
    df["hyp_len_bin"] = df.hyp_len.map(hyp_len_bin)
    return df


def table_2x2(df, col):
    t = Counter(zip(df[col], df["y"]))
    return {f"{col}={c}|y={'E' if y == 0 else 'C'}": int(n) for (c, y), n in sorted(t.items(), key=str)}


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else np.nan
    return float((a.mean() - b.mean()) / s) if s and s > 0 else float("nan")


def smd_table(pool):
    out = {}
    for y, name in [(0, "E"), (1, "C")]:
        sub = pool[pool.y == y]
        for v in ["hyp_len", "prem_len", "neg_p", "neg_h"]:
            out[f"y={name}|{v}"] = round(smd(sub.loc[sub.lo == 1, v], sub.loc[sub.lo == 0, v]), 3)
    return out


def handcheck_sample(pool, out_dir, seed=7):
    """SPEC_DULIEU §3.5: 50 LO_only, 50 SUB_only, 50 CON, 50 NONE có overlap_rate >= 0.8."""
    rng = np.random.default_rng(seed)
    groups = {
        "LO_only": pool[pool.c4 == "LO_only"],
        "SUB_only": pool[pool.c4 == "SUB_only"],
        "CON": pool[pool.c4 == "CON"],
        "NONE_hi": pool[(pool.c4 == "NONE") & (pool.overlap_rate >= 0.8)],
    }
    parts = []
    for g, df in groups.items():
        k = min(50, len(df))
        if k:
            parts.append(df.iloc[rng.choice(len(df), size=k, replace=False)].assign(machine=g))
    s = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    s["hc_id"] = [f"hc{i:03d}" for i in range(len(s))]
    blind = s[["hc_id", "premise", "hypothesis"]].copy()
    for col in ["annot_LO(0/1)", "annot_SUB(0/1)", "annot_CON(0/1)", "ghi_chu"]:
        blind[col] = ""
    blind.to_csv(os.path.join(out_dir, "handcheck_200_blind.csv"), index=False)
    s[["hc_id", "uid", "machine", "premise_parse"]].to_csv(os.path.join(out_dir, "handcheck_200_key.csv"), index=False)
    return {g: int(min(50, len(df))) for g, df in groups.items()}


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/data")
    ap.add_argument("--data-seed", type=int, default=1234)
    ap.add_argument("--hans-a", type=int, default=8000)
    ap.add_argument("--hf-revision", default=None, help="ghim commit của nyu-mll/multi_nli (tuỳ chọn)")
    ap.add_argument("--skip-mnli", action="store_true", help="chỉ xử lý HANS (kiểm thử offline)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    man = {"script_version": "1.1", "data_seed": args.data_seed, "hans_commit": HANS_COMMIT,
           "python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__}

    # ---- HANS
    log("HANS: tải + kiểm SHA256")
    he, ht = load_hans(os.path.join(args.out, "_raw_hans"))
    he, agree_e = add_hans_cols(he, "hans_eval")
    ht, agree_t = add_hans_cols(ht, "hans_train")
    man["hans_extractor_agreement"] = {"eval": agree_e, "train": agree_t}
    if not (agree_e == 1.0 and agree_t == 1.0):
        sys.exit(f"[DỪNG] bộ trích không khớp 100% HANS: {agree_e}, {agree_t} — xem SPEC_DULIEU §3.3")
    pairs_e = {(norm_str(p), norm_str(h)) for p, h in zip(he.sentence1, he.sentence2)}
    man["hans_train_eval_pair_overlap"] = int(sum((norm_str(p), norm_str(h)) in pairs_e
                                                  for p, h in zip(ht.sentence1, ht.sentence2)))
    ha, hb = split_hans_train(ht, args.hans_a, args.data_seed)
    he.to_json(os.path.join(args.out, "hans_eval.jsonl"), orient="records", lines=True)
    ha.to_json(os.path.join(args.out, "hans_train_A.jsonl"), orient="records", lines=True)
    hb.to_json(os.path.join(args.out, "hans_train_B.jsonl"), orient="records", lines=True)
    man["hans_counts"] = {
        "eval": len(he), "train_A": len(ha), "train_B": len(hb),
        "train_A_per_subcase_min_max": [int(ha.subcase.value_counts().min()), int(ha.subcase.value_counts().max())],
        "train_B_by_heuristic_label": {f"{k[0]}|{k[1]}": int(v) for k, v in Counter(zip(hb.heuristic, hb.gold_label)).items()},
        "train_B_subj_num": {str(k): int(v) for k, v in Counter(hb.subj_num).items()},
    }
    man["fingerprint_hans_A"] = fingerprint(ha.uid)
    log(f"HANS xong: eval={len(he)} A={len(ha)} B={len(hb)}")

    if not args.skip_mnli:
        # ---- MNLI
        log("MNLI: tải từ HF nyu-mll/multi_nli (cần Internet)")
        m, meta = load_mnli(args.hf_revision)
        man["mnli_source"] = meta
        log("MNLI: tính đặc trưng (train ~262K dòng nhị phân; vài phút)")
        tr = add_mnli_cols(m["train"], "mnli_train")
        dm = add_mnli_cols(m["validation_matched"], "mnli_dev_m")
        dmm = add_mnli_cols(m["validation_mismatched"], "mnli_dev_mm")
        if not tr.uid.is_unique:
            dup = int(tr.uid.duplicated().sum())
            sys.exit(f"[DỪNG] pairID MNLI train không duy nhất ({dup} trùng) — báo lại cho Claude")
        rng = np.random.default_rng(args.data_seed)
        tr["ft_rank"] = rng.permutation(len(tr))       # tập con lồng nhau: ft_rank < N
        for n in FT_SIZES:
            with open(os.path.join(args.out, f"ft_uids_{n}.txt"), "w") as f:
                f.write("\n".join(tr.loc[tr.ft_rank < n].sort_values("ft_rank").uid))
        with open(os.path.join(args.out, "mix_uids.txt"), "w") as f:
            f.write("\n".join(list(tr.loc[tr.ft_rank < 8000].sort_values("ft_rank").uid) + list(ha.uid)))
        tr["in_pool"] = (tr.ft_rank >= max(FT_SIZES)).astype(int)  # MNLI-POOL: không mô hình nào thấy khi train
        tr.to_parquet(os.path.join(args.out, "mnli_bin_train_all.parquet"))
        dm.to_parquet(os.path.join(args.out, "mnli_bin_dev_m.parquet"))
        dmm.to_parquet(os.path.join(args.out, "mnli_bin_dev_mm.parquet"))
        pool = tr[tr.in_pool == 1]

        man["mnli_counts"] = {
            "train_binary": len(tr), "train_label(0=E,1=C)": {str(k): int(v) for k, v in Counter(tr.y).items()},
            "dev_m_binary": len(dm), "dev_mm_binary": len(dmm), "pool": len(pool),
            "ft_label_counts": {str(n): {str(k): int(v) for k, v in Counter(tr.loc[tr.ft_rank < n, "y"]).items()}
                                for n in FT_SIZES},
        }
        man["fingerprint_ft16k"] = fingerprint(tr.loc[tr.ft_rank < 16000].sort_values("ft_rank").uid)
        # chất lượng parse (rủi ro R-11)
        man["mnli_parse_quality"] = {
            "train_parse_aligned_rate": round(float(tr.parse_aligned.mean()), 4),
            "dev_m_parse_aligned_rate": round(float(dm.parse_aligned.mean()), 4),
            "train_SUBorCON_parse_aligned_rate": round(float(tr.loc[tr.c4.isin(["SUB_only", "CON"]), "parse_aligned"].mean()), 4),
            "con_anynode_but_not_clause_train": int((tr.con_anynode & tr.c4.eq("SUB_only")).sum()),
        }
        # A-06: bảng 2x2
        man["A06_2x2_pool"] = {"lo": table_2x2(pool, "lo"), "c4": table_2x2(pool, "c4")}
        man["A06_2x2_train_all"] = {"lo": table_2x2(tr, "lo"), "c4": table_2x2(tr, "c4")}
        man["A06_2x2_dev_m"] = {"lo": table_2x2(dm, "lo"), "c4": table_2x2(dm, "c4")}
        cov = pool.groupby(["lo", "y"])[["hyp_len", "prem_len", "neg_p", "neg_h"]].mean().round(3)
        man["A06_covariates_pool"] = {f"lo={k[0]}|y={k[1]}": v for k, v in cov.to_dict(orient="index").items()}
        man["A06_negdrop_rate_pool"] = {f"lo={k[0]}|y={k[1]}": round(float(v), 3) for k, v in
                                        pool.assign(nd=(pool.neg_pattern == "P1H0")).groupby(["lo", "y"]).nd.mean().items()}
        man["A06_SMD_before_matching_pool"] = smd_table(pool)
        n_rare = int(((pool.lo == 1) & (pool.y == 1)).sum())
        n_fit = int(round(n_rare * 0.6))  # 40% ô LO+ để dành cho MNLI-HC-EVAL (SPEC_DULIEU §6.1 bước 2)
        man["A06_rare_cell"] = {"pool_LO+_C": n_rare, "available_for_fit_after_HC_holdout": n_fit,
                                "threshold": RARE_CELL_MIN}
        man["A06_decision"] = ("NHÁNH CHÍNH: c_LO dùng 2x2 + ghép hiệp biến" if n_fit >= RARE_CELL_MIN else
                               "DỰ PHÒNG: ô hiếm < 300 sau khi giữ lại MNLI-HC -> kích hoạt F1/F2 (SPEC_DULIEU §6.1)")
        man["A14_handcheck_counts"] = handcheck_sample(pool, args.out)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    write_summary(man, os.path.join(args.out, "A06_summary.txt"))
    log("XONG. Gửi lại nội dung A06_summary.txt (và manifest.json nếu được).")


def write_summary(man, path):
    keys = ["script_version", "data_seed", "hans_extractor_agreement", "hans_train_eval_pair_overlap", "hans_counts",
            "mnli_source", "mnli_counts", "fingerprint_ft16k", "fingerprint_hans_A", "mnli_parse_quality",
            "A06_2x2_pool", "A06_2x2_dev_m", "A06_covariates_pool", "A06_negdrop_rate_pool",
            "A06_SMD_before_matching_pool", "A06_rare_cell", "A06_decision", "A14_handcheck_counts"]
    with open(path, "w") as f:
        for k in keys:
            if k in man:
                f.write(f"## {k}\n{json.dumps(man[k], ensure_ascii=False)}\n")
    print(open(path).read())


if __name__ == "__main__":
    main()
