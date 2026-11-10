"""
p2_report.py — gộp kết quả P2 và đánh giá gate G2 THEO QUY TẮC VIẾT TRƯỚC KHI CÓ KẾT QUẢ (SPEC_REPRODUCE_VDL v1.1 §7–§8).
Viết 10/11/2026, trước các phiên S1–S4. Không sửa ngưỡng sau khi thấy số liệu; mọi sửa phải ghi vào changelog của SPEC.

Đầu vào: thư mục chứa (ở đâu đó bên dưới) các thư mục run có p2_metrics.json (+ lr_summary.csv, ccs_summary.csv,
dr1_summary.csv, lr_aux_summary.csv, ccs_aux_summary.csv, hans_subcase.csv) — ví dụ giải nén các p2_small_S*.zip.
Đầu ra: P2_REPORT.md + p2_runs.csv + p2_models.csv.

Quy ước tầng: chỉ số của nhóm là hidden_states[ℓ] (0 = embedding). "Tầng 23" của V&dL ↔ hidden_states[23 + k],
k ∈ {−1, 0, +1} do D-R2 chọn: k có tương quan Spearman trung bình cao nhất giữa đường LR theo tầng của mô hình then chốt
(8K, 16K; MNLI và HANS) và Bảng 6/7. Tầng dùng cho G2 = 23 + k — chọn theo HÌNH DẠNG ĐƯỜNG, không theo kết quả G2.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

MODEL_ORDER = ["pt", "25", "100", "1k", "8k", "16k", "mix"]
# Bảng phụ lục V&dL (Findings ACL 2025). LR/CCS tầng 23, trung bình 5 lần: Bảng 2–5; M-PT LR theo văn bản §4.3.1.
REF_LR23 = {"pt": (0.59, 0.61), "25": (0.59, 0.67), "100": (0.58, 0.68), "1k": (0.71, 0.70),
            "8k": (0.73, 0.58), "16k": (0.86, 0.72), "mix": (0.83, 0.83)}
REF_CCS23 = {"25": (0.52, 0.53), "100": (0.53, 0.51), "1k": (0.69, 0.53), "8k": (0.70, 0.51),
             "16k": (0.65, 0.51), "mix": (0.84, 0.65)}
# Bảng 1 (MNLI, tầng 23): (LR, CCS) theo token
REF_T1 = {"8k": {"mean": (0.73, 0.50), "last": (0.84, 0.62), "cls": (0.73, 0.70)},
          "16k": {"mean": (0.86, 0.50), "last": (0.81, 0.52), "cls": (0.86, 0.65)}}
# Bảng 6 (8K) và 7 (16K): tầng -> (LR MNLI, LR HANS, CCS MNLI, CCS HANS); trích tự động từ PDF và đối chiếu bản in
T6 = {1: (0.49, 0.53, 0.52, 0.51), 2: (0.49, 0.54, 0.52, 0.51), 3: (0.49, 0.55, 0.53, 0.53), 4: (0.52, 0.53, 0.52, 0.53),
      5: (0.51, 0.54, 0.53, 0.52), 6: (0.51, 0.55, 0.51, 0.52), 7: (0.51, 0.57, 0.52, 0.52), 8: (0.51, 0.52, 0.54, 0.53),
      9: (0.52, 0.53, 0.55, 0.53), 10: (0.54, 0.55, 0.53, 0.52), 11: (0.56, 0.55, 0.51, 0.53),
      12: (0.56, 0.55, 0.51, 0.52), 13: (0.55, 0.58, 0.52, 0.52), 14: (0.52, 0.56, 0.51, 0.52),
      15: (0.58, 0.58, 0.52, 0.51), 16: (0.60, 0.57, 0.52, 0.51), 17: (0.58, 0.54, 0.52, 0.52),
      18: (0.56, 0.52, 0.52, 0.52), 19: (0.61, 0.55, 0.52, 0.51), 20: (0.68, 0.51, 0.56, 0.51),
      21: (0.76, 0.59, 0.70, 0.52), 22: (0.80, 0.61, 0.68, 0.53), 23: (0.73, 0.58, 0.70, 0.51),
      24: (0.80, 0.64, 0.66, 0.51)}
T7 = {1: (0.50, 0.54, 0.51, 0.52), 2: (0.50, 0.55, 0.51, 0.53), 3: (0.51, 0.55, 0.52, 0.52), 4: (0.50, 0.54, 0.52, 0.52),
      5: (0.50, 0.56, 0.52, 0.52), 6: (0.50, 0.55, 0.52, 0.51), 7: (0.51, 0.56, 0.52, 0.52), 8: (0.52, 0.55, 0.52, 0.52),
      9: (0.53, 0.59, 0.51, 0.51), 10: (0.57, 0.55, 0.52, 0.52), 11: (0.59, 0.57, 0.51, 0.52),
      12: (0.60, 0.57, 0.53, 0.51), 13: (0.61, 0.58, 0.53, 0.52), 14: (0.61, 0.57, 0.51, 0.52),
      15: (0.61, 0.57, 0.53, 0.51), 16: (0.57, 0.53, 0.52, 0.52), 17: (0.57, 0.54, 0.53, 0.52),
      18: (0.58, 0.53, 0.53, 0.52), 19: (0.61, 0.57, 0.52, 0.53), 20: (0.77, 0.53, 0.79, 0.51),
      21: (0.84, 0.57, 0.81, 0.52), 22: (0.85, 0.69, 0.59, 0.52), 23: (0.86, 0.72, 0.65, 0.51),
      24: (0.85, 0.67, 0.61, 0.52)}
# McCoy, Min & Linzen 2020 (100 BERT-base, tính lại từ CSV gốc — SPEC_DULIEU §4.4): SD của HANS-NE theo heuristic
MCCOY2020_SD_NE = {"LO": 0.1230, "SUB": 0.0240, "CON": 0.0404}
TOL = 0.05


def model_key(run_id):
    """bertL-mnliB-16k-s22 -> ('16k', 22); bertL-mix-16k-s11 -> ('mix', 11); bertL-pt -> ('pt', None)."""
    parts = run_id.split("-")
    if parts[1] == "pt":
        return "pt", None
    seed = int(parts[-1].lstrip("s")) if parts[-1].startswith("s") else None
    return ("mix" if parts[1] == "mix" else parts[2]), seed


def load_runs(root):
    """Đọc mọi run dưới root. Trả về list dict: run_id, model, seed, degenerate, is_key, metrics, lr, ccs, dr1, aux."""
    runs = []
    for mp in sorted(glob.glob(os.path.join(root, "**", "p2_metrics.json"), recursive=True)):
        d = os.path.dirname(mp)
        rec = json.load(open(mp))
        if "-rep" in rec["run_id"]:
            continue                                   # chạy lặp kiểm S0 xử lý riêng
        model, seed = model_key(rec["run_id"])

        def csv(name):
            p = os.path.join(d, name)
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                return None
            try:
                return pd.read_csv(p)
            except pd.errors.EmptyDataError:
                return None
        runs.append({"run_id": rec["run_id"], "model": model, "seed": seed, "dir": d,
                     "degenerate": bool(rec.get("degenerate")), "reason": rec.get("degenerate_reason"),
                     "is_key": bool(rec.get("is_key_model")), "metrics": rec.get("metrics") or {},
                     "train": rec.get("train") or {}, "lr": csv("lr_summary.csv"), "ccs": csv("ccs_summary.csv"),
                     "dr1": csv("dr1_summary.csv"), "lr_aux": csv("lr_aux_summary.csv"),
                     "ccs_aux": csv("ccs_aux_summary.csv")})
    return runs


def probe_at(df, layer, dataset, variant):
    if df is None:
        return np.nan
    s = df[(df.layer == layer) & (df.dataset == dataset) & (df.variant == variant)]
    return float(s.acc_mean.iloc[0]) if len(s) else np.nan


def curve(df, dataset, variant="faithful"):
    if df is None:
        return {}
    s = df[(df.dataset == dataset) & (df.variant == variant)]
    return {int(l): float(a) for l, a in zip(s.layer, s.acc_mean)}


def run_table(runs, layers=(22, 23, 24)):
    rows = []
    for r in runs:
        m = r["metrics"]
        row = {"run_id": r["run_id"], "model": r["model"], "seed": r["seed"], "degenerate": r["degenerate"],
               "reason": r["reason"], "is_key": r["is_key"], "devm": m.get("devm_acc"), "devmm": m.get("devmm_acc"),
               "hans_total": m.get("hans_acc_total"), "hans_E_all": m.get("hans_E_all"),
               "hans_NE_mean3": m.get("hans_NE_mean3"), "NE_LO": m.get("hans_NE_LO"), "NE_SUB": m.get("hans_NE_SUB"),
               "NE_CON": m.get("hans_NE_CON"), "train_minutes": r["train"].get("wall_minutes_train")}
        for l in layers:
            for ds, tag in (("dev_m", "MNLI"), ("hans_eval", "HANS")):
                row[f"LR{l}_{tag}"] = probe_at(r["lr"], l, ds, "faithful")
                row[f"LRrob{l}_{tag}"] = probe_at(r["lr"], l, ds, "robust")
                row[f"CCS{l}_{tag}"] = probe_at(r["ccs"], l, ds, "linear")
            row[f"gap{l}"] = row[f"LR{l}_MNLI"] - row[f"LR{l}_HANS"]
        rows.append(row)
    return pd.DataFrame(rows)


def model_table(rt):
    ok = rt[~rt.degenerate]
    num = [c for c in ok.columns if c not in ("run_id", "model", "seed", "degenerate", "reason", "is_key")]
    g = ok.groupby("model")[num]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    models = [m for m in MODEL_ORDER if m in set(rt.model)]      # giữ cả mô hình mà mọi seed đều thoái hoá
    out = out.reindex(models)
    out["n_ok"] = ok.groupby("model").size().reindex(models).fillna(0).astype(int)
    out["n_degenerate"] = rt[rt.degenerate].groupby("model").size().reindex(models).fillna(0).astype(int)
    return out


def _ranks(x):
    x = np.asarray(x, float)
    order = x.argsort()
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):                              # hạng trung bình cho giá trị trùng
        idx = x == v
        r[idx] = r[idx].mean()
    return r


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def dr2(runs):
    """D-R2: với k ∈ {−1,0,+1}, hàng ℓ của Bảng 6/7 ↔ hidden_states[ℓ + k]; Spearman trên các tầng chồng nhau."""
    out = {}
    keys = {r["model"]: r for r in runs if r["is_key"] and not r["degenerate"] and r["model"] in ("8k", "16k")}
    for k in (-1, 0, 1):
        cors = {}
        for model, table in (("8k", T6), ("16k", T7)):
            if model not in keys:
                continue
            for ds, col in (("dev_m", 0), ("hans_eval", 1)):
                c = curve(keys[model]["lr"], ds)
                rows = [l for l in table if (l + k) in c]
                if len(rows) >= 10:
                    cors[f"{model}_{ds}"] = spearman([table[l][col] for l in rows], [c[l + k] for l in rows])
        out[k] = {"per_curve": cors, "mean": float(np.nanmean(list(cors.values()))) if cors else np.nan}
    valid = {k: v["mean"] for k, v in out.items() if not np.isnan(v["mean"])}
    best = max(valid, key=valid.get) if valid else None
    return out, best


def evaluate_g2(mt, rt, runs, layer):
    """Q1–Q5 + mốc §7.2 + quy tắc G2 §7.4 ở tầng `layer` (chỉ số hidden_states của nhóm)."""
    def mm(model, col):
        return float(mt.loc[model, f"{col}_mean"]) if model in mt.index and f"{col}_mean" in mt.columns else np.nan
    gap = f"gap{layer}"
    q = {}
    q["Q1"] = {"gap_pt": mm("pt", gap), "gap_16k": mm("16k", gap)}
    q["Q1"]["pass"] = bool(q["Q1"]["gap_pt"] <= 0.02 and q["Q1"]["gap_16k"] >= 0.08)
    q["Q2"] = {"gap_mix": mm("mix", gap)}
    q["Q2"]["pass"] = bool(abs(q["Q2"]["gap_mix"]) <= 0.05)
    q3 = {}
    for model in ("8k", "16k"):
        key = [r for r in runs if r["model"] == model and r["is_key"] and not r["degenerate"]]
        if not key:
            q3[model] = {"pass": False, "note": "thiếu mô hình then chốt"}
            continue
        cm, ch = curve(key[0]["lr"], "dev_m"), curve(key[0]["lr"], "hans_eval")
        lo = max((v for l, v in cm.items() if 1 <= l <= 19), default=np.nan)
        hi = max((v for l, v in cm.items() if 20 <= l <= 24), default=np.nan)
        gaps = {l: cm[l] - ch[l] for l in cm if l in ch and l >= 1}
        arg = max(gaps, key=gaps.get) if gaps else None
        q3[model] = {"max_MNLI_L1_19": lo, "max_MNLI_L20_24": hi, "argmax_gap_layer": arg,
                     "pass": bool(lo <= 0.66 and hi >= 0.75 and arg is not None and 20 <= arg <= 24)}
    q["Q3"] = {**q3, "pass": all(v["pass"] for v in q3.values())}
    q["Q4"] = {"16k_devm": mm("16k", "devm"), "16k_hans_E_all": mm("16k", "hans_E_all"),
               "16k_hans_NE_mean3": mm("16k", "hans_NE_mean3"), "mix_hans_total": mm("mix", "hans_total")}
    q["Q4"]["pass"] = bool(q["Q4"]["16k_devm"] >= 0.85 and q["Q4"]["16k_hans_E_all"] >= 0.90
                           and q["Q4"]["16k_hans_NE_mean3"] <= 0.30 and q["Q4"]["mix_hans_total"] >= 0.90)
    q5 = {}
    for model in ("1k", "8k", "16k"):
        cm, ch = mm(model, f"CCS{layer}_MNLI"), mm(model, f"CCS{layer}_HANS")
        q5[model] = {"CCS_MNLI": cm, "CCS_HANS": ch, "pass": bool(cm > ch and ch <= 0.55)}
    q["Q5_khong_gating"] = {**q5, "pass": all(v["pass"] for v in q5.values())}
    # đa số seed (SPEC §7.4)
    ok = rt[~rt.degenerate]
    pt_gap = q["Q1"]["gap_pt"]
    s16 = ok[ok.model == "16k"]
    seed16 = ((s16[gap] >= 0.08) & (pt_gap <= 0.02) & (s16.devm >= 0.85) & (s16.hans_E_all >= 0.90)
              & (s16.hans_NE_mean3 <= 0.30))
    smix = ok[ok.model == "mix"]
    seedmix = (smix[gap].abs() <= 0.05) & (smix.hans_total >= 0.90)
    q["seed_majority"] = {"16k": f"{int(seed16.sum())}/{len(s16)}", "mix": f"{int(seedmix.sum())}/{len(smix)}",
                          "pass": bool(len(s16) and seed16.sum() >= 3 and len(smix) and seedmix.sum() >= 2)}
    # mốc định lượng §7.2 (luôn so với "tầng 23" của V&dL = tầng `layer` của nhóm)
    anchors = {"LR_MNLI_16k": (mm("16k", f"LR{layer}_MNLI"), REF_LR23["16k"][0]),
               "LR_HANS_16k": (mm("16k", f"LR{layer}_HANS"), REF_LR23["16k"][1]),
               "LR_HANS_mix": (mm("mix", f"LR{layer}_HANS"), REF_LR23["mix"][1])}
    q["anchors"] = {k: {"nhom": v[0], "VdL": v[1], "lech": v[0] - v[1], "pass": bool(abs(v[0] - v[1]) <= TOL)}
                    for k, v in anchors.items()}
    n_anchor = sum(v["pass"] for v in q["anchors"].values())
    q["anchors_pass"] = bool(n_anchor >= 2)
    core = q["Q1"]["pass"] and q["Q2"]["pass"] and q["Q3"]["pass"] and q["Q4"]["pass"] and q["seed_majority"]["pass"]
    if not q["Q4"]["pass"]:
        verdict = "KHÔNG ĐẠT (Q4 sai: mô hình không thất bại trên HANS) -> dừng, xem lại pipeline (SPEC §7.4)"
    elif core and q["anchors_pass"]:
        verdict = "ĐẠT"
    elif q["Q1"]["pass"] and q["Q2"]["pass"] and q["Q4"]["pass"]:
        verdict = (f"ĐẠT MỘT PHẦN (Q1, Q2, Q4 đúng; Q3={'đúng' if q['Q3']['pass'] else 'sai'}, "
                   f"đa số seed={'đúng' if q['seed_majority']['pass'] else 'sai'}, mốc {n_anchor}/3) "
                   "-> ghi là phát hiện, tiếp tục (SPEC §7.4)")
    else:
        failed = [k for k in ("Q1", "Q2", "Q3") if not q[k]["pass"]]
        verdict = (f"NGOÀI CÁC TRƯỜNG HỢP SPEC §7.4 ĐỊNH NGHĨA (Q4 đúng nhưng {', '.join(failed)} sai) "
                   "-> không tự xếp loại; báo GVHD, ghi là phát hiện")
    q["G2"] = verdict
    return q


def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def build_report(root, out_dir):
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"không tìm thấy p2_metrics.json dưới {root}")
    rt = run_table(runs)
    mt = model_table(rt)
    os.makedirs(out_dir, exist_ok=True)
    rt.to_csv(os.path.join(out_dir, "p2_runs.csv"), index=False)
    mt.to_csv(os.path.join(out_dir, "p2_models.csv"))
    d2, best = dr2(runs)
    L = []
    L.append("# P2 — Báo cáo tái lập V&dL (sinh tự động, quy tắc viết trước khi có kết quả)\n")
    L.append(f"Số run: {len(rt)} | thoái hoá: {int(rt.degenerate.sum())} "
             f"({', '.join(rt[rt.degenerate].run_id)})\n")
    L.append("## 1. Hành vi (trung bình ± SD qua seed hợp lệ)\n")
    L.append("| Mô hình | n | thoái hoá | DEV-M | DEV-MM | HANS-E | HANS-NE TB3 | NE LO | NE SUB | NE CON | HANS tổng |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m, r in mt.iterrows():
        L.append(f"| {m} | {int(r.n_ok)} | {int(r.n_degenerate)} | {_f(r.devm_mean)} ± {_f(r.devm_sd)} | {_f(r.devmm_mean)} | "
                 f"{_f(r.hans_E_all_mean)} | {_f(r.hans_NE_mean3_mean)} ± {_f(r.hans_NE_mean3_sd)} | "
                 f"{_f(r.NE_LO_mean)} | {_f(r.NE_SUB_mean)} | {_f(r.NE_CON_mean)} | {_f(r.hans_total_mean)} |")
    L.append("\n## 2. LR faithful (TB qua seed) so với V&dL tầng 23 (Bảng 2–3)\n")
    L.append("| Mô hình | L22 MNLI/HANS | L23 MNLI/HANS (gap) | L24 MNLI/HANS (gap) | V&dL '23' MNLI/HANS (gap) |")
    L.append("|---|---|---|---|---|")
    for m, r in mt.iterrows():
        ref = REF_LR23.get(m, (np.nan, np.nan))
        L.append(f"| {m} | {_f(r.LR22_MNLI_mean)}/{_f(r.LR22_HANS_mean)} | "
                 f"{_f(r.LR23_MNLI_mean)}/{_f(r.LR23_HANS_mean)} ({_f(r.gap23_mean, 3)}) | "
                 f"{_f(r.LR24_MNLI_mean)}/{_f(r.LR24_HANS_mean)} ({_f(r.gap24_mean, 3)}) | "
                 f"{_f(ref[0], 2)}/{_f(ref[1], 2)} ({_f(ref[0] - ref[1], 2)}) |")
    L.append("\n## 3. CCS tuyến tính (TB qua seed) so với V&dL tầng 23 (Bảng 4–5; chance ≈ 0.513–0.53)\n")
    L.append("| Mô hình | L23 MNLI/HANS | L24 MNLI/HANS | V&dL '23' MNLI/HANS |")
    L.append("|---|---|---|---|")
    for m, r in mt.iterrows():
        ref = REF_CCS23.get(m, (np.nan, np.nan))
        L.append(f"| {m} | {_f(r.CCS23_MNLI_mean)}/{_f(r.CCS23_HANS_mean)} | {_f(r.CCS24_MNLI_mean)}/"
                 f"{_f(r.CCS24_HANS_mean)} | {_f(ref[0], 2)}/{_f(ref[1], 2)} |")
    L.append("\n## 4. D-R2 — quy ước tầng (Spearman giữa đường LR theo tầng và Bảng 6/7)\n")
    for k, v in d2.items():
        L.append(f"- k = {k:+d} (hàng ℓ ↔ hidden_states[ℓ{k:+d}]): TB {_f(v['mean'])} | "
                 + ", ".join(f"{n} {_f(c)}" for n, c in v["per_curve"].items()))
    primary = 23 + best if best is not None else 23
    L.append(f"\n-> Quy ước chọn theo D-R2: k = {best} -> **'tầng 23' của V&dL = hidden_states[{primary}]**"
             + (" (KHÔNG đủ dữ liệu D-R2 -> mặc định hidden_states[23] theo root §3.7)" if best is None else ""))
    L.append("\n## 5. Gate G2 (SPEC §7.3–§7.4)\n")
    results = {}
    for layer in sorted({primary, 23, 24}):
        q = evaluate_g2(mt, rt, runs, layer)
        results[layer] = q
        tag = "QUY ƯỚC CHÍNH (D-R2)" if layer == primary else "tham khảo — KHÔNG dùng để kết luận"
        L.append(f"### Tầng hidden_states[{layer}] — {tag}\n")
        L.append(f"- Q1 (gap PT ≤ +0.02; gap 16K ≥ +0.08): PT {_f(q['Q1']['gap_pt'])}, 16K {_f(q['Q1']['gap_16k'])} "
                 f"-> {'ĐÚNG' if q['Q1']['pass'] else 'SAI'}")
        L.append(f"- Q2 (|gap MIX| ≤ 0.05): {_f(q['Q2']['gap_mix'])} -> {'ĐÚNG' if q['Q2']['pass'] else 'SAI'}")
        for m in ("8k", "16k"):
            v = q["Q3"].get(m, {})
            L.append(f"- Q3 {m} (max MNLI tầng 1–19 ≤ 0.66; tầng 20–24 ≥ 0.75; gap lớn nhất ở 20–24): "
                     f"{_f(v.get('max_MNLI_L1_19'))} / {_f(v.get('max_MNLI_L20_24'))} / tầng {v.get('argmax_gap_layer')}"
                     f" -> {'ĐÚNG' if v.get('pass') else 'SAI'} {v.get('note', '')}")
        v = q["Q4"]
        L.append(f"- Q4 (16K: DEV-M ≥ 0.85, HANS-E ≥ 0.90, NE TB3 ≤ 0.30; MIX HANS ≥ 0.90): {_f(v['16k_devm'])}, "
                 f"{_f(v['16k_hans_E_all'])}, {_f(v['16k_hans_NE_mean3'])}, {_f(v['mix_hans_total'])} "
                 f"-> {'ĐÚNG' if v['pass'] else 'SAI'}")
        L.append("- Q5 (không gating): " + ", ".join(
            f"{m} {_f(q['Q5_khong_gating'][m]['CCS_MNLI'])}/{_f(q['Q5_khong_gating'][m]['CCS_HANS'])}"
            for m in ("1k", "8k", "16k")) + f" -> {'ĐÚNG' if q['Q5_khong_gating']['pass'] else 'SAI'}")
        L.append(f"- Đa số seed: 16K {q['seed_majority']['16k']} (cần ≥ 3), MIX {q['seed_majority']['mix']} (cần ≥ 2)")
        L.append("- Mốc ±0.05: " + "; ".join(f"{k} {_f(v['nhom'])} vs {v['VdL']} ({_f(v['lech'], 3)}) "
                                            f"{'ĐẠT' if v['pass'] else 'LỆCH'}" for k, v in q["anchors"].items()))
        L.append(f"- **G2: {q['G2']}**\n")
    L.append("## 6. Chẩn đoán phụ\n")
    s16 = rt[(rt.model == "16k") & ~rt.degenerate]
    L.append("- D-R3 (SD HANS-NE qua seed M-16K, so với 100 BERT-base của McCoy 2020): "
             + ", ".join(f"{h} {_f(s16[f'NE_{h}'].std())} (McCoy {MCCOY2020_SD_NE[h]})" for h in ("LO", "SUB", "CON")))
    for r in runs:
        if r["dr1"] is not None and not r["degenerate"] and r["is_key"]:
            f = r["dr1"][r["dr1"].variant == "faithful"]
            parts = [f"L{int(l)} held-out {_f(g[g.split == 'subcase_heldout']['mean'].iloc[0])} / ngẫu nhiên "
                     f"{_f(g[g.split == 'random_matched']['mean'].iloc[0])}" for l, g in f.groupby("layer")]
            L.append(f"- D-R1 {r['run_id']}: " + "; ".join(parts))
    for r in runs:
        if r["lr_aux"] is not None and not r["degenerate"] and r["model"] in REF_T1:
            ref = REF_T1[r["model"]]
            parts = []
            for tok in ("last", "mean"):
                for l in (23, 24):
                    s = r["lr_aux"][(r["lr_aux"].token == tok) & (r["lr_aux"].layer == l)
                                    & (r["lr_aux"].dataset == "dev_m")]
                    if len(s):
                        parts.append(f"{tok} L{l} LR-MNLI {_f(float(s.acc_mean.iloc[0]))}")
            L.append(f"- D-R4 {r['run_id']}: CLS L23 {_f(probe_at(r['lr'], 23, 'dev_m', 'faithful'))}, "
                     + ", ".join(parts) + f" | Bảng 1 V&dL (LR): CLS {ref['cls'][0]}, last {ref['last'][0]}, "
                                          f"mean {ref['mean'][0]}")
    L.append("\nGhi chú: số V&dL lấy từ BẢNG phụ lục (không dùng số trong văn bản, root §2.4 #5). "
             "LR-HANS có thể phản ánh nhận diện template (D-R1). Báo cáo này là đầu vào cho KETQUA_THUCNGHIEM, "
             "không tự thay đổi SPEC.")
    text = "\n".join(L)
    with open(os.path.join(out_dir, "P2_REPORT.md"), "w") as f:
        f.write(text)
    with open(os.path.join(out_dir, "p2_g2.json"), "w") as f:
        json.dump({"dr2": {str(k): v for k, v in d2.items()}, "dr2_best_k": best, "primary_layer": primary,
                   "g2": {str(k): v for k, v in results.items()}}, f, indent=2, ensure_ascii=False, default=str)
    return text, results, primary
