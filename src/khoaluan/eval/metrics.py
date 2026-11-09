"""
metrics.py — chỉ số hành vi cho P2 (SPEC_REPRODUCE_VDL v1.1 §4, §7.3; root §3.4).
Hàm thuần (pandas/numpy), không phụ thuộc GPU -> unit test được.

Quy ước nhãn: 0 = entailment (E), 1 = non-entailment (NE). Dự đoán = argmax 2 logit.
RG_h = Acc_{h,E} − Acc_{h,NE} (reliance gap; endpoint chính của P4 — ở P2 chỉ báo cáo).
"""
import numpy as np
import pandas as pd

HEURS = ("lexical_overlap", "subsequence", "constituent")
SHORT = {"lexical_overlap": "LO", "subsequence": "SUB", "constituent": "CON"}


def accuracy(df):
    return float((df.pred == df.y).mean()) if len(df) else float("nan")


def hans_table(df):
    """df: preds HANS (cột y, pred, heuristic, subcase). Trả về dict phẳng + bảng subcase."""
    out = {"hans_acc_total": accuracy(df)}
    for h in HEURS:
        s = df[df.heuristic == h]
        e, ne = accuracy(s[s.y == 0]), accuracy(s[s.y == 1])
        out[f"hans_E_{SHORT[h]}"] = e
        out[f"hans_NE_{SHORT[h]}"] = ne
        out[f"RG_{SHORT[h]}"] = e - ne
        out[f"n_E_{SHORT[h]}"] = int((s.y == 0).sum())
        out[f"n_NE_{SHORT[h]}"] = int((s.y == 1).sum())
    out["hans_E_all"] = accuracy(df[df.y == 0])
    out["hans_NE_mean3"] = float(np.mean([out[f"hans_NE_{SHORT[h]}"] for h in HEURS]))
    sub = (df.assign(correct=(df.pred == df.y))
             .groupby(["heuristic", "subcase", "y"], as_index=False)
             .agg(n=("correct", "size"), acc=("correct", "mean"), pE=("p_E", "mean")))
    return out, sub


def q4_check(m, model_kind):
    """Tiêu chí định tính Q4 (SPEC_REPRODUCE §7.3), tiền đăng ký. Ngưỡng DEV-M 0.85 là [PHỎNG ĐOÁN ngưỡng].
    model_kind: 'mnli' (M-16K) | 'mix' (M-MIX). Trả về dict tiêu chí -> (giá trị, đạt?)."""
    if model_kind == "mnli":
        crit = {
            "devm_acc>=0.85": (m["devm_acc"], m["devm_acc"] >= 0.85),
            "hans_E_all>=0.90": (m["hans_E_all"], m["hans_E_all"] >= 0.90),
            "hans_NE_mean3<=0.30": (m["hans_NE_mean3"], m["hans_NE_mean3"] <= 0.30),
        }
    elif model_kind == "mix":
        crit = {"hans_acc_total>=0.90": (m["hans_acc_total"], m["hans_acc_total"] >= 0.90)}
    else:
        raise ValueError(model_kind)
    crit["Q4_pass"] = (None, all(v[1] for v in crit.values()))
    return crit


def is_degenerate(size, devm_acc):
    """SPEC_REPRODUCE §3.3: mô hình >= 1K ví dụ có DEV-M acc < 0.60 -> thoái hoá (giữ bản ghi, chạy lại seed+100)."""
    return bool(size >= 1000 and devm_acc < 0.60)


def preds_frame(frame, logits):
    """Ghép logit với metadata theo đúng thứ tự dòng của frame. logits: (N,2) float32."""
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (len(frame), 2):
        raise ValueError(f"logits shape {logits.shape} != ({len(frame)}, 2)")
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    cols = [c for c in ["uid", "y", "heuristic", "subcase", "template"] if c in frame.columns]
    out = frame[cols].copy().reset_index(drop=True)
    out["logit_0"], out["logit_1"] = logits[:, 0], logits[:, 1]
    out["p_E"] = p[:, 0]
    out["pred"] = logits.argmax(axis=1).astype(int)
    return out


def flatten(d, prefix=""):
    return {f"{prefix}{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def summarize_sets(preds):
    """preds: dict tên tập -> preds_frame. Trả về dict chỉ số cho registry."""
    m = {}
    if "dev_m" in preds:
        m["devm_acc"] = accuracy(preds["dev_m"])
    if "dev_mm" in preds:
        m["devmm_acc"] = accuracy(preds["dev_mm"])
    if "hans_eval" in preds:
        h, _ = hans_table(preds["hans_eval"])
        m.update(h)
    if "hans_train_B" in preds:
        m["htB_acc"] = accuracy(preds["hans_train_B"])
    return m


def subcase_table(preds):
    return hans_table(preds["hans_eval"])[1] if "hans_eval" in preds else pd.DataFrame()
