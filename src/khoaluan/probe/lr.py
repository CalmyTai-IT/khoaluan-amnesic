"""
lr.py — probe LR trên contrast pair (SPEC_REPRODUCE_VDL v1.1 §5.3, §6.1).

Quy ước DLK (đã đọc mã evaluate.py/utils.py): `neg` = đáp án của nhãn 0 = answer_choices[0] = "Yes",
`pos` = đáp án của nhãn 1 = "No"; đặc trưng LR là x = h_neg − h_pos = h(Yes) − h(No).
(Độ chính xác LR và CCS không đổi khi đổi dấu x, nhưng quy ước phải thống nhất trong luận văn.)

- 'faithful': LogisticRegression(class_weight="balanced") mặc định sklearn (L2, C=1, lbfgs, max_iter=100),
  KHÔNG chuẩn hoá; đếm cảnh báo không hội tụ, KHÔNG tự tăng max_iter.  [TÁC GIẢ – trí nhớ] + mã DLK
- 'robust': StandardScaler (fit trên train) + max_iter=5000.
Train/test chia trên VÍ DỤ: h(Yes), h(No) của cùng một ví dụ luôn cùng phía.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def dlk_x(h_yes, h_no):
    return h_yes - h_no


def splits_mnli(y, n_runs=5, n_sample=2000):
    """Lần r (1..5): rút n_sample ví dụ phân tầng theo nhãn (seed r), chia 1,000/1,000 phân tầng.
    Các lần có thể chồng lấn (DEV-M chỉ 6.7K) — ghi rõ khi báo cáo."""
    idx = np.arange(len(y))
    out = []
    for r in range(1, n_runs + 1):
        s, _ = train_test_split(idx, train_size=n_sample, stratify=y, random_state=r) if len(y) > n_sample \
            else (idx, None)
        tr, te = train_test_split(s, test_size=0.5, stratify=y[s], random_state=r)
        out.append((np.sort(tr), np.sort(te)))
    return out


def splits_hans(strata, n_runs=5, seed=1234):
    """HANS: chia tập trích (10K) thành n_runs phần RỜI NHAU phân tầng theo heuristic × nhãn;
    lần r: chia đôi phần r phân tầng theo heuristic × nhãn (seed r)."""
    strata = np.asarray(strata)
    skf = StratifiedKFold(n_splits=n_runs, shuffle=True, random_state=seed)
    out = []
    for r, (_, part) in enumerate(skf.split(np.zeros(len(strata)), strata), start=1):
        tr, te = train_test_split(part, test_size=0.5, stratify=strata[part], random_state=r)
        out.append((np.sort(tr), np.sort(te)))
    return out


def fit_eval(x_tr, y_tr, x_te, y_te, variant):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        if variant == "faithful":
            clf = LogisticRegression(class_weight="balanced")
        elif variant == "robust":
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
        else:
            raise ValueError(variant)
        clf.fit(x_tr, y_tr)
        acc = float((clf.predict(x_te) == y_te).mean())
    n_warn = sum(issubclass(x.category, ConvergenceWarning) for x in w)
    return acc, n_warn


def run_lr(h_yes, h_no, y, splits, layers, layer_ids, variants=("faithful", "robust"), dataset="", log=None):
    """h_yes/h_no: (N, L, H) (np.memmap được). layer_ids: số hiệu tầng thật tương ứng chỉ số trục 1.
    layers: các tầng cần chạy (số hiệu thật). Trả về DataFrame dài: dataset, layer, run, variant, acc, n_conv_warn."""
    pos = {l: i for i, l in enumerate(layer_ids)}
    rows = []
    for l in layers:
        x = dlk_x(np.asarray(h_yes[:, pos[l]], dtype=np.float32), np.asarray(h_no[:, pos[l]], dtype=np.float32))
        for r, (tr, te) in enumerate(splits, start=1):
            for v in variants:
                acc, nw = fit_eval(x[tr], y[tr], x[te], y[te], v)
                rows.append({"dataset": dataset, "layer": l, "run": r, "variant": v, "acc": acc, "n_conv_warn": nw})
        if log:
            sub = [r_["acc"] for r_ in rows if r_["layer"] == l and r_["variant"] == "faithful"]
            log(f"  LR {dataset} tầng {l:2d}: faithful {np.mean(sub):.3f}")
    return pd.DataFrame(rows)


def summarize(df):
    """Trung bình ± SD qua các lần chạy, theo dataset × layer × variant."""
    g = df.groupby(["dataset", "layer", "variant"]).agg(acc_mean=("acc", "mean"), acc_sd=("acc", "std"),
                                                       n_runs=("acc", "size"), conv_warn=("n_conv_warn", "sum"))
    return g.reset_index()


def fp16_storage_diagnostic(h_yes, h_no, layer_ids):
    """Lưu fp16 có làm hỏng hiệu h(Yes) − h(No) không? Trả về theo tầng:
    rel_diff = trung vị ||a−b|| / ||a||  (hai biểu diễn khác nhau bao nhiêu);
    rel_err16 = trung vị ||(f16(a)−f16(b)) − (a−b)|| / ||a−b||  (sai số do lưu fp16)."""
    rows = []
    for i, l in enumerate(layer_ids):
        a = np.asarray(h_yes[:, i], dtype=np.float32)
        b = np.asarray(h_no[:, i], dtype=np.float32)
        d = a - b
        d16 = a.astype(np.float16).astype(np.float32) - b.astype(np.float16).astype(np.float32)
        nd = np.linalg.norm(d, axis=1)
        ok = nd > 0
        rel_diff = float(np.median(nd / np.maximum(np.linalg.norm(a, axis=1), 1e-12)))
        rel_err = float(np.median(np.linalg.norm(d16 - d, axis=1)[ok] / nd[ok])) if ok.any() else float("nan")
        rows.append({"layer": l, "rel_diff": rel_diff, "rel_err16": rel_err, "frac_zero_diff": float(1 - ok.mean())})
    return pd.DataFrame(rows)
