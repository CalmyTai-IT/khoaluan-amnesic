"""
pilot.py — erasure tại L_out = pooler_output (SPEC_ERASURE v0.1 §2.1, §4.1, §5–§7), bản pilot / bản chạy đủ dùng chung code.

    python scripts/run_erase_pilot.py --data-root /kaggle/input --runs-root /kaggle/input \
        --out /kaggle/working/erase --run-ids bertL-mnliB-16k-s111 bertL-mnliB-16k-s22 bertL-mnliB-16k-s33

Bước:
  1. Dựng tập fit khái niệm (sets.py, tất định, seed 1234) — giống nhau cho mọi mô hình.
  2. Mỗi mô hình: trích pooler_output cho ~11.5K ví dụ MNLI-POOL cần cho c_LO / c_LABEL / MNLI-HC (GPU, ~1 phút);
     HANS-EVAL, HT-B, DEV-M lấy từ cache P2 (std_*_pooler.npy). Head đọc từ checkpoint (classifier.weight/bias).
  3. Mỗi khái niệm: LEACE fit trên phần "fit"; đo ΔRG (3 heuristic), ΔBalAcc, ΔAUROC trên HANS-EVAL; ΔAcc DEV-M
     (toàn bộ, LO−); ΔRG trên MNLI-HC; probe-after (LR); giải mã được trước khi xoá (held-out ngẫu nhiên và held-out
     theo template); chuyển miền c_LO -> HANS; kiểm bảo đảm (chênh logit trung bình giữa hai lớp khái niệm ≈ 0).
  4. Null: N3 hoán vị nhãn khái niệm trong từng ô (fit lại đúng estimator) và N1 hướng ngẫu nhiên cùng rank,
     mỗi loại N_NULL lần. p một phía; Holm trên 9 ô (c_LO, c_SUB, c_CON × 3 cột) cho từng mô hình.
  5. Tổng hợp qua seed: quy tắc "vững" SPEC_ERASURE §7.3 ở mức MVP (mọi seed đã chạy).

NGƯỠNG (SPEC_ERASURE v0.1 §9, cố định TRƯỚC khi có kết quả erasure; ghi 10/11/2026):
  δ_min = 0.05; thiệt hại MNLI |ΔAcc DEV-M| ≤ 2.0 điểm và ΔAcc DEV-M LO− ≥ −2.0 điểm; probe-after LR BalAcc ≤ 0.55;
  control dương c_LABEL: DEV-M acc sau xoá ∈ [0.45, 0.60]; G3: BalAcc giải mã ≥ 0.70, c_LO -> HANS ≥ 0.80.
  SỬA so với v0.1 (trước khi thấy dữ liệu): N_NULL = 1000 thay vì 100 — với 100 lần, p nhỏ nhất 1/101 ≈ 0.0099 lớn hơn
  ngưỡng Holm đầu tiên 0.05/9 ≈ 0.0056, nên không ô nào có thể có ý nghĩa. Probe dùng LR (StandardScaler, C = 1) thay
  vì chọn C bằng CV (rút gọn cho pilot). Chưa có: MP/INLP, N2, re-injection, xoá tuần tự, L_ℓ, độ ổn định tập fit.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from khoaluan.data import load as L
from khoaluan.erase import sets as SETS
from khoaluan.erase.leace_fast import DT, FitStats, stratified_permutation, var_removed

DELTA_MIN = 0.05
MNLI_DAMAGE = 2.0          # điểm %
PROBE_AFTER_MAX = 0.55
POS_CTRL = (0.45, 0.60)
G3_BALACC, G3_TRANSFER = 0.70, 0.80
ALPHA = 0.05
HEURS = (("lexical_overlap", "LO"), ("subsequence", "SUB"), ("constituent", "CON"))
CONCEPTS = ["c_LABEL", "c_LO", "c_SUB", "c_CON", "c_JOINT", "c_CTRL_H"]
HOLM_FAMILY = ["c_LO", "c_SUB", "c_CON"]
# Khuôn kỳ vọng (SPEC_DULIEU §5.3): ● có thể khác 0; ○ phải ≈ 0 (phép thử đặc hiệu)
EXPECTED = {"c_LO": "●●●", "c_SUB": "○●●", "c_CON": "○○●", "c_JOINT": "○●●", "c_CTRL_H": "○○○"}
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} phút] {msg}", flush=True)


# ------------------------------------------------------------------ chỉ số hành vi từ logits
def hans_metrics(logits, heur, y, auroc=True):
    logits = np.asarray(logits)
    pred = (logits[:, 1] > logits[:, 0]).astype(int)
    corr = pred == y
    out = {}
    for h, tag in HEURS:
        m = heur == h
        e, ne = corr[m & (y == 0)].mean(), corr[m & (y == 1)].mean()
        out[f"RG_{tag}"], out[f"accE_{tag}"], out[f"accNE_{tag}"] = e - ne, e, ne
        out[f"BalAcc_{tag}"] = (e + ne) / 2
        if auroc:
            z = logits[m, 0] - logits[m, 1]                       # điểm "entailment" (đơn điệu với P(E))
            out[f"AUROC_{tag}"] = roc_auc_score((y[m] == 0).astype(int), z)
    out["hans_acc"] = corr.mean()
    return out


def rg_vector(logits, heur_masks):
    """Nhanh cho null: ΔRG 3 heuristic. heur_masks: list (maskE, maskNE) theo HEURS."""
    pred_ne = logits[:, 1] > logits[:, 0]
    return np.array([(~pred_ne[mE]).mean() - pred_ne[mN].mean() for mE, mN in heur_masks])


def acc(logits, y):
    return float(((np.asarray(logits)[:, 1] > np.asarray(logits)[:, 0]).astype(int) == y).mean())


# ------------------------------------------------------------------ nạp dữ liệu
def find_run_dir(runs_root, run_id):
    hits = sorted({os.path.realpath(os.path.dirname(p)) for p in
                   glob.glob(os.path.join(runs_root, "**", run_id, "std_hans_eval_pooler.npy"), recursive=True)})
    if len(hits) != 1:
        raise FileNotFoundError(f"[DỪNG] cần đúng 1 thư mục cache cho {run_id} dưới {runs_root}, thấy {hits}")
    if not os.path.exists(os.path.join(hits[0], "model", "model.safetensors")):
        raise FileNotFoundError(f"[DỪNG] {run_id}: thiếu model/model.safetensors (cần head + trích MNLI-POOL)")
    return hits[0]


def load_head(model_dir):
    from safetensors import safe_open
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        keys = set(f.keys())
        wk = next(k for k in ("classifier.weight", "bert.classifier.weight") if k in keys)
        W = f.get_tensor(wk).to(DT)
        b = f.get_tensor(wk.replace("weight", "bias")).to(DT)
    return W, b


def load_std(run_dir, name, frame):
    X = np.load(os.path.join(run_dir, f"std_{name}_pooler.npy")).astype(np.float64)
    with open(os.path.join(run_dir, f"std_{name}_uids.txt")) as f:
        uids = f.read().split("\n")
    if uids != frame.uid.tolist():
        raise ValueError(f"[DỪNG] thứ tự uid của cache std_{name} khác dữ liệu")
    return X


def extract_pool(model_dir, union, out_path, batch, log=log):
    """Trích pooler_output cho các ví dụ MNLI-POOL cần dùng (đầu vào chuẩn, max_length 128, fp16 như P2)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from khoaluan.eval.forward import run_pairs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    t = time.time()
    r = run_pairs(model, tok, union, 128, batch, fp16=device.type == "cuda", cls_layers=[])
    np.save(out_path, r["pooler"].astype(np.float16))
    np.save(out_path.replace("_pooler.npy", "_logits.npy"), r["logits"])
    log(f"  trích MNLI-POOL: {len(union)} ví dụ, {(time.time() - t) / 60:.1f} phút ({device})")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.load(out_path).astype(np.float64)


def probe_balacc(Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2 or len(yte) == 0:
        return float("nan"), None
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    clf.fit(Xtr, ytr)
    return float(balanced_accuracy_score(yte, clf.predict(Xte))), clf


# ------------------------------------------------------------------ một mô hình
def analyze_model(run_id, run_dir, data_dir, sets, union, pool_X, n_null, method, seed, log=log):
    W, b = load_head(os.path.join(run_dir, "model"))
    W_np = W.numpy()
    fr = {s: L.eval_frame(data_dir, s) for s in ("hans_eval", "hans_train_B", "dev_m")}
    X = {s: load_std(run_dir, s, fr[s]) for s in fr}
    # head(cache fp16) phải tái tạo dự đoán đã lưu ở P2
    checks = {}
    for s in ("hans_eval", "dev_m"):
        saved = pd.read_parquet(os.path.join(run_dir, f"preds_{s}.parquet"))
        rec = X[s] @ W_np.T + b.numpy()
        agree = float((rec.argmax(1) == saved.pred.to_numpy()).mean())
        checks[f"head_reproduces_preds_{s}"] = agree
        if agree < 0.99:
            raise ValueError(f"[DỪNG] {run_id}: head trên cache {s} chỉ khớp {agree:.4f} dự đoán đã lưu")
    pool_logits = pool_X @ W_np.T + b.numpy()
    base_logits = {s: X[s] @ W_np.T + b.numpy() for s in ("hans_eval", "dev_m")}
    heur, yh = fr["hans_eval"].heuristic.to_numpy(), fr["hans_eval"].y.to_numpy()
    masks = [((heur == h) & (yh == 0), (heur == h) & (yh == 1)) for h, _ in HEURS]
    devm = pd.read_parquet(os.path.join(data_dir, "mnli_bin_dev_m.parquet"))[["uid", "lo"]].set_index("uid")
    dev_lo = devm.loc[fr["dev_m"].uid, "lo"].to_numpy()
    ydev = fr["dev_m"].y.to_numpy()
    pidx = {u: i for i, u in enumerate(union.uid)}
    hc = sets["MNLI_HC"]
    hc_i = np.array([pidx[u] for u in hc.uid])
    hc_y = hc.y.to_numpy()

    def behaviour(hans_lg, dev_lg, hc_lg, auroc=True):
        m = hans_metrics(hans_lg, heur, yh, auroc=auroc)
        m["devm_acc"] = acc(dev_lg, ydev)
        m["devm_acc_LOminus"] = acc(dev_lg[dev_lo == 0], ydev[dev_lo == 0])
        m["devm_acc_LOplus"] = acc(dev_lg[dev_lo == 1], ydev[dev_lo == 1])
        pc = (hc_lg[:, 1] > hc_lg[:, 0]).astype(int) == hc_y
        m["RG_HC"] = float(pc[hc_y == 0].mean() - pc[hc_y == 1].mean())
        return m

    base = behaviour(base_logits["hans_eval"], base_logits["dev_m"], pool_logits[hc_i])
    D_hans = torch.as_tensor(X["hans_eval"], dtype=DT)
    D_dev = torch.as_tensor(X["dev_m"], dtype=DT)
    D_hc = torch.as_tensor(pool_X[hc_i], dtype=DT)
    htb_idx = {u: i for i, u in enumerate(fr["hans_train_B"].uid)}
    cells, concepts_out, nulls = [], {}, {}
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    for cname in CONCEPTS:
        df = sets[cname]
        if cname in ("c_LO", "c_LABEL"):
            Xc = pool_X[[pidx[u] for u in df.uid]]
            lg_c = pool_logits[[pidx[u] for u in df.uid]]
        else:
            ii = [htb_idx[u] for u in df.uid]
            Xc = X["hans_train_B"][ii]
            lg_c = X["hans_train_B"][ii] @ W_np.T + b.numpy()
        z = df.z.to_numpy()
        fit = (df.split == "fit").to_numpy()
        held = ~fit
        k = int(z.max()) + 1
        stats = FitStats(Xc[fit], method=method)
        er = stats.eraser(z[fit])
        info = {"n_fit": int(fit.sum()), "n_held": int(held.sum()), "k": k, "rank": er.rank,
                "rank_expected": k - 1, "var_removed": var_removed(stats, er)}
        # giải mã được trước khi xoá (G3) + probe-after
        info["decode_balacc_held"], probe = probe_balacc(Xc[fit], z[fit], Xc[held], z[held])
        Xe = er(Xc)
        info["probe_after_balacc_held"] = probe_balacc(Xe[fit], z[fit], Xe[held], z[held])[0]
        if "tmpl_held" in df.columns and df.tmpl_held.any():
            th = df.tmpl_held.to_numpy()
            info["decode_balacc_template_heldout"] = probe_balacc(Xc[~th], z[~th], Xc[th], z[th])[0]
        if cname == "c_LO":
            info["transfer_HANS_LOplus_rate"] = float(probe.predict(X["hans_eval"]).mean())
        # kiểm bảo đảm: chênh logit (E − NE) trung bình giữa các lớp khái niệm, trên held-out
        dz0 = lg_c[:, 0] - lg_c[:, 1]
        lg_after = Xe @ W_np.T + b.numpy()
        dz1 = lg_after[:, 0] - lg_after[:, 1]
        def gap(d, m, z=z, k=k):
            return float(np.ptp([d[m & (z == c)].mean() for c in range(k)]))
        info["logit_gap_between_classes_before_held"] = gap(dz0, held)
        info["logit_gap_between_classes_after_held"] = gap(dz1, held)
        info["logit_gap_between_classes_after_fit"] = gap(dz1, fit)
        # hiệu ứng quan sát
        Dh, Dd, Dc = D_hans - stats.mu, D_dev - stats.mu, D_hc - stats.mu
        after = behaviour(base_logits["hans_eval"] + er.logit_shift(Dh, W).numpy(),
                          base_logits["dev_m"] + er.logit_shift(Dd, W).numpy(),
                          pool_logits[hc_i] + er.logit_shift(Dc, W).numpy())
        delta = {kk: after[kk] - base[kk] for kk in base}
        info["after"] = after
        info["delta"] = delta
        info["mnli_damage_ok"] = bool(abs(delta["devm_acc"]) * 100 <= MNLI_DAMAGE
                                      and delta["devm_acc_LOminus"] * 100 >= -MNLI_DAMAGE)
        if cname == "c_LABEL":
            info["positive_control_ok"] = bool(POS_CTRL[0] <= after["devm_acc"] <= POS_CTRL[1])
            concepts_out[cname] = info
            log(f"  {cname}: DEV-M acc sau xoá {after['devm_acc']:.3f} (control dương "
                f"{'ĐẠT' if info['positive_control_ok'] else 'KHÔNG ĐẠT'}), rank {er.rank}")
            continue
        # null N3 (hoán vị trong ô, fit lại) và N1 (ngẫu nhiên cùng rank)
        strata = df.stratum.to_numpy()[fit]
        base_rg = rg_vector(base_logits["hans_eval"], masks)
        base_dev = base["devm_acc"]
        res = {"N3": np.zeros((n_null, 4)), "N1": np.zeros((n_null, 4))}
        for i in range(n_null):
            for kind in ("N3", "N1"):
                e = stats.eraser(stratified_permutation(z[fit], strata, rng)) if kind == "N3" \
                    else stats.random_eraser(k - 1, gen)
                hl = base_logits["hans_eval"] + e.logit_shift(Dh, W).numpy()
                dl = base_logits["dev_m"] + e.logit_shift(Dd, W).numpy()
                res[kind][i, :3] = rg_vector(hl, masks) - base_rg
                res[kind][i, 3] = acc(dl, ydev) - base_dev
        nulls[cname] = res
        for j, (_, tag) in enumerate(HEURS):
            obs = delta[f"RG_{tag}"]
            row = {"run_id": run_id, "concept": cname, "column": tag, "expected": EXPECTED[cname][j],
                   "RG_base": base[f"RG_{tag}"], "dRG": obs, "dBalAcc": delta[f"BalAcc_{tag}"],
                   "dAUROC": delta[f"AUROC_{tag}"], "dAccE": delta[f"accE_{tag}"], "dAccNE": delta[f"accNE_{tag}"]}
            for kind in ("N3", "N1"):
                nv = res[kind][:, j]
                row[f"p_{kind}"] = float((1 + (nv <= obs).sum()) / (1 + n_null))
                row[f"q05_{kind}"] = float(np.quantile(nv, 0.05))
                row[f"mean_{kind}"] = float(nv.mean())
            cells.append(row)
        for kind in ("N3", "N1"):
            info[f"null_{kind}_dDevAcc_q025_q975"] = [float(np.quantile(res[kind][:, 3], q)) for q in (0.025, 0.975)]
        concepts_out[cname] = info
        log(f"  {cname}: rank {er.rank}/{k - 1}, giải mã {info['decode_balacc_held']:.3f}, sau xoá "
            f"{info['probe_after_balacc_held']:.3f} | ΔRG LO/SUB/CON "
            + "/".join(f"{delta[f'RG_{t}']:+.3f}" for _, t in HEURS) + f" | ΔDEV-M {delta['devm_acc'] * 100:+.2f} điểm")
    cells = pd.DataFrame(cells)
    fam = cells.concept.isin(HOLM_FAMILY)
    cells["p_holm"] = np.nan
    cells.loc[fam, "p_holm"] = holm(cells.loc[fam, "p_N3"].to_numpy())
    cells.loc[~fam, "p_holm"] = cells.loc[~fam, "p_N3"]          # ngoài họ Holm: p chưa hiệu chỉnh
    cells["cell_robust_this_seed"] = ((cells.dRG <= -DELTA_MIN) & (cells.p_holm < ALPHA)
                                      & (cells.dRG < cells.q05_N3))
    return {"run_id": run_id, "checks": checks, "base": base, "concepts": concepts_out}, cells, nulls


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


# ------------------------------------------------------------------ tổng hợp qua seed
def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def family(run_id):
    import re
    return re.sub(r"-s\d+$", "", run_id)


def summarize(models, cells, sets_report, n_null, method):
    """Báo cáo theo từng họ mô hình (bertL-mnliB-16k, bertL-mix-16k, ...): đếm seed 'vững' KHÔNG trộn giữa các họ."""
    fams = sorted({family(m["run_id"]) for m in models})
    if len(fams) > 1:
        return "\n\n".join(summarize([m for m in models if family(m["run_id"]) == f],
                                      cells[cells.run_id.map(family) == f], sets_report, n_null, method)
                            for f in fams)
    n = len(models)
    L_ = [f"# {fams[0]}: erasure tại L_out ({method.upper()}) — {n} mô hình, null {n_null} lần mỗi loại\n"]
    L_.append("## Kiểm tra bắt buộc")
    for m in models:
        c = m["concepts"]
        L_.append(f"- {m['run_id']}: head tái tạo dự đoán P2 HANS {m['checks']['head_reproduces_preds_hans_eval']:.4f}"
                  f" / DEV-M {m['checks']['head_reproduces_preds_dev_m']:.4f}; control dương c_LABEL: DEV-M "
                  f"{c['c_LABEL']['after']['devm_acc']:.3f} -> {'ĐẠT' if c['c_LABEL']['positive_control_ok'] else 'KHÔNG ĐẠT'}"
                  f" (cần {POS_CTRL[0]}–{POS_CTRL[1]})")
    L_.append(f"- Tập c_LO: n* = {sets_report['mnli']['n_star_LO+C_fit']}, tổng {sets_report['mnli']['c_LO_total']}"
              f" (fit {sets_report['mnli']['c_LO_fit']}); |SMD| lớn nhất {sets_report['mnli']['SMD_max_abs']:.3f}"
              f" (cần < 0.1); MNLI-HC {sets_report['mnli']['MNLI_HC']}")
    L_.append("\n## Khái niệm: giải mã được (G3) và xoá có hiệu lực (TB qua seed)")
    L_.append("| Khái niệm | rank | giải mã held | giải mã template mới | sau xoá (≤0.55) | VarRemoved | "
              "ΔDEV-M (điểm) | ΔDEV-M LO− | ΔRG MNLI-HC | khác |")
    L_.append("|---|---|---|---|---|---|---|---|---|---|")
    for cn in CONCEPTS:
        rows = [m["concepts"][cn] for m in models]
        def g(key, rows=rows):
            v = [r.get(key, np.nan) for r in rows]
            v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
            return float(np.mean(v)) if v else float("nan")
        def gd(key, rows=rows):
            return float(np.mean([r["delta"][key] for r in rows]))
        extra = ""
        if cn == "c_LO":
            extra = f"c_LO→HANS LO+ {g('transfer_HANS_LOplus_rate'):.2f} (cần ≥ 0.80)"
        if cn == "c_LABEL":
            extra = f"DEV-M sau xoá {np.mean([r['after']['devm_acc'] for r in rows]):.3f}"
        L_.append(f"| {cn} | {g('rank'):.0f}/{rows[0]['rank_expected']} | {_f(g('decode_balacc_held'))} | "
                  f"{_f(g('decode_balacc_template_heldout'))} | {_f(g('probe_after_balacc_held'))} | "
                  f"{g('var_removed'):.4f} | {gd('devm_acc') * 100:+.2f} | {gd('devm_acc_LOminus') * 100:+.2f} | "
                  f"{gd('RG_HC'):+.3f} | {extra} |")
    L_.append("\n## Ma trận cross-effect: ΔRG trên HANS (TB ± SD qua seed) | số seed 'vững' / tổng")
    L_.append("'vững' ở một seed = ΔRG ≤ −0.05 VÀ p (N3, Holm trong họ c_LO/c_SUB/c_CON) < 0.05 VÀ thấp hơn phân vị 5% của N3.")
    L_.append("| Khái niệm | kỳ vọng LO/SUB/CON | cột LO | cột SUB | cột CON |")
    L_.append("|---|---|---|---|---|")
    for cn in [c for c in CONCEPTS if c != "c_LABEL"]:
        parts = []
        for tag in ("LO", "SUB", "CON"):
            s = cells[(cells.concept == cn) & (cells.column == tag)]
            parts.append(f"{s.dRG.mean():+.3f} ± {s.dRG.std(ddof=1) if len(s) > 1 else 0:.3f} | "
                         f"{int(s.cell_robust_this_seed.sum())}/{len(s)} "
                         f"(p_N3 max {s.p_N3.max():.3f}; N3 q05 TB {s.q05_N3.mean():+.3f}; N1 q05 TB {s.q05_N1.mean():+.3f})")
        L_.append(f"| {cn} | {EXPECTED[cn]} | " + " | ".join(parts) + " |")
    L_.append("\n## Chiều thay đổi trên HANS (TB qua seed): ΔAccE / ΔAccNE / ΔAUROC")
    for cn in [c for c in CONCEPTS if c != "c_LABEL"]:
        s = cells[cells.concept == cn].groupby("column")[["dAccE", "dAccNE", "dAUROC"]].mean()
        L_.append(f"- {cn}: " + "; ".join(f"{t} {s.loc[t, 'dAccE']:+.3f}/{s.loc[t, 'dAccNE']:+.3f}/{s.loc[t, 'dAUROC']:+.3f}"
                                         for t in ("LO", "SUB", "CON")))
    L_.append("\n## Baseline (trước xoá, TB qua seed)")
    keys = ["RG_LO", "RG_SUB", "RG_CON", "devm_acc", "RG_HC"]
    L_.append(", ".join(f"{k} {np.mean([m['base'][k] for m in models]):.3f}" for k in keys))
    L_.append("\nGhi chú: pilot dùng ngưỡng SPEC_ERASURE v0.1 §9 (cố định trước khi chạy); đây KHÔNG phải kết quả cuối — "
              "chưa có MP/INLP, N2, re-injection, xoá tuần tự, M-MIX, tầng giữa.")
    return "\n".join(L_)


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-ids", nargs="+", required=True)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--method", choices=["leace", "orth"], default="leace")
    ap.add_argument("--n-label", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--allow-unofficial-data", action="store_true")
    a = ap.parse_args(argv)
    runs_root = a.runs_root or a.data_root
    os.makedirs(a.out, exist_ok=True)
    data_dir = L.find_data_dir(a.data_root)
    if not a.allow_unofficial_data:
        L.verify_data(data_dir)
        log(f"dữ liệu khớp lần chạy A-06 chính thức: {data_dir}")
    run_dirs = {r: find_run_dir(runs_root, r) for r in a.run_ids}       # kiểm đủ mô hình trước khi làm gì
    log("tìm thấy cache P2: " + "; ".join(f"{k} -> {v}" for k, v in run_dirs.items()))
    sets, union, rep = SETS.build_all(data_dir, seed=a.seed, n_label=a.n_label)
    SETS.save_sets(sets, union, rep, os.path.join(a.out, "sets"))
    log(f"tập fit: c_LO {rep['mnli']['c_LO_total']} (n* {rep['mnli']['n_star_LO+C_fit']}), "
        f"|SMD| max {rep['mnli']['SMD_max_abs']:.3f}, MNLI-POOL cần trích {len(union)}")
    models, all_cells = [], []
    for rid, rdir in run_dirs.items():
        log(f"=== {rid}")
        odir = os.path.join(a.out, rid)
        os.makedirs(odir, exist_ok=True)
        ppath = os.path.join(odir, "pool_pooler.npy")
        pool_X = extract_pool(os.path.join(rdir, "model"), union, ppath, a.batch) if not os.path.exists(ppath) \
            else np.load(ppath).astype(np.float64)
        res, cells, nulls = analyze_model(rid, rdir, data_dir, sets, union, pool_X, a.n_null, a.method, a.seed)
        cells.to_csv(os.path.join(odir, "erase_cells.csv"), index=False)
        np.savez_compressed(os.path.join(odir, "nulls.npz"),
                            **{f"{c}_{k}": v for c, d in nulls.items() for k, v in d.items()})
        with open(os.path.join(odir, "erase_model.json"), "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False, default=float)
        models.append(res)
        all_cells.append(cells)
    cells = pd.concat(all_cells, ignore_index=True)
    cells.to_csv(os.path.join(a.out, "erase_cells_all.csv"), index=False)
    text = summarize(models, cells, rep, a.n_null, a.method)
    with open(os.path.join(a.out, "erase_summary.md"), "w") as f:
        f.write(text)
    print(text)
    log(f"XONG. Gửi Claude: {a.out}/erase_summary.md + erase_cells_all.csv + sets/sets_report.json")
    return models, cells


if __name__ == "__main__":
    main()
