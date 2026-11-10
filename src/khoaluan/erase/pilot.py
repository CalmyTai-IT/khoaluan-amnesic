"""
pilot.py — erasure tại L_out = pooler_output (SPEC_ERASURE v0.1 §2.1, §4.1, §5–§7) và điều phối L_ℓ (layers.py).

    python scripts/run_erase.py --data-root /kaggle/input --runs-root /kaggle/input --out /kaggle/working/erase \
        --run-ids bertL-mnliB-16k-s111 bertL-mnliB-16k-s22 --stages lout,lmid

Mỗi mô hình:
  1. Dựng tập fit khái niệm (sets.py, tất định, seed 1234) — giống nhau cho mọi mô hình.
  2. Trích pooler_output + [CLS] các tầng --cls-layers cho các ví dụ MNLI-POOL cần dùng (c_LO + các mẫu lặp, MNLI-HC,
     c_LABEL_40K) — GPU. HANS-EVAL, HT-B, DEV-M lấy từ cache P2 (std_*_pooler.npy, std_*_cls.npy).
  3. stage `lmid`: erasure tại [CLS] tầng giữa (layers.py) — GPU.
  4. stage `lout`: mỗi khái niệm: LEACE fit trên phần "fit"; đo ΔRG (3 heuristic), ΔBalAcc, ΔAUROC, ΔP̄E, Δmargin, flip trên
     HANS-EVAL; ΔAcc DEV-M (toàn bộ, LO−); ΔRG trên MNLI-HC; probe-after (LR); giải mã được trước khi xoá (held-out ngẫu
     nhiên và held-out theo template); chuyển miền c_LO -> HANS; kiểm bảo đảm (chênh logit giữa hai lớp khái niệm).
     Null: N3 hoán vị nhãn khái niệm trong từng ô (fit lại đúng estimator) và N1 hướng ngẫu nhiên cùng rank, mỗi loại
     N_NULL lần; p một phía; Holm trên 9 ô (c_LO, c_SUB, c_CON × 3 cột) cho từng mô hình.
  5. Tổng hợp qua seed: quy tắc "vững" SPEC_ERASURE §7.3 ở mức MVP (mọi seed đã chạy).

NGƯỠNG (SPEC_ERASURE v0.1 §9, cố định TRƯỚC pilot 1; ghi 10/11/2026 — KHÔNG đổi ở pilot 2):
  δ_min = 0.05; thiệt hại MNLI |ΔAcc DEV-M| ≤ 2.0 điểm và ΔAcc DEV-M LO− ≥ −2.0 điểm; probe-after LR BalAcc ≤ 0.55;
  control dương c_LABEL: DEV-M acc sau xoá ∈ [0.45, 0.60]; G3: BalAcc giải mã ≥ 0.70, c_LO -> HANS ≥ 0.80.
  Sửa so với v0.1 TRƯỚC pilot 1: N_NULL = 1000 (với 100 lần, p nhỏ nhất 0.0099 > ngưỡng Holm đầu tiên 0.0056); probe LR
  (StandardScaler, C = 1) thay vì chọn C bằng CV. Chưa có: MP/INLP, N2, re-injection, xoá tuần tự.

THAY ĐỔI SAU PILOT 1 (commit 084eaf7) — 10/11/2026, SỬA SAU KHI THẤY DỮ LIỆU (ghi rõ để không lẫn với tiền đăng ký):
  (a) c_LO: ghép thêm láng giềng gần nhất theo (prem_len, hyp_len) trong tầng (sets.py), vì pilot 1 có |SMD| prem_len
      = 0.232 > 0.1 (vi phạm tiêu chí tiền đăng ký SPEC_DULIEU §6.1 bước 5). Việc ghép không dùng bất kỳ đầu ra mô hình nào.
  (b) Control dương: GIỮ tiêu chí v0.1 và báo cáo đúng tiêu chí đó; KHÔNG đặt ngưỡng mới. Thêm chẩn đoán:
      chênh logit giữa hai lớp trên tập fit (bảo đảm toán học, phải ≈ 0) và trên held-out (sai số ước lượng ngoài mẫu);
      c_LABEL_40K (cùng quy trình, 40K ví dụ) để xem phần còn lại có giảm theo cỡ mẫu không.
      Lý do: pilot 1 cho chênh logit trên fit ≈ 1e-12 nhưng còn 1.0–1.65 (từ 16–21) trên held-out. Tiêu chí v0.1 ngầm giả
      định bảo đảm của LEACE giữ ngoài tập fit — SPEC_DULIEU §8 đã ghi bảo đảm chỉ giữ trên phân phối fit. Lỗi ở thiết kế
      tiêu chí, ghi nhận như vậy.
  (c) Cài các chỉ số SPEC_ERASURE §4.1, §6, §7.1 đã định trước nhưng pilot 1 chưa cài: P̄E_{h,E}, P̄E_{h,NE},
      margin = logit_E − logit_NE, Flip_{h,y}, ‖h′ − h‖. Null cho ΔP̄E_{h,NE}: p CHƯA hiệu chỉnh, chỉ để mô tả.
      Endpoint chính vẫn là ΔRG với quy tắc "vững" như cũ.
  (d) Cài SPEC_DULIEU §6.1 bước 7 / SPEC_ERASURE §4.5 (đã có trong SPEC, pilot 1 chưa cài): độ ổn định hướng c_LO qua 10
      mẫu (cosine trong không gian whitened chung) và SD của ΔRG do lấy mẫu tập fit.
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from khoaluan.data import load as L
from khoaluan.erase import sets as SETS
from khoaluan.erase.leace_fast import DT, FitStats, mean_diff_direction, stratified_permutation, var_removed

DELTA_MIN = 0.05
MNLI_DAMAGE = 2.0          # điểm %
PROBE_AFTER_MAX = 0.55
POS_CTRL = (0.45, 0.60)
G3_BALACC, G3_TRANSFER = 0.70, 0.80
COS_STABLE = 0.80          # SPEC_DULIEU §6.1 bước 7
ALPHA = 0.05
HEURS = (("lexical_overlap", "LO"), ("subsequence", "SUB"), ("constituent", "CON"))
POS_CONCEPTS = ["c_LABEL", "c_LABEL_40K"]
CONCEPTS = POS_CONCEPTS + ["c_LO", "c_SUB", "c_CON", "c_JOINT", "c_CTRL_H"]
HEUR_CONCEPTS = ["c_LO", "c_SUB", "c_CON", "c_JOINT", "c_CTRL_H"]
HOLM_FAMILY = ["c_LO", "c_SUB", "c_CON"]
MNLI_SET_CONCEPTS = ("c_LO", "c_LABEL", "c_LABEL_40K")
# Khuôn kỳ vọng (SPEC_DULIEU §5.3): ● có thể khác 0; ○ phải ≈ 0 (phép thử đặc hiệu)
EXPECTED = {"c_LO": "●●●", "c_SUB": "○●●", "c_CON": "○○●", "c_JOINT": "○●●", "c_CTRL_H": "○○○"}
NULL_COLS = ["dRG_LO", "dRG_SUB", "dRG_CON", "dDevAcc", "dPE_NE_LO", "dPE_NE_SUB", "dPE_NE_CON"]
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} phút] {msg}", flush=True)


# ------------------------------------------------------------------ chỉ số hành vi từ logits
def margin_of(logits):
    """margin = logit_E − logit_NE; dự đoán E khi margin ≥ 0 (cùng quy ước với pred = logit_NE > logit_E)."""
    logits = np.asarray(logits)
    return logits[:, 0] - logits[:, 1]


def hans_metrics(logits, heur, y, auroc=True):
    logits = np.asarray(logits)
    pred = (logits[:, 1] > logits[:, 0]).astype(int)
    corr = pred == y
    mg = margin_of(logits)
    pe = expit(mg)
    out = {}
    for h, tag in HEURS:
        m = heur == h
        mE, mN = m & (y == 0), m & (y == 1)
        e, ne = corr[mE].mean(), corr[mN].mean()
        out[f"RG_{tag}"], out[f"accE_{tag}"], out[f"accNE_{tag}"] = e - ne, e, ne
        out[f"BalAcc_{tag}"] = (e + ne) / 2
        out[f"PE_E_{tag}"], out[f"PE_NE_{tag}"] = float(pe[mE].mean()), float(pe[mN].mean())
        out[f"margin_E_{tag}"], out[f"margin_NE_{tag}"] = float(mg[mE].mean()), float(mg[mN].mean())
        if auroc:
            out[f"AUROC_{tag}"] = roc_auc_score((y[m] == 0).astype(int), mg[m])   # điểm "entailment" (đơn điệu với P(E))
    out["hans_acc"] = corr.mean()
    return out


def flip_rates(base_logits, after_logits, heur, y):
    pb = np.asarray(base_logits)[:, 1] > np.asarray(base_logits)[:, 0]
    pa = np.asarray(after_logits)[:, 1] > np.asarray(after_logits)[:, 0]
    out = {}
    for h, tag in HEURS:
        for yy, nm in ((0, "E"), (1, "NE")):
            m = (heur == h) & (y == yy)
            out[f"flip_{nm}_{tag}"] = float((pb[m] != pa[m]).mean())
    return out


def null_vector(logits, heur_masks):
    """Nhanh cho null: [RG × 3 heuristic, P̄E trên NE × 3]. heur_masks: list (maskE, maskNE) theo HEURS."""
    mg = logits[:, 0] - logits[:, 1]
    pred_ne = mg < 0
    rg = [(~pred_ne[mE]).mean() - pred_ne[mN].mean() for mE, mN in heur_masks]
    pe = expit(mg)
    return np.array(rg + [pe[mN].mean() for _, mN in heur_masks])


def rg_vector(logits, heur_masks):
    return null_vector(logits, heur_masks)[:3]


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
    check_uids(run_dir, name, frame)
    return X


def check_uids(run_dir, name, frame):
    with open(os.path.join(run_dir, f"std_{name}_uids.txt")) as f:
        uids = f.read().split("\n")
    if uids != frame.uid.tolist():
        raise ValueError(f"[DỪNG] thứ tự uid của cache std_{name} khác dữ liệu")


def _uid_hash(uids):
    return hashlib.sha256("\n".join(uids).encode()).hexdigest()[:16]


def extract_pool(model_dir, union, odir, batch, cls_layers, log=log):
    """Trích pooler_output + logits + [CLS] các tầng cls_layers cho các ví dụ MNLI-POOL cần dùng (max_length 128, fp16 như
    P2). Dùng lại cache nếu cùng danh sách uid và cùng tầng. Trả về (pooler float64, cls float16 (N, len(cls_layers), H))."""
    meta_p = os.path.join(odir, "pool_meta.json")
    paths = {k: os.path.join(odir, f"pool_{k}.npy") for k in ("pooler", "logits", "cls")}
    meta = {"n": len(union), "uid_sha": _uid_hash(union.uid.tolist()), "cls_layers": list(cls_layers)}
    if os.path.exists(meta_p) and all(os.path.exists(p) for p in paths.values()):
        if json.load(open(meta_p)) == meta:
            log(f"  dùng lại cache MNLI-POOL ({len(union)} ví dụ)")
            return np.load(paths["pooler"]).astype(np.float64), np.load(paths["cls"])
        log("  cache MNLI-POOL khác tập hiện tại -> trích lại")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from khoaluan.eval.forward import run_pairs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    t = time.time()
    r = run_pairs(model, tok, union, 128, batch, fp16=device.type == "cuda", cls_layers=list(cls_layers))
    np.save(paths["pooler"], r["pooler"].astype(np.float16))
    np.save(paths["logits"], r["logits"])
    np.save(paths["cls"], r["cls"].astype(np.float16))
    json.dump(meta, open(meta_p, "w"))
    log(f"  trích MNLI-POOL: {len(union)} ví dụ, tầng CLS {list(cls_layers)}, {(time.time() - t) / 60:.1f} phút ({device})")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.load(paths["pooler"]).astype(np.float64), np.load(paths["cls"])


def probe_balacc(Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2 or len(yte) == 0:
        return float("nan"), None
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    clf.fit(Xtr, ytr)
    return float(balanced_accuracy_score(yte, clf.predict(Xte))), clf


def class_gap(d, mask, z, k):
    """Chênh (max − min) của trung bình d giữa các lớp khái niệm, trên mask."""
    return float(np.ptp([d[mask & (z == c)].mean() for c in range(k)]))


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


def bh(p):
    """Benjamini–Hochberg q-value."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        q[i] = running
    return q


def p_one_sided(null, obs):
    """p một phía (giảm): (1 + #{null ≤ obs}) / (1 + B)."""
    null = np.asarray(null)
    return float((1 + (null <= obs).sum()) / (1 + len(null)))


# ------------------------------------------------------------------ một mô hình
def analyze_model(run_id, run_dir, data_dir, sets, union, pool_X, n_null, method, seed, log=log):
    W, b = load_head(os.path.join(run_dir, "model"))
    W_np, b_np = W.numpy(), b.numpy()
    fr = {s: L.eval_frame(data_dir, s) for s in ("hans_eval", "hans_train_B", "dev_m")}
    X = {s: load_std(run_dir, s, fr[s]) for s in fr}
    # head(cache fp16) phải tái tạo dự đoán đã lưu ở P2
    checks = {}
    for s in ("hans_eval", "dev_m"):
        saved = pd.read_parquet(os.path.join(run_dir, f"preds_{s}.parquet"))
        rec = X[s] @ W_np.T + b_np
        agree = float((rec.argmax(1) == saved.pred.to_numpy()).mean())
        checks[f"head_reproduces_preds_{s}"] = agree
        if agree < 0.99:
            raise ValueError(f"[DỪNG] {run_id}: head trên cache {s} chỉ khớp {agree:.4f} dự đoán đã lưu")
    pool_logits = pool_X @ W_np.T + b_np
    base_logits = {s: X[s] @ W_np.T + b_np for s in ("hans_eval", "dev_m")}
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
    base_null = null_vector(base_logits["hans_eval"], masks)
    D_hans = torch.as_tensor(X["hans_eval"], dtype=DT)
    D_dev = torch.as_tensor(X["dev_m"], dtype=DT)
    D_hc = torch.as_tensor(pool_X[hc_i], dtype=DT)
    htb_idx = {u: i for i, u in enumerate(fr["hans_train_B"].uid)}
    cells, concepts_out, nulls = [], {}, {}
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    fitted = {}

    def effect(stats, er):
        """Hành vi sau xoá (dùng head tuyến tính: logit' = logit + Δlogit)."""
        sh = {k: er.logit_shift(D - stats.mu, W).numpy() for k, D in (("h", D_hans), ("d", D_dev), ("c", D_hc))}
        lg = (base_logits["hans_eval"] + sh["h"], base_logits["dev_m"] + sh["d"], pool_logits[hc_i] + sh["c"])
        return lg, behaviour(*lg)

    for cname in CONCEPTS:
        if cname not in sets:
            continue
        df = sets[cname]
        if cname in MNLI_SET_CONCEPTS:
            ii = [pidx[u] for u in df.uid]
            Xc, lg_c = pool_X[ii], pool_logits[ii]
        else:
            ii = [htb_idx[u] for u in df.uid]
            Xc = X["hans_train_B"][ii]
            lg_c = Xc @ W_np.T + b_np
        z = df.z.to_numpy()
        fit = (df.split == "fit").to_numpy()
        held = ~fit
        k = int(z.max()) + 1
        stats = FitStats(Xc[fit], method=method)
        er = stats.eraser(z[fit])
        fitted[cname] = (stats, er)
        info = {"n_fit": int(fit.sum()), "n_held": int(held.sum()), "k": k, "rank": er.rank,
                "rank_expected": k - 1, "var_removed": var_removed(stats, er)}
        # giải mã được trước khi xoá (G3) + probe-after
        info["decode_balacc_held"], probe = probe_balacc(Xc[fit], z[fit], Xc[held], z[held])
        Xe = er(Xc)
        info["probe_after_balacc_held"] = probe_balacc(Xe[fit], z[fit], Xe[held], z[held])[0]
        dn = np.linalg.norm(Xe - Xc, axis=1)
        info["mean_shift_norm"] = float(dn.mean())
        info["mean_shift_rel"] = float(dn.mean() / np.linalg.norm(Xc - Xc[fit].mean(0), axis=1).mean())
        if "tmpl_held" in df.columns and df.tmpl_held.any():
            th = df.tmpl_held.to_numpy()
            info["decode_balacc_template_heldout"] = probe_balacc(Xc[~th], z[~th], Xc[th], z[th])[0]
        if cname == "c_LO":
            info["transfer_HANS_LOplus_rate"] = float(probe.predict(X["hans_eval"]).mean())
        # kiểm bảo đảm: chênh margin (E − NE) trung bình giữa các lớp khái niệm
        dz0 = margin_of(lg_c)
        dz1 = margin_of(Xe @ W_np.T + b_np)
        info["logit_gap_between_classes_before_held"] = class_gap(dz0, held, z, k)
        info["logit_gap_between_classes_after_held"] = class_gap(dz1, held, z, k)
        info["logit_gap_between_classes_after_fit"] = class_gap(dz1, fit, z, k)
        info["logit_gap_residual_ratio_held"] = (info["logit_gap_between_classes_after_held"]
                                                 / max(info["logit_gap_between_classes_before_held"], 1e-12))
        # hiệu ứng quan sát
        lg_after, after = effect(stats, er)
        delta = {kk: after[kk] - base[kk] for kk in base}
        info["after"], info["delta"] = after, delta
        info["flip"] = flip_rates(base_logits["hans_eval"], lg_after[0], heur, yh)
        info["mnli_damage_ok"] = bool(abs(delta["devm_acc"]) * 100 <= MNLI_DAMAGE
                                      and delta["devm_acc_LOminus"] * 100 >= -MNLI_DAMAGE)
        if cname in POS_CONCEPTS:
            info["positive_control_ok"] = bool(POS_CTRL[0] <= after["devm_acc"] <= POS_CTRL[1])   # tiêu chí v0.1
            info["role"] = "control dương tiền đăng ký" if cname == "c_LABEL" else "chẩn đoán thêm sau pilot 1"
            concepts_out[cname] = info
            log(f"  {cname} (n_fit {info['n_fit']}): DEV-M sau xoá {after['devm_acc']:.3f} (tiêu chí v0.1 "
                f"{'ĐẠT' if info['positive_control_ok'] else 'KHÔNG ĐẠT'}); chênh logit fit "
                f"{info['logit_gap_between_classes_after_fit']:.1e}, held {info['logit_gap_between_classes_before_held']:.2f}"
                f" -> {info['logit_gap_between_classes_after_held']:.2f}")
            continue
        # null N3 (hoán vị trong ô, fit lại) và N1 (ngẫu nhiên cùng rank)
        strata = df.stratum.to_numpy()[fit]
        Dh, Dd = D_hans - stats.mu, D_dev - stats.mu
        res = {"N3": np.zeros((n_null, len(NULL_COLS))), "N1": np.zeros((n_null, len(NULL_COLS)))}
        for i in range(n_null):
            for kind in ("N3", "N1"):
                e = stats.eraser(stratified_permutation(z[fit], strata, rng)) if kind == "N3" \
                    else stats.random_eraser(k - 1, gen)
                hl = base_logits["hans_eval"] + e.logit_shift(Dh, W).numpy()
                dl = base_logits["dev_m"] + e.logit_shift(Dd, W).numpy()
                v = null_vector(hl, masks) - base_null
                res[kind][i, :3], res[kind][i, 4:] = v[:3], v[3:]
                res[kind][i, 3] = acc(dl, ydev) - base["devm_acc"]
        nulls[cname] = res
        for j, (_, tag) in enumerate(HEURS):
            obs = delta[f"RG_{tag}"]
            row = {"run_id": run_id, "position": "L_out", "concept": cname, "column": tag,
                   "expected": EXPECTED[cname][j], "RG_base": base[f"RG_{tag}"], "dRG": obs,
                   "dBalAcc": delta[f"BalAcc_{tag}"], "dAUROC": delta[f"AUROC_{tag}"], "dAccE": delta[f"accE_{tag}"],
                   "dAccNE": delta[f"accNE_{tag}"], "dPE_E": delta[f"PE_E_{tag}"], "dPE_NE": delta[f"PE_NE_{tag}"],
                   "margin_NE_base": base[f"margin_NE_{tag}"], "dMargin_E": delta[f"margin_E_{tag}"],
                   "dMargin_NE": delta[f"margin_NE_{tag}"], "flip_E": info["flip"][f"flip_E_{tag}"],
                   "flip_NE": info["flip"][f"flip_NE_{tag}"]}
            for kind in ("N3", "N1"):
                nv = res[kind][:, j]
                row[f"p_{kind}"] = p_one_sided(nv, obs)
                row[f"q05_{kind}"] = float(np.quantile(nv, 0.05))
                row[f"mean_{kind}"] = float(nv.mean())
                row[f"sd_{kind}"] = float(nv.std(ddof=1))
                npe = res[kind][:, 4 + j]
                row[f"p_{kind}_PE_NE"] = p_one_sided(npe, delta[f"PE_NE_{tag}"])
                row[f"q05_{kind}_PE_NE"] = float(np.quantile(npe, 0.05))
            cells.append(row)
        for kind in ("N3", "N1"):
            info[f"null_{kind}_dDevAcc_q025_q975"] = [float(np.quantile(res[kind][:, 3], q)) for q in (0.025, 0.975)]
        concepts_out[cname] = info
        log(f"  {cname}: rank {er.rank}/{k - 1}, giải mã {info['decode_balacc_held']:.3f}, sau xoá "
            f"{info['probe_after_balacc_held']:.3f} | ΔRG LO/SUB/CON "
            + "/".join(f"{delta[f'RG_{t}']:+.3f}" for _, t in HEURS) + " | ΔP̄E_NE "
            + "/".join(f"{delta[f'PE_NE_{t}']:+.3f}" for _, t in HEURS) + f" | ΔDEV-M {delta['devm_acc'] * 100:+.2f} điểm")
    cells = pd.DataFrame(cells)
    fam = cells.concept.isin(HOLM_FAMILY)
    cells["p_holm"] = np.nan
    cells.loc[fam, "p_holm"] = holm(cells.loc[fam, "p_N3"].to_numpy())
    cells.loc[~fam, "p_holm"] = cells.loc[~fam, "p_N3"]          # ngoài họ Holm: p chưa hiệu chỉnh
    cells["cell_robust_this_seed"] = ((cells.dRG <= -DELTA_MIN) & (cells.p_holm < ALPHA)
                                      & (cells.dRG < cells.q05_N3))
    out = {"run_id": run_id, "checks": checks, "base": base, "concepts": concepts_out}
    if "c_LO" in fitted and "c_LO_resamples" in sets and len(sets["c_LO_resamples"]):
        out["c_LO_stability"] = c_lo_stability(sets, pidx, pool_X, fitted["c_LO"][0], method, effect, base, log)
    return out, cells, nulls


def c_lo_stability(sets, pidx, pool_X, stats_main, method, effect, base, log=log):
    """SPEC_DULIEU §6.1 bước 7 / SPEC_ERASURE §4.5: lặp fit c_LO trên các mẫu lấy lại (mẫu 0 = mẫu chính)."""
    groups = [(0, sets["c_LO"])] + list(sets["c_LO_resamples"].groupby("resample"))
    dirs, rows = [], []
    for r, df in groups:
        Xr = pool_X[[pidx[u] for u in df.uid]]
        z = df.z.to_numpy()
        fit = (df.split == "fit").to_numpy()
        st = FitStats(Xr[fit], method=method)
        er = st.eraser(z[fit])
        dirs.append(mean_diff_direction(Xr[fit], z[fit], stats_main.W))
        _, after = effect(st, er)
        rows.append({"resample": int(r), **{f"dRG_{t}": after[f"RG_{t}"] - base[f"RG_{t}"] for _, t in HEURS},
                     **{f"dPE_NE_{t}": after[f"PE_NE_{t}"] - base[f"PE_NE_{t}"] for _, t in HEURS},
                     "dDevAcc": after["devm_acc"] - base["devm_acc"]})
    D = np.array(dirs)
    cos = D @ D.T
    off = cos[~np.eye(len(D), dtype=bool)]
    t = pd.DataFrame(rows)
    out = {"n_samples": len(groups), "cos_median": float(np.median(off)), "cos_min": float(off.min()),
           "cos_with_main_median": float(np.median(cos[0, 1:])), "unstable_flag": bool(np.median(off) < COS_STABLE),
           "per_sample": rows,
           "sd_over_samples": {c: float(t[c].std(ddof=1)) for c in t.columns if c != "resample"},
           "mean_over_samples": {c: float(t[c].mean()) for c in t.columns if c != "resample"}}
    log(f"  c_LO ổn định qua {len(groups)} mẫu: cosine trung vị {out['cos_median']:.3f} (min {out['cos_min']:.3f}); "
        f"ΔRG LO TB {out['mean_over_samples']['dRG_LO']:+.3f} ± {out['sd_over_samples']['dRG_LO']:.3f}")
    return out


# ------------------------------------------------------------------ tổng hợp qua seed
def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def family(run_id):
    import re
    return re.sub(r"-s\d+$", "", run_id)


def _mean(rows, key):
    v = [r.get(key, np.nan) for r in rows]
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def summarize(models, cells, sets_report, n_null, method):
    """Báo cáo theo từng họ mô hình (bertL-mnliB-16k, bertL-mix-16k, ...): đếm seed 'vững' KHÔNG trộn giữa các họ."""
    fams = sorted({family(m["run_id"]) for m in models})
    if len(fams) > 1:
        return "\n\n".join(summarize([m for m in models if family(m["run_id"]) == f],
                                      cells[cells.run_id.map(family) == f], sets_report, n_null, method)
                            for f in fams)
    n = len(models)
    rm = sets_report["mnli"]
    L_ = [f"# {fams[0]}: erasure tại L_out ({method.upper()}) — {n} mô hình, null {n_null} lần mỗi loại\n"]
    L_.append("## 1. Kiểm tra bắt buộc")
    for m in models:
        c = m["concepts"]
        L_.append(f"- {m['run_id']}: head tái tạo dự đoán P2 HANS {m['checks']['head_reproduces_preds_hans_eval']:.4f}"
                  f" / DEV-M {m['checks']['head_reproduces_preds_dev_m']:.4f}")
    L_.append(f"- Tập c_LO: n* = {rm['n_star_LO+C_fit']}, tổng {rm['c_LO_total']} (fit {rm['c_LO_fit']}); ghép: "
              f"{rm.get('matching', 'ngẫu nhiên trong tầng (pilot 1)')}; |SMD| lớn nhất {rm['SMD_max_abs']:.3f} "
              f"(cần < 0.1; trước ghép {rm.get('SMD_max_abs_before', float('nan')):.3f}); MNLI-HC {rm['MNLI_HC']}")
    L_.append("\n## 2. Control dương (tiêu chí v0.1: DEV-M sau xoá ∈ [0.45, 0.60]; cột chẩn đoán thêm sau pilot 1)")
    L_.append("| Mô hình | Khái niệm | n_fit | DEV-M trước → sau | tiêu chí v0.1 | chênh logit fit (≈0 theo toán) | "
              "chênh logit held trước → sau (tỉ lệ còn lại) | probe-after |")
    L_.append("|---|---|---|---|---|---|---|---|")
    for m in models:
        for cn in POS_CONCEPTS:
            c = m["concepts"].get(cn)
            if not c:
                continue
            nan = float("nan")
            L_.append(f"| {m['run_id']} | {cn} | {c.get('n_fit', '—')} | {m['base']['devm_acc']:.3f} → "
                      f"{c['after']['devm_acc']:.3f} | {'ĐẠT' if c['positive_control_ok'] else 'KHÔNG ĐẠT'} | "
                      f"{c.get('logit_gap_between_classes_after_fit', nan):.1e} | "
                      f"{c.get('logit_gap_between_classes_before_held', nan):.2f} → "
                      f"{c.get('logit_gap_between_classes_after_held', nan):.2f}"
                      f" ({c.get('logit_gap_residual_ratio_held', nan):.3f}) | {c.get('probe_after_balacc_held', nan):.3f} |")
    L_.append("Đọc: chênh trên fit ≈ 0 xác nhận phép xoá đúng toán học; phần còn lại trên held-out là sai số ước lượng ngoài "
              "mẫu. Nếu c_LABEL_40K giảm rõ phần còn lại so với c_LABEL 10K -> nguyên nhân là cỡ mẫu, không phải pipeline.")
    L_.append("\n## 3. Khái niệm: giải mã được (G3) và xoá có hiệu lực (TB qua seed)")
    L_.append("| Khái niệm | rank | giải mã held | giải mã template mới | sau xoá (≤0.55) | VarRemoved | ‖h′−h‖ tương đối | "
              "ΔDEV-M (điểm) | ΔDEV-M LO− | ΔRG MNLI-HC | khác |")
    L_.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for cn in CONCEPTS:
        rows = [m["concepts"][cn] for m in models if cn in m["concepts"]]
        if not rows:
            continue
        def gd(key, rows=rows):
            return float(np.mean([r["delta"][key] for r in rows]))
        extra = ""
        if cn == "c_LO":
            extra = f"c_LO→HANS LO+ {_mean(rows, 'transfer_HANS_LOplus_rate'):.2f} (cần ≥ 0.80)"
        if cn in POS_CONCEPTS:
            extra = f"DEV-M sau xoá {np.mean([r['after']['devm_acc'] for r in rows]):.3f}"
        L_.append(f"| {cn} | {_mean(rows, 'rank'):.0f}/{rows[0]['rank_expected']} | {_f(_mean(rows, 'decode_balacc_held'))} | "
                  f"{_f(_mean(rows, 'decode_balacc_template_heldout'))} | {_f(_mean(rows, 'probe_after_balacc_held'))} | "
                  f"{_mean(rows, 'var_removed'):.4f} | {_f(_mean(rows, 'mean_shift_rel'))} | {gd('devm_acc') * 100:+.2f} | "
                  f"{gd('devm_acc_LOminus') * 100:+.2f} | {gd('RG_HC'):+.3f} | {extra} |")
    L_.append("\n## 4. Ma trận cross-effect (ENDPOINT CHÍNH): ΔRG trên HANS (TB ± SD qua seed) | số seed 'vững' / tổng")
    L_.append("'vững' ở một seed = ΔRG ≤ −0.05 VÀ p (N3, Holm trong họ c_LO/c_SUB/c_CON) < 0.05 VÀ thấp hơn phân vị 5% của N3.")
    L_.append("| Khái niệm | kỳ vọng LO/SUB/CON | cột LO | cột SUB | cột CON |")
    L_.append("|---|---|---|---|---|")
    for cn in HEUR_CONCEPTS:
        parts = []
        for tag in ("LO", "SUB", "CON"):
            s = cells[(cells.concept == cn) & (cells.column == tag)]
            if not len(s):
                parts.append("—")
                continue
            parts.append(f"{s.dRG.mean():+.3f} ± {s.dRG.std(ddof=1) if len(s) > 1 else 0:.3f} | "
                         f"{int(s.cell_robust_this_seed.sum())}/{len(s)} "
                         f"(p_N3 max {s.p_N3.max():.3f}; N3 q05 TB {s.q05_N3.mean():+.3f}; N1 q05 TB {s.q05_N1.mean():+.3f})")
        L_.append(f"| {cn} | {EXPECTED[cn]} | " + " | ".join(parts) + " |")
    if "dPE_NE" in cells.columns:
        L_.append("\n## 5. Chỉ số phụ (mô tả, KHÔNG dùng để kết luận): trên các ví dụ NE của HANS")
        L_.append("ΔP̄E_NE < 0 = mô hình bớt tin 'entailment' ở ví dụ ngược heuristic (đúng hướng). p_N3 chưa hiệu chỉnh. "
                  "Δmargin so với margin gốc cho biết còn cách ranh giới quyết định bao xa.")
        L_.append("| Khái niệm | cột | ΔP̄E_NE (TB seed) | p_N3 max qua seed | margin_NE gốc | Δmargin_NE | ΔP̄E_E | flip NE | flip E |")
        L_.append("|---|---|---|---|---|---|---|---|---|")
        for cn in HEUR_CONCEPTS:
            for tag in ("LO", "SUB", "CON"):
                s = cells[(cells.concept == cn) & (cells.column == tag)]
                if not len(s):
                    continue
                L_.append(f"| {cn} | {tag} | {s.dPE_NE.mean():+.4f} | {s.p_N3_PE_NE.max():.3f} | {s.margin_NE_base.mean():+.2f}"
                          f" | {s.dMargin_NE.mean():+.3f} | {s.dPE_E.mean():+.4f} | {s.flip_NE.mean():.4f} | {s.flip_E.mean():.4f} |")
    L_.append("\n## 6. Chiều thay đổi trên HANS (TB qua seed): ΔAccE / ΔAccNE / ΔAUROC")
    for cn in HEUR_CONCEPTS:
        s = cells[cells.concept == cn].groupby("column")[["dAccE", "dAccNE", "dAUROC"]].mean()
        if not len(s):
            continue
        L_.append(f"- {cn}: " + "; ".join(f"{t} {s.loc[t, 'dAccE']:+.3f}/{s.loc[t, 'dAccNE']:+.3f}/{s.loc[t, 'dAUROC']:+.3f}"
                                         for t in ("LO", "SUB", "CON")))
    stab = [m["c_LO_stability"] for m in models if m.get("c_LO_stability")]
    if stab:
        L_.append(f"\n## 7. Độ ổn định hướng c_LO qua {stab[0]['n_samples']} mẫu tập fit (SPEC_DULIEU §6.1 bước 7; cần cosine "
                  f"trung vị ≥ {COS_STABLE})")
        L_.append("| Mô hình | cosine trung vị | cosine min | ΔRG LO / SUB / CON: TB ± SD qua mẫu | ΔP̄E_NE LO TB ± SD |")
        L_.append("|---|---|---|---|---|")
        for m in models:
            s = m.get("c_LO_stability")
            if not s:
                continue
            mo, sd = s["mean_over_samples"], s["sd_over_samples"]
            L_.append(f"| {m['run_id']} | {s['cos_median']:.3f}{' (KHÔNG ổn định)' if s['unstable_flag'] else ''} | "
                      f"{s['cos_min']:.3f} | " + " / ".join(f"{mo[f'dRG_{t}']:+.3f} ± {sd[f'dRG_{t}']:.3f}"
                                                         for t in ("LO", "SUB", "CON"))
                      + f" | {mo['dPE_NE_LO']:+.4f} ± {sd['dPE_NE_LO']:.4f} |")
    L_.append("\n## 8. Baseline (trước xoá, TB qua seed)")
    keys = ["RG_LO", "RG_SUB", "RG_CON", "devm_acc", "RG_HC", "PE_NE_LO", "PE_NE_SUB", "PE_NE_CON",
            "margin_NE_LO", "margin_NE_SUB", "margin_NE_CON"]
    L_.append(", ".join(f"{k} {np.mean([m['base'][k] for m in models]):.3f}" for k in keys if k in models[0]["base"]))
    L_.append("\nGhi chú: ngưỡng SPEC_ERASURE v0.1 §9 giữ nguyên từ pilot 1; các thay đổi sau pilot 1 ghi ở đầu "
              "src/khoaluan/erase/pilot.py và sets.py. Chưa có MP/INLP, N2, re-injection, xoá tuần tự, M-MIX.")
    return "\n".join(L_)


# ------------------------------------------------------------------ main
def parse_int_list(s):
    return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-ids", nargs="+", required=True)
    ap.add_argument("--stages", default="lout", help="lout | lmid | lout,lmid")
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--method", choices=["leace", "orth"], default="leace")
    ap.add_argument("--n-label", type=int, default=10000)
    ap.add_argument("--n-label-big", type=int, default=40000)
    ap.add_argument("--n-resample", type=int, default=SETS.N_RESAMPLE)
    ap.add_argument("--cls-layers", default="20,22,24", help="tầng CLS trích cho MNLI-POOL (phải chứa mọi --layers)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--allow-unofficial-data", action="store_true")
    # L_ℓ (layers.py)
    ap.add_argument("--layers", default="20,22,24")
    ap.add_argument("--all-token-layers", default="20,22")
    ap.add_argument("--null-layers", default="22,24")
    ap.add_argument("--n-null-last", type=int, default=1000, help="null N3 và N1 tại tầng cuối (chỉ pooler, rẻ)")
    ap.add_argument("--n-null-mid-n3", type=int, default=500)
    ap.add_argument("--n-null-mid-n1", type=int, default=100)
    ap.add_argument("--null-subset-per-subcase", type=int, default=200)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    stages = {s.strip() for s in a.stages.split(",") if s.strip()}
    if not stages or stages - {"lout", "lmid"}:
        raise SystemExit(f"[DỪNG] --stages không hợp lệ: {a.stages}")
    runs_root = a.runs_root or a.data_root
    os.makedirs(a.out, exist_ok=True)
    data_dir = L.find_data_dir(a.data_root)
    if not a.allow_unofficial_data:
        L.verify_data(data_dir)
        log(f"dữ liệu khớp lần chạy A-06 chính thức: {data_dir}")
    run_dirs = {r: find_run_dir(runs_root, r) for r in a.run_ids}       # kiểm đủ mô hình trước khi làm gì
    log("tìm thấy cache P2: " + "; ".join(f"{k} -> {v}" for k, v in run_dirs.items()))
    cls_layers = parse_int_list(a.cls_layers)
    if "lmid" in stages and not set(parse_int_list(a.layers)) <= set(cls_layers):
        raise SystemExit("[DỪNG] --cls-layers phải chứa mọi tầng trong --layers")
    sets, union, rep = SETS.build_all(data_dir, seed=a.seed, n_label=a.n_label, n_label_big=a.n_label_big,
                                      n_resample=a.n_resample)
    SETS.save_sets(sets, union, rep, os.path.join(a.out, "sets"))
    log(f"tập fit: c_LO {rep['mnli']['c_LO_total']} (n* {rep['mnli']['n_star_LO+C_fit']}), |SMD| max "
        f"{rep['mnli']['SMD_max_abs']:.3f} (trước ghép {rep['mnli']['SMD_max_abs_before']:.3f}), MNLI-POOL cần trích "
        f"{len(union)}")
    if rep["mnli"]["SMD_max_abs"] >= 0.1:
        log(f"CẢNH BÁO: |SMD| = {rep['mnli']['SMD_max_abs']:.3f} vẫn ≥ 0.1 sau ghép láng giềng gần nhất")
    models, all_cells, lmid_models, lmid_cells = [], [], [], []
    for rid, rdir in run_dirs.items():
        log(f"=== {rid}")
        odir = os.path.join(a.out, rid)
        os.makedirs(odir, exist_ok=True)
        pool_X, pool_cls = extract_pool(os.path.join(rdir, "model"), union, odir, a.batch, cls_layers)
        if "lmid" in stages:
            from khoaluan.erase import layers as LAY
            res_l, cells_l = LAY.analyze_model_layers(rid, rdir, data_dir, sets, union, pool_cls, cls_layers, a, log=log)
            cells_l.to_csv(os.path.join(odir, "lmid_cells.csv"), index=False)
            with open(os.path.join(odir, "lmid_model.json"), "w") as f:
                json.dump(res_l, f, indent=2, ensure_ascii=False, default=float)
            lmid_models.append(res_l)
            lmid_cells.append(cells_l)
        if "lout" in stages:
            res, cells, nulls = analyze_model(rid, rdir, data_dir, sets, union, pool_X, a.n_null, a.method, a.seed)
            cells.to_csv(os.path.join(odir, "erase_cells.csv"), index=False)
            np.savez_compressed(os.path.join(odir, "nulls.npz"),
                                **{f"{c}_{k}": v for c, d in nulls.items() for k, v in d.items()})
            with open(os.path.join(odir, "erase_model.json"), "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False, default=float)
            models.append(res)
            all_cells.append(cells)
    cells = None
    if models:
        cells = pd.concat(all_cells, ignore_index=True)
        cells.to_csv(os.path.join(a.out, "erase_cells_all.csv"), index=False)
        text = summarize(models, cells, rep, a.n_null, a.method)
        with open(os.path.join(a.out, "erase_summary.md"), "w") as f:
            f.write(text)
        print(text)
    if lmid_models:
        from khoaluan.erase import layers as LAY
        lc = pd.concat(lmid_cells, ignore_index=True)
        lc.to_csv(os.path.join(a.out, "lmid_cells_all.csv"), index=False)
        text = LAY.summarize_layers(lmid_models, lc)
        with open(os.path.join(a.out, "lmid_summary.md"), "w") as f:
            f.write(text)
        print(text)
    log(f"XONG. Kết quả: {a.out}/erase_summary.md, lmid_summary.md, *_cells_all.csv, sets/sets_report.json")
    return models, cells


if __name__ == "__main__":
    main()
