"""
layers.py — erasure tại L_ℓ = [CLS] của hidden_states[ℓ] (SPEC_ERASURE v0.1 §2.2) — pilot 2. Gọi từ pilot.main (--stages lmid).

Quy ước tầng (root §3.7): hidden_states[0] = embedding, hidden_states[ℓ] = đầu ra tầng Transformer thứ ℓ = ĐẦU VÀO của
encoder.layer[ℓ] (chỉ số từ 0); ℓ = L (tầng cuối, 24 với BERT-large) = đầu vào pooler.

Can thiệp:
  - mode "cls": thay h_ℓ[:, 0] bằng r(h_ℓ[:, 0]) rồi chạy tiếp encoder.layer[ℓ..L−1] → pooler → classifier (SPEC §2.2).
  - mode "all" (THĂM DÒ, SPEC §2.2): áp CÙNG eraser fit trên [CLS] cho mọi vị trí không phải padding. Chỉ ℓ < L.
  r = LEACE affine fit trên [CLS] tầng ℓ của tập fit khái niệm (cache fp16 của P2 cho HT-B; trích mới cho MNLI-POOL).
Cài đặt: mỗi lô chạy MỘT lượt forward sạch; hook trước encoder.layer[ℓ] ghi lại đầu vào và mọi tham số khác của tầng (mask
attention, ...), rồi với từng eraser chạy tiếp từ trạng thái đã ghi. Kết quả trùng cách làm bằng forward hook (SPEC §2.2) nhưng
không chạy lại phần đầu. Unit test (SPEC §11.4): eraser đồng nhất -> logits trùng forward gốc; eraser khác đồng nhất -> trùng
forward hook trên toàn mô hình; mode "all" không phụ thuộc padding.
ℓ = L: pooler chỉ đọc vị trí 0 nên tính thẳng từ cache [CLS]_L: logit = W_cls tanh(W_p x + b_p) + b_cls — không cần chạy
Transformer, null đủ --n-null-last lần trên toàn HANS-EVAL như L_out. Đây là phép thử kiểu Rozanova (xoá TRƯỚC pooler phi
tuyến) để so với L_out.

Đo cho mỗi (khái niệm, ℓ, mode): như L_out (ΔRG, ΔBalAcc, ΔAUROC, ΔP̄E, Δmargin, flip trên HANS-EVAL; ΔDEV-M, ΔDEV-M LO−;
ΔRG MNLI-HC) + phục hồi khái niệm ở pooler_output sau can thiệp (R-09: probe LR fit trên phần fit, đo trên held-out cùng
phân phối) + giải mã tại [CLS]_ℓ trước/sau xoá + chuẩn ‖h′ − h‖ (SPEC §4.1: theo dõi bùng nổ chuẩn; chưa cài
constrain_cov_trace).
Null: chỉ ở --null-layers (SPEC §5: ℓ ∈ {22, 24}); khái niệm c_LO, c_SUB, c_CON:
  ℓ = L: N3 + N1 mỗi loại --n-null-last lần, toàn HANS-EVAL + DEV-M.
  ℓ < L: N3 --n-null-mid-n3 lần, N1 --n-null-mid-n1 lần, trên HANS-EVAL rút gọn (--null-subset-per-subcase/subcase,
         SPEC §10); p tính bằng giá trị quan sát TRÊN CÙNG tập rút gọn.
  BH-FDR trên (tầng null × 3 khái niệm × 3 cột) cho mỗi mô hình (SPEC §7.2); thêm Holm trong từng tầng để so với L_out.
QUY TẮC ĐỌC (viết 10/11/2026, TRƯỚC khi có bất kỳ kết quả L_ℓ nào; chi tiết trong HUONGDAN_ERASE_PILOT2 §4.2):
  1. Kiểm tra: forward trực tiếp / pooler+head trên cache tái tạo dự đoán P2 ≥ 0.99; chạy tiếp không can thiệp lệch < 1e-3.
  2. Độ mạnh can thiệp tại ℓ: Δ_lab(ℓ) = DEV-M gốc − DEV-M sau khi xoá c_LABEL tại [CLS]_ℓ (mode cls).
     ≥ 0.30 "mạnh" (kết quả ≈ 0 của khái niệm heuristic ở ℓ có giá trị thông tin); 0.10–0.30 "một phần" ([GỢI Ý yếu]);
     < 0.10 "bị vá" (kết quả ≈ 0 ở ℓ KHÔNG dùng làm bằng chứng).
  3. Phục hồi ở pooler ≥ mốc − 0.05 -> phục hồi gần hoàn toàn; ≤ 0.55 -> không phục hồi.
  4. Ô "vững" (chỉ tầng có null): ΔRG ≤ −0.05, q_BH < 0.05, ΔRG trên tập null < q05(N3); kết luận cần 3/3 seed.
  5. Đặc hiệu: ô ○ vững -> khái niệm không đặc hiệu; c_CTRL_H |ΔRG| < 0.05; |ΔDEV-M| ≤ 2 điểm.
SỬA SO VỚI SPEC (ghi rõ, quyết định trước khi có kết quả L_ℓ): null tại ℓ < L tăng từ 20 lên 500 (N3) / 100 (N1) vì với
20 lần p nhỏ nhất là 1/21 ≈ 0.048, không thể < 0.05 sau BH trên 18 ô; với 500 lần p nhỏ nhất ≈ 0.002 (×18 = 0.036). N3 cũng
chạy tại L_ℓ (SPEC chỉ ghi N1) vì N3 là null chính để suy luận (SPEC §5).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from khoaluan.data import load as L
from khoaluan.erase.leace_fast import FitStats, stratified_permutation, var_removed
from khoaluan.erase.pilot import (ALPHA, DELTA_MIN, EXPECTED, HEURS, HOLM_FAMILY, MNLI_SET_CONCEPTS, POS_CONCEPTS, POS_CTRL,
                                  _f, acc, bh, check_uids, family, flip_rates, hans_metrics, holm, load_head, null_vector,
                                  p_one_sided, parse_int_list, probe_balacc)

L_CONCEPTS = POS_CONCEPTS + ["c_LO", "c_SUB", "c_CON", "c_JOINT", "c_CTRL_H"]
ALL_TOKEN_CONCEPTS = ["c_LABEL", "c_LO", "c_SUB", "c_CON"]
HTB_CONCEPTS = ["c_SUB", "c_CON", "c_JOINT", "c_CTRL_H"]


def _log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ can thiệp trong mô hình
def encoder_of(model):
    return getattr(model, "bert", model)


class Continuer:
    """Ghi trạng thái vào encoder.layer[ℓ] trong một lượt forward sạch, rồi chạy tiếp từ trạng thái (đã sửa) đó."""

    def __init__(self, model, layers):
        self.model = model
        self.enc = encoder_of(model)
        self.blocks = self.enc.encoder.layer
        self.L = len(self.blocks)
        self.layers = sorted(set(layers))
        if any(not 0 <= l < self.L for l in self.layers):
            raise ValueError(f"tầng tiếp tục phải trong [0, {self.L - 1}], nhận {self.layers}")
        self.cap, self.capturing = {}, False
        self.handles = [self.blocks[l].register_forward_pre_hook(self._hook(l), with_kwargs=True) for l in self.layers]

    def _hook(self, l):
        def hook(_mod, args, kwargs):
            if self.capturing:
                self.cap[l] = (args, kwargs)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _head(self, pooled):
        return self.model.classifier(self.model.dropout(pooled))

    @torch.no_grad()
    def prefix(self, batch):
        """Forward sạch; trả về (logits, pooled) gốc và ghi trạng thái vào các tầng đã đăng ký."""
        self.cap = {}
        self.capturing = True
        try:
            o = self.enc(**batch)
        finally:
            self.capturing = False
        missing = set(self.layers) - set(self.cap)
        if missing:
            raise RuntimeError(f"[DỪNG] không ghi được đầu vào tầng {sorted(missing)}")
        return self._head(o.pooler_output), o.pooler_output

    def _split(self, l):
        args, kwargs = self.cap[l]
        if args:
            return args[0], tuple(args[1:]), kwargs
        kw = dict(kwargs)
        return kw.pop("hidden_states"), (), kw

    def state(self, l):
        return self._split(l)[0]

    @torch.no_grad()
    def finish(self, l, h):
        _, rest, kw = self._split(l)
        for i in range(l, self.L):
            out = self.blocks[i](h, *rest, **kw)
            h = out[0] if isinstance(out, (tuple, list)) else out
        pooled = self.enc.pooler(h)
        return self._head(pooled), pooled


def apply_eraser(h, er, mode, mask=None):
    """h (B, T, d): mode 'cls' sửa vị trí 0; mode 'all' sửa mọi vị trí có mask = 1; 'none' giữ nguyên."""
    if mode == "none" or er is None:
        return h
    h2 = h.clone()
    if mode == "cls":
        h2[:, 0] = er(h[:, 0])
    elif mode == "all":
        m = mask.bool()
        h2[m] = er(h[m])
    else:
        raise ValueError(mode)
    return h2


def _autocast(device, fp16):
    import contextlib
    if fp16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def run_configs(model, tok, frame, configs, batch_size, fp16, keep_pooled=(), log=None, tag=""):
    """configs: list dict(name, layer, mode, eraser (TorchEraser hoặc None)). Mỗi lô: 1 forward sạch + 1 lần chạy tiếp cho
    mỗi cấu hình. Trả về dict: base_logits (N,2), base_pooled (N,H) f16, logits {name: (N,2)}, pooled {name: (N,H) f16},
    identity_max_abs_diff {ℓ: ...} (lô đầu: chạy tiếp không sửa so với forward gốc)."""
    from khoaluan.eval.forward import _batches, _pad
    device = next(model.parameters()).device
    model.eval()
    enc = tok(frame.premise.tolist(), frame.hypothesis.tolist(), truncation=True, max_length=128)
    enc = {k: enc[k] for k in ("input_ids", "token_type_ids", "attention_mask") if k in enc}
    n, hdim = len(frame), model.config.hidden_size
    layers = sorted({c["layer"] for c in configs})
    out = {"base_logits": np.zeros((n, 2), np.float32), "base_pooled": np.zeros((n, hdim), np.float16),
           "logits": {c["name"]: np.zeros((n, 2), np.float32) for c in configs},
           "pooled": {nm: np.zeros((n, hdim), np.float16) for nm in keep_pooled}, "identity_max_abs_diff": {}}
    t = time.time()
    with Continuer(model, layers) as cont:
        first = True
        for idx in _batches([len(x) for x in enc["input_ids"]], batch_size):
            b = _pad(tok, enc, idx, device)
            with _autocast(device, fp16):
                lg0, p0 = cont.prefix(b)
                if first:
                    for l in layers:
                        lg_id, _ = cont.finish(l, cont.state(l))
                        out["identity_max_abs_diff"][l] = float((lg_id.float() - lg0.float()).abs().max())
                    first = False
                out["base_logits"][idx] = lg0.float().cpu().numpy()
                out["base_pooled"][idx] = p0.float().cpu().numpy().astype(np.float16)
                buf = torch.empty((len(configs), len(idx), 2), device=device)     # gom trên GPU, chép về 1 lần/lô
                keep = {}
                for j, c in enumerate(configs):
                    h = apply_eraser(cont.state(c["layer"]), c["eraser"], c["mode"], b["attention_mask"])
                    lg, pooled = cont.finish(c["layer"], h)
                    buf[j] = lg.float()
                    if c["name"] in out["pooled"]:
                        keep[c["name"]] = pooled.half()
                buf = buf.cpu().numpy()
                for j, c in enumerate(configs):
                    out["logits"][c["name"]][idx] = buf[j]
                for nm, p in keep.items():
                    out["pooled"][nm][idx] = p.cpu().numpy()
    if log:
        log(f"    chạy tiếp {tag}: {n} ví dụ × {len(configs)} cấu hình, {(time.time() - t) / 60:.1f} phút")
    return out


# ------------------------------------------------------------------ tầng cuối: tính từ cache [CLS]_L
class Readout:
    """pooler + classifier đọc từ checkpoint. pre(x) = W_p x + b_p; logits = W tanh(pre) + b."""

    def __init__(self, model_dir, device, dtype=torch.float32):
        from safetensors import safe_open
        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
            keys = set(f.keys())
            pk = next(k for k in ("bert.pooler.dense.weight", "pooler.dense.weight") if k in keys)
            self.Wp = f.get_tensor(pk).to(device, dtype)
            self.bp = f.get_tensor(pk.replace("weight", "bias")).to(device, dtype)
        W, b = load_head(model_dir)
        self.W, self.b = W.to(device, dtype), b.to(device, dtype)
        self.device, self.dtype = device, dtype

    def t(self, X):
        return torch.as_tensor(np.asarray(X), device=self.device).to(self.dtype)

    def pre(self, X):
        return self.t(X) @ self.Wp.T + self.bp

    def from_pre(self, pre):
        pooled = torch.tanh(pre)
        return pooled @ self.W.T + self.b, pooled

    def erased_pre(self, pre0, Xt, er):
        """pre sau khi xoá x -> r(x) = x − ((x − μ) P_Rᵀ) P_Lᵀ: pre0 − ((x − μ) P_Rᵀ)(W_p P_L)ᵀ (cập nhật hạng thấp)."""
        return pre0 - ((Xt - er.mu) @ er.PR.T) @ (self.Wp @ er.PL).T


# ------------------------------------------------------------------ dữ liệu
def load_cls_cache(run_dir, name, frame, layers):
    meta = json.load(open(os.path.join(run_dir, "std_layers.json")))
    have = list(meta["cls_layers"])
    miss = [l for l in layers if l not in have]
    if miss:
        raise ValueError(f"[DỪNG] cache std_{name}_cls thiếu tầng {miss} (có {have})")
    check_uids(run_dir, name, frame)
    arr = np.load(os.path.join(run_dir, f"std_{name}_cls.npy"), mmap_mode="r")
    if arr.shape[0] != len(frame) or arr.shape[1] != len(have):
        raise ValueError(f"[DỪNG] std_{name}_cls.npy có shape {arr.shape}, cần ({len(frame)}, {len(have)}, H)")
    return {l: np.asarray(arr[:, have.index(l)], dtype=np.float32) for l in layers}


def null_subset(frame, per_subcase, seed=1234):
    """Chỉ số HANS-EVAL rút gọn: per_subcase ví dụ mỗi subcase (sắp theo uid rồi rút, tất định)."""
    rng = np.random.default_rng([seed, 9])
    idx = []
    for sc in sorted(frame.subcase.unique()):
        ii = frame.index[frame.subcase == sc].to_numpy()
        ii = ii[np.argsort(frame.uid.to_numpy()[ii], kind="stable")]
        idx += list(np.sort(rng.choice(ii, size=min(per_subcase, len(ii)), replace=False)))
    return np.array(sorted(idx))


# ------------------------------------------------------------------ một mô hình
def analyze_model_layers(run_id, run_dir, data_dir, sets, union, pool_cls, pool_layers, a, log=_log):
    """a: Namespace từ pilot.build_parser (layers, all_token_layers, null_layers, n_null_last, n_null_mid_n3, ...)."""
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = device.type == "cuda"
    model_dir = os.path.join(run_dir, "model")
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    Lmax = int(cfg["num_hidden_layers"])
    layers = parse_int_list(a.layers)
    all_layers = [l for l in parse_int_list(a.all_token_layers) if l in layers and l < Lmax]
    null_layers = [l for l in parse_int_list(a.null_layers) if l in layers]
    if any(l < 1 or l > Lmax for l in layers):
        raise ValueError(f"[DỪNG] --layers phải trong [1, {Lmax}]")
    mid = [l for l in layers if l < Lmax]
    rng = np.random.default_rng([a.seed, 5])
    gen = torch.Generator().manual_seed(a.seed + 5)
    dt_dev = torch.float32
    fr = {s: L.eval_frame(data_dir, s) for s in ("hans_eval", "hans_train_B", "dev_m")}
    cls = {s: load_cls_cache(run_dir, s, fr[s], layers) for s in fr}
    ppos = {l: list(pool_layers).index(l) for l in layers}
    pidx = {u: i for i, u in enumerate(union.uid)}
    htb_idx = {u: i for i, u in enumerate(fr["hans_train_B"].uid)}
    heur, yh = fr["hans_eval"].heuristic.to_numpy(), fr["hans_eval"].y.to_numpy()
    masks = [((heur == h) & (yh == 0), (heur == h) & (yh == 1)) for h, _ in HEURS]
    devm = pd.read_parquet(os.path.join(data_dir, "mnli_bin_dev_m.parquet"))[["uid", "lo"]].set_index("uid")
    dev_lo = devm.loc[fr["dev_m"].uid, "lo"].to_numpy()
    ydev = fr["dev_m"].y.to_numpy()
    hc = sets["MNLI_HC"]
    hc_y = hc.y.to_numpy()
    concepts = [c for c in L_CONCEPTS if c in sets]
    saved = {s: pd.read_parquet(os.path.join(run_dir, f"preds_{s}.parquet")).pred.to_numpy() for s in ("hans_eval", "dev_m")}

    def behaviour(hans_lg, dev_lg, hc_lg):
        m = hans_metrics(hans_lg, heur, yh)
        m["devm_acc"] = acc(dev_lg, ydev)
        m["devm_acc_LOminus"] = acc(dev_lg[dev_lo == 0], ydev[dev_lo == 0])
        pc = (hc_lg[:, 1] > hc_lg[:, 0]).astype(int) == hc_y
        m["RG_HC"] = float(pc[hc_y == 0].mean() - pc[hc_y == 1].mean())
        return m

    # ---------- dữ liệu fit theo khái niệm và tầng; eraser
    def concept_data(cname, l):
        df = sets[cname]
        if cname in MNLI_SET_CONCEPTS:
            Xc = pool_cls[[pidx[u] for u in df.uid], ppos[l]].astype(np.float64)
        else:
            Xc = cls["hans_train_B"][l][[htb_idx[u] for u in df.uid]].astype(np.float64)
        return df, Xc, df.z.to_numpy(), (df.split == "fit").to_numpy()

    fits, info = {}, {}
    for cname in concepts:
        for l in layers:
            df, Xc, z, fit = concept_data(cname, l)
            st = FitStats(Xc[fit], method="leace")
            er = st.eraser(z[fit])
            fits[(cname, l)] = (st, er)
            Xe = er(Xc)
            d = {"layer": l, "concept": cname, "rank": er.rank, "rank_expected": int(z.max()),
                 "var_removed": var_removed(st, er), "n_fit": int(fit.sum()),
                 "norm_ratio_after_before": float(np.linalg.norm(Xe, axis=1).mean() / np.linalg.norm(Xc, axis=1).mean()),
                 "mean_shift_rel": float(np.linalg.norm(Xe - Xc, axis=1).mean()
                                         / np.linalg.norm(Xc - Xc[fit].mean(0), axis=1).mean())}
            if cname not in POS_CONCEPTS:
                d["decode_cls_held"], probe = probe_balacc(Xc[fit], z[fit], Xc[~fit], z[~fit])
                d["probe_after_cls_held"] = probe_balacc(Xe[fit], z[fit], Xe[~fit], z[~fit])[0]
                if cname == "c_LO" and probe is not None:
                    d["transfer_HANS_LOplus_rate"] = float(probe.predict(cls["hans_eval"][l]).mean())
            info[(cname, l)] = d
    log(f"  L_ℓ: fit {len(fits)} eraser ({len(concepts)} khái niệm × tầng {layers}), {(time.time() - t0) / 60:.1f} phút")

    logits = {}          # (name) -> dict(hans, dev, hc)
    pooled = {}          # (cname, l, mode) -> pooled trên dữ liệu khái niệm (f16, theo thứ tự df)
    checks = {}
    # ---------- ℓ = L: từ cache qua pooler
    if Lmax in layers:
        ro = Readout(model_dir, device, dt_dev)
        XL = {"hans": ro.t(cls["hans_eval"][Lmax]), "dev": ro.t(cls["dev_m"][Lmax]),
              "hc": ro.t(pool_cls[[pidx[u] for u in hc.uid], ppos[Lmax]])}
        pre0 = {k: v @ ro.Wp.T + ro.bp for k, v in XL.items()}
        base_ro = {k: ro.from_pre(v)[0].double().cpu().numpy() for k, v in pre0.items()}
        for s, k in (("hans_eval", "hans"), ("dev_m", "dev")):
            checks[f"readout_L{Lmax}_reproduces_preds_{s}"] = float((base_ro[k].argmax(1) == saved[s]).mean())
            if checks[f"readout_L{Lmax}_reproduces_preds_{s}"] < 0.99:
                raise ValueError(f"[DỪNG] {run_id}: pooler+head trên cache [CLS]_{Lmax} không tái tạo dự đoán {s}")
        logits[("base", Lmax)] = base_ro
        for cname in concepts:
            st, er = fits[(cname, Lmax)]
            et = er.on(device, dt_dev)
            logits[(cname, Lmax, "cls")] = {k: ro.from_pre(ro.erased_pre(pre0[k], XL[k], et))[0].double().cpu().numpy()
                                            for k in XL}
            if cname not in POS_CONCEPTS:
                df, Xc, z, fit = concept_data(cname, Lmax)
                Xt = ro.t(Xc)
                pooled[(cname, Lmax, "cls")] = ro.from_pre(ro.erased_pre(Xt @ ro.Wp.T + ro.bp, Xt, et))[1].cpu().numpy()
                pooled[(cname, Lmax, "none")] = ro.from_pre(Xt @ ro.Wp.T + ro.bp)[1].cpu().numpy()
    # ---------- ℓ < L: chạy tiếp trong mô hình
    model = tok = None
    if mid:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
        cfgs = []
        for l in mid:
            for cname in concepts:
                cfgs.append({"name": f"{cname}|{l}|cls", "concept": cname, "layer": l, "mode": "cls",
                             "eraser": fits[(cname, l)][1].on(device, dt_dev)})
            if l in all_layers:
                for cname in [c for c in ALL_TOKEN_CONCEPTS if c in concepts]:
                    cfgs.append({"name": f"{cname}|{l}|all", "concept": cname, "layer": l, "mode": "all",
                                 "eraser": fits[(cname, l)][1].on(device, dt_dev)})
        rh = run_configs(model, tok, fr["hans_eval"], cfgs, a.batch, fp16, log=log, tag="HANS-EVAL")
        rd = run_configs(model, tok, fr["dev_m"], cfgs, a.batch, fp16, log=log, tag="DEV-M")
        small_uids = list(dict.fromkeys(list(hc.uid) + list(sets["c_LO"].uid)))
        small = union.set_index("uid").loc[small_uids].reset_index()
        lo_cfg = [c["name"] for c in cfgs if c["concept"] == "c_LO"]
        rs = run_configs(model, tok, small, cfgs, a.batch, fp16, keep_pooled=lo_cfg, log=log, tag="MNLI-HC + c_LO")
        htb_cfgs = [c for c in cfgs if c["concept"] in HTB_CONCEPTS]
        rb = run_configs(model, tok, fr["hans_train_B"], htb_cfgs, a.batch, fp16,
                         keep_pooled=[c["name"] for c in htb_cfgs], log=log, tag="HT-B (phục hồi)")
        for s, r in (("hans_eval", rh), ("dev_m", rd)):
            checks[f"live_reproduces_preds_{s}"] = float((r["base_logits"].argmax(1) == saved[s]).mean())
            if checks[f"live_reproduces_preds_{s}"] < 0.99:
                raise ValueError(f"[DỪNG] {run_id}: forward trực tiếp không tái tạo dự đoán P2 trên {s}")
        idd = max([max(r["identity_max_abs_diff"].values()) for r in (rh, rd, rs, rb) if r["identity_max_abs_diff"]])
        checks["identity_continuation_max_abs_logit_diff"] = idd
        if idd > 1e-3:
            raise ValueError(f"[DỪNG] {run_id}: chạy tiếp không can thiệp lệch forward gốc {idd:.2e}")
        spos = {u: i for i, u in enumerate(small.uid)}
        hc_s = [spos[u] for u in hc.uid]
        base_mid = {"hans": rh["base_logits"].astype(np.float64), "dev": rd["base_logits"].astype(np.float64),
                    "hc": rs["base_logits"][hc_s].astype(np.float64)}
        for l in mid:
            logits[("base", l)] = base_mid
        for c in cfgs:
            key = (c["concept"], c["layer"], c["mode"])
            logits[key] = {"hans": rh["logits"][c["name"]].astype(np.float64),
                           "dev": rd["logits"][c["name"]].astype(np.float64),
                           "hc": rs["logits"][c["name"]][hc_s].astype(np.float64)}
            if c["concept"] == "c_LO":
                pooled[key] = rs["pooled"][c["name"]][[spos[u] for u in sets["c_LO"].uid]]
            elif c["concept"] in HTB_CONCEPTS:
                pooled[key] = rb["pooled"][c["name"]][[htb_idx[u] for u in sets[c["concept"]].uid]]
        pooled[("c_LO", "base")] = rs["base_pooled"][[spos[u] for u in sets["c_LO"].uid]]
        pooled[("htb", "base")] = rb["base_pooled"]
        log(f"  L_ℓ: chạy tiếp xong, {(time.time() - t0) / 60:.1f} phút")

    # ---------- chỉ số quan sát
    rows, out_cfg = [], {}
    for key, lg in logits.items():
        if key[0] == "base":
            continue
        cname, l, mode = key
        base_lg = logits[("base", l)]
        b0, a1 = behaviour(base_lg["hans"], base_lg["dev"], base_lg["hc"]), behaviour(lg["hans"], lg["dev"], lg["hc"])
        delta = {k: a1[k] - b0[k] for k in b0}
        fl = flip_rates(base_lg["hans"], lg["hans"], heur, yh)
        d = {"concept": cname, "layer": l, "mode": mode, "after": a1, "delta": delta, "flip": fl,
             "mnli_damage_ok": bool(abs(delta["devm_acc"]) * 100 <= 2.0 and delta["devm_acc_LOminus"] * 100 >= -2.0)}
        if cname in POS_CONCEPTS:
            d["positive_control_v01_ok"] = bool(POS_CTRL[0] <= a1["devm_acc"] <= POS_CTRL[1])
            d["devm_acc_base"] = b0["devm_acc"]
        if (cname, l, mode) in pooled:
            df = sets[cname]
            z, fit = df.z.to_numpy(), (df.split == "fit").to_numpy()
            P = pooled[(cname, l, mode)].astype(np.float32)
            d["recovery_pooler_held"] = probe_balacc(P[fit], z[fit], P[~fit], z[~fit])[0]
        out_cfg[f"{cname}|{l}|{mode}"] = d
        if cname in POS_CONCEPTS:
            continue
        for j, (_, tag) in enumerate(HEURS):
            rows.append({"run_id": run_id, "position": f"L{l}", "layer": l, "mode": mode, "concept": cname,
                         "column": tag, "expected": EXPECTED[cname][j], "RG_base": b0[f"RG_{tag}"],
                         "dRG": delta[f"RG_{tag}"], "dBalAcc": delta[f"BalAcc_{tag}"], "dAUROC": delta[f"AUROC_{tag}"],
                         "dAccE": delta[f"accE_{tag}"], "dAccNE": delta[f"accNE_{tag}"], "dPE_E": delta[f"PE_E_{tag}"],
                         "dPE_NE": delta[f"PE_NE_{tag}"], "margin_NE_base": b0[f"margin_NE_{tag}"],
                         "dMargin_NE": delta[f"margin_NE_{tag}"], "dMargin_E": delta[f"margin_E_{tag}"],
                         "flip_E": fl[f"flip_E_{tag}"], "flip_NE": fl[f"flip_NE_{tag}"],
                         "dDevAcc": delta["devm_acc"], "dRG_HC": delta["RG_HC"]})
    # giải mã ở pooler không can thiệp (mốc cho phục hồi)
    for cname in [c for c in concepts if c not in POS_CONCEPTS]:
        df = sets[cname]
        z, fit = df.z.to_numpy(), (df.split == "fit").to_numpy()
        P = None
        if cname == "c_LO" and ("c_LO", "base") in pooled:
            P = pooled[("c_LO", "base")]
        elif cname in HTB_CONCEPTS and ("htb", "base") in pooled:
            P = pooled[("htb", "base")][[htb_idx[u] for u in df.uid]]
        elif (cname, Lmax, "none") in pooled:
            P = pooled[(cname, Lmax, "none")]
        if P is not None:
            info[(cname, "pooler")] = {"decode_pooler_no_intervention_held":
                                       probe_balacc(P[fit].astype(np.float32), z[fit], P[~fit].astype(np.float32), z[~fit])[0]}
    cells = pd.DataFrame(rows)

    # ---------- null
    null_res = {}
    sub = null_subset(fr["hans_eval"], a.null_subset_per_subcase, a.seed)
    sub_masks = [(mE[sub], mN[sub]) for mE, mN in masks]
    for l in null_layers:
        for cname in [c for c in HOLM_FAMILY if c in concepts]:
            st, er = fits[(cname, l)]
            df, Xc, z, fit = concept_data(cname, l)
            strata = df.stratum.to_numpy()[fit]
            k = int(z.max()) + 1
            if l == Lmax:
                n3 = n1 = a.n_null_last
            else:
                n3, n1 = a.n_null_mid_n3, a.n_null_mid_n1
            ers = [("N3", st.eraser(stratified_permutation(z[fit], strata, rng))) for _ in range(n3)] + \
                  [("N1", st.random_eraser(k - 1, gen)) for _ in range(n1)]
            res = {"N3": [], "N1": []}
            if l == Lmax:
                b_null = null_vector(logits[("base", l)]["hans"], masks)
                b_dev = acc(logits[("base", l)]["dev"], ydev)
                for kind, e in ers:
                    et = e.on(device, dt_dev)
                    hl = ro.from_pre(ro.erased_pre(pre0["hans"], XL["hans"], et))[0].double().cpu().numpy()
                    dl = ro.from_pre(ro.erased_pre(pre0["dev"], XL["dev"], et))[0].double().cpu().numpy()
                    v = null_vector(hl, masks) - b_null
                    res[kind].append(list(v[:3]) + [acc(dl, ydev) - b_dev] + list(v[3:]))
                obs = null_vector(logits[(cname, l, "cls")]["hans"], masks) - b_null
                basis = "HANS-EVAL đầy đủ"
            else:
                cf = [{"name": f"{kind}{i}", "layer": l, "mode": "cls", "eraser": e.on(device, dt_dev)}
                      for i, (kind, e) in enumerate(ers)]
                rn = run_configs(model, tok, fr["hans_eval"].iloc[sub].reset_index(drop=True), cf, a.batch, fp16,
                                 log=log, tag=f"null {cname} tầng {l} (HANS rút gọn {len(sub)})")
                b_null = null_vector(rn["base_logits"].astype(np.float64), sub_masks)
                for c_, (kind, _) in zip(cf, ers):
                    v = null_vector(rn["logits"][c_["name"]].astype(np.float64), sub_masks) - b_null
                    res[kind].append(list(v[:3]) + [np.nan] + list(v[3:]))
                obs = null_vector(logits[(cname, l, "cls")]["hans"][sub], sub_masks) \
                    - null_vector(logits[("base", l)]["hans"][sub], sub_masks)
                basis = f"HANS-EVAL rút gọn {len(sub)}"
            res = {k_: np.array(v_) for k_, v_ in res.items()}
            null_res[f"{cname}_L{l}"] = res
            for j, (_, tag) in enumerate(HEURS):
                m = (cells.concept == cname) & (cells.layer == l) & (cells["mode"] == "cls") & (cells.column == tag)
                cells.loc[m, "null_basis"] = basis
                cells.loc[m, "dRG_on_null_basis"] = obs[j]
                cells.loc[m, "dPE_NE_on_null_basis"] = obs[3 + j]
                for kind in ("N3", "N1"):
                    nv = res[kind][:, j]
                    cells.loc[m, f"n_{kind}"] = len(nv)
                    cells.loc[m, f"p_{kind}"] = p_one_sided(nv, obs[j])
                    cells.loc[m, f"q05_{kind}"] = float(np.quantile(nv, 0.05))
                    cells.loc[m, f"mean_{kind}"] = float(nv.mean())
                    cells.loc[m, f"sd_{kind}"] = float(nv.std(ddof=1))
                    cells.loc[m, f"p_{kind}_PE_NE"] = p_one_sided(res[kind][:, 4 + j], obs[3 + j])
            log(f"  null {cname} tầng {l}: N3 {n3}, N1 {n1} ({basis}), {(time.time() - t0) / 60:.1f} phút")
    if "p_N3" in cells.columns:
        fam = cells.p_N3.notna()
        cells["q_bh"] = np.nan
        cells.loc[fam, "q_bh"] = bh(cells.loc[fam, "p_N3"].to_numpy())
        cells["p_holm_layer"] = np.nan
        for l in null_layers:
            m = fam & (cells.layer == l)
            if m.any():
                cells.loc[m, "p_holm_layer"] = holm(cells.loc[m, "p_N3"].to_numpy())
        cells["cell_robust_this_seed"] = (fam & (cells.dRG <= -DELTA_MIN) & (cells.q_bh < ALPHA)
                                          & (cells.dRG_on_null_basis < cells.q05_N3))
    if model is not None:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    res = {"run_id": run_id, "num_hidden_layers": Lmax, "layers": layers, "all_token_layers": all_layers,
           "null_layers": null_layers, "checks": checks, "null_subset_n": int(len(sub)),
           "concepts": {f"{c}|{l}": v for (c, l), v in info.items()}, "configs": out_cfg,
           "wall_minutes": round((time.time() - t0) / 60, 2)}
    if null_res:
        np.savez_compressed(os.path.join(a.out, run_id, "lmid_nulls.npz"),
                            **{f"{k}_{kind}": v[kind] for k, v in null_res.items() for kind in v})
    log(f"  L_ℓ xong: {res['wall_minutes']} phút; kiểm: {checks}")
    return res, cells


# ------------------------------------------------------------------ tổng hợp
def summarize_layers(models, cells):
    fams = sorted({family(m["run_id"]) for m in models})
    if len(fams) > 1:
        return "\n\n".join(summarize_layers([m for m in models if family(m["run_id"]) == f],
                                            cells[cells.run_id.map(family) == f]) for f in fams)
    n = len(models)
    layers = models[0]["layers"]
    Lmax = models[0]["num_hidden_layers"]
    T = [f"# {fams[0]}: erasure tại [CLS] tầng giữa L_ℓ (LEACE) — {n} mô hình, tầng {layers} (L = {Lmax})\n"]
    T.append("Vị trí phụ theo SPEC_ERASURE §2.2. ℓ = L: xoá ngay TRƯỚC pooler (tanh); ℓ < L: xoá rồi chạy tiếp các tầng sau. "
             "Khác L_out, ở đây KHÔNG có bảo đảm toán học về logit -> hiệu ứng (nếu có) là kết quả thực nghiệm.")
    T.append("\n## 1. Kiểm tra")
    for m in models:
        T.append(f"- {m['run_id']}: " + ", ".join(f"{k} {v:.4g}" for k, v in m["checks"].items())
                 + f"; HANS rút gọn cho null: {m['null_subset_n']}; {m['wall_minutes']} phút")
    T.append("\n## 2. Control dương theo tầng: DEV-M acc sau khi xoá nhãn vàng (tiêu chí v0.1 của L_out: 0.45–0.60)")
    T.append("Đọc: DEV-M gần như không đổi -> can thiệp ở tầng đó bị các tầng sau 'vá' lại (attention lấy lại thông tin từ "
             "token khác); khi đó hiệu ứng ≈ 0 của khái niệm heuristic ở tầng đó KHÔNG có giá trị thông tin.")
    T.append("| Khái niệm | mode | " + " | ".join(f"ℓ={l}" for l in layers) + " |")
    T.append("|---|---|" + "---|" * len(layers))
    base_dev = np.mean([v["devm_acc_base"] for m in models for v in m["configs"].values() if "devm_acc_base" in v])
    for cn in POS_CONCEPTS:
        for mode in ("cls", "all"):
            vals = []
            for l in layers:
                v = [m["configs"].get(f"{cn}|{l}|{mode}") for m in models]
                v = [x["after"]["devm_acc"] for x in v if x]
                vals.append(f"{np.mean(v):.3f}" if v else "—")
            if any(x != "—" for x in vals):
                T.append(f"| {cn} | {mode} | " + " | ".join(vals) + " |")
    T.append(f"(DEV-M gốc TB {base_dev:.3f})")
    T.append("Độ mạnh can thiệp theo quy tắc đặt trước (c_LABEL, mode cls; Δ_lab = DEV-M gốc − sau xoá, từng seed): "
             "≥ 0.30 mạnh | 0.10–0.30 một phần | < 0.10 bị vá")
    for l in layers:
        per = []
        for m in models:
            v = m["configs"].get(f"c_LABEL|{l}|cls")
            if v:
                d = v["devm_acc_base"] - v["after"]["devm_acc"]
                per.append(f"{d:.3f} ({'mạnh' if d >= 0.30 else 'một phần' if d >= 0.10 else 'bị vá'})")
        if per:
            T.append(f"- ℓ={l}: " + "; ".join(per))
    T.append("\n## 3. Giải mã và phục hồi (TB qua seed): BalAcc held-out")
    T.append("giải mã [CLS]_ℓ trước xoá | sau xoá tại [CLS]_ℓ (≈0.5 theo toán) | phục hồi ở pooler sau can thiệp CLS | "
             "phục hồi sau can thiệp mọi token | mốc: giải mã ở pooler không can thiệp. Phục hồi ≈ mốc -> khái niệm được "
             "đưa trở lại [CLS] qua attention (R-09).")
    T.append("| Khái niệm | ℓ | rank | giải mã CLS | sau xoá CLS | phục hồi pooler (cls) | phục hồi pooler (all) | mốc pooler |"
             " c_LO→HANS |")
    T.append("|---|---|---|---|---|---|---|---|---|")
    for cn in ["c_LO", "c_SUB", "c_CON", "c_JOINT", "c_CTRL_H"]:
        mk = [m["concepts"].get(f"{cn}|pooler", {}).get("decode_pooler_no_intervention_held", np.nan) for m in models]
        for l in layers:
            ci = [m["concepts"].get(f"{cn}|{l}") for m in models]
            ci = [x for x in ci if x]
            if not ci:
                continue
            def g(key, ci=ci):
                v = [x.get(key, np.nan) for x in ci]
                v = [x for x in v if x == x]
                return float(np.mean(v)) if v else float("nan")
            rc = [m["configs"].get(f"{cn}|{l}|cls", {}).get("recovery_pooler_held", np.nan) for m in models]
            ra = [m["configs"].get(f"{cn}|{l}|all", {}).get("recovery_pooler_held", np.nan) for m in models]
            T.append(f"| {cn} | {l} | {g('rank'):.0f}/{ci[0]['rank_expected']} | {_f(g('decode_cls_held'))} | "
                     f"{_f(g('probe_after_cls_held'))} | {_f(float(np.nanmean(rc)) if np.isfinite(rc).any() else np.nan)} | "
                     f"{_f(float(np.nanmean(ra)) if np.isfinite(ra).any() else np.nan)} | "
                     f"{_f(float(np.nanmean(mk)) if np.isfinite(mk).any() else np.nan)} | "
                     f"{_f(g('transfer_HANS_LOplus_rate'), 2) if cn == 'c_LO' else ''} |")
    T.append("\n## 4. Ma trận ΔRG theo tầng (mode cls; TB ± SD qua seed) | ô có null: số seed 'vững' / tổng")
    T.append("'vững' ở một seed = ΔRG ≤ −0.05 VÀ q (BH trên mọi ô có null của mô hình) < 0.05 VÀ ΔRG (trên tập dùng cho null) "
             "thấp hơn phân vị 5% của N3. Ô không có null: chỉ mô tả.")
    T.append("| Khái niệm | ℓ | kỳ vọng | cột LO | cột SUB | cột CON | ΔDEV-M (điểm) |")
    T.append("|---|---|---|---|---|---|---|")
    cl = cells[cells["mode"] == "cls"]
    for cn in ["c_LO", "c_SUB", "c_CON", "c_JOINT", "c_CTRL_H"]:
        for l in layers:
            s0 = cl[(cl.concept == cn) & (cl.layer == l)]
            if not len(s0):
                continue
            parts = []
            for tag in ("LO", "SUB", "CON"):
                s = s0[s0.column == tag]
                txt = f"{s.dRG.mean():+.3f} ± {s.dRG.std(ddof=1) if len(s) > 1 else 0:.3f}"
                if "p_N3" in s and s.p_N3.notna().any():
                    txt += (f" | {int(s.cell_robust_this_seed.sum())}/{len(s)} (q max {s.q_bh.max():.3f}; "
                            f"z_N3 TB {((s.dRG_on_null_basis - s.mean_N3) / s.sd_N3).mean():+.1f})")
                parts.append(txt)
            T.append(f"| {cn} | {l} | {EXPECTED[cn]} | " + " | ".join(parts) + f" | {s0.dDevAcc.mean() * 100:+.2f} |")
    T.append("\n## 5. Chỉ số phụ (mô tả): ΔP̄E trên NE và Δmargin_NE (mode cls, TB qua seed)")
    T.append("| Khái niệm | ℓ | ΔP̄E_NE LO/SUB/CON | p_N3 (ΔP̄E_NE) max qua seed | Δmargin_NE LO/SUB/CON | margin_NE gốc |")
    T.append("|---|---|---|---|---|---|")
    for cn in ["c_LO", "c_SUB", "c_CON", "c_JOINT", "c_CTRL_H"]:
        for l in layers:
            s0 = cl[(cl.concept == cn) & (cl.layer == l)].groupby("column").mean(numeric_only=True)
            if not len(s0):
                continue
            pcol = "p_N3_PE_NE" if "p_N3_PE_NE" in s0 else None
            pv = cl[(cl.concept == cn) & (cl.layer == l)].groupby("column")[pcol].max() if pcol else None
            T.append(f"| {cn} | {l} | " + "/".join(f"{s0.loc[t, 'dPE_NE']:+.4f}" for t in ("LO", "SUB", "CON")) + " | "
                     + ("/".join(_f(pv.loc[t]) for t in ("LO", "SUB", "CON")) if pv is not None else "—") + " | "
                     + "/".join(f"{s0.loc[t, 'dMargin_NE']:+.3f}" for t in ("LO", "SUB", "CON")) + " | "
                     + "/".join(f"{s0.loc[t, 'margin_NE_base']:+.2f}" for t in ("LO", "SUB", "CON")) + " |")
    ca = cells[cells["mode"] == "all"]
    if len(ca):
        T.append("\n## 6. THĂM DÒ — eraser [CLS] áp cho mọi token (không có null; không dùng để kết luận)")
        T.append("| Khái niệm | ℓ | ΔRG LO/SUB/CON | ΔP̄E_NE LO/SUB/CON | ΔDEV-M (điểm) |")
        T.append("|---|---|---|---|---|")
        for cn in ["c_LO", "c_SUB", "c_CON"]:
            for l in layers:
                s0 = ca[(ca.concept == cn) & (ca.layer == l)].groupby("column").mean(numeric_only=True)
                if not len(s0):
                    continue
                T.append(f"| {cn} | {l} | " + "/".join(f"{s0.loc[t, 'dRG']:+.3f}" for t in ("LO", "SUB", "CON")) + " | "
                         + "/".join(f"{s0.loc[t, 'dPE_NE']:+.4f}" for t in ("LO", "SUB", "CON"))
                         + f" | {s0.dDevAcc.mean() * 100:+.2f} |")
    T.append("\n## 7. Chuẩn biểu diễn tại [CLS]_ℓ (TB qua seed): ‖r(h)‖/‖h‖ và ‖r(h) − h‖ tương đối")
    T.append("| Khái niệm | " + " | ".join(f"ℓ={l}" for l in layers) + " |")
    T.append("|---|" + "---|" * len(layers))
    for cn in L_CONCEPTS:
        vals = []
        for l in layers:
            ci = [m["concepts"].get(f"{cn}|{l}") for m in models]
            ci = [x for x in ci if x]
            vals.append(f"{np.mean([x['norm_ratio_after_before'] for x in ci]):.3f} / "
                        f"{np.mean([x['mean_shift_rel'] for x in ci]):.3f}" if ci else "—")
        T.append(f"| {cn} | " + " | ".join(vals) + " |")
    T.append("\nGhi chú: null tại ℓ < L tính trên HANS rút gọn và giá trị quan sát dùng để so cũng tính trên đúng tập đó. "
             "Chưa có: MP/INLP, N2, re-injection tại L_ℓ, constrain_cov_trace, probe MLP (SPEC §5).")
    return "\n".join(T)
