"""
sets.py — dựng tập fit khái niệm cho erasure (SPEC_DULIEU_KHAINIEM v1.2 §5–§6), tất định theo seed.

c_LO (MNLI-POOL, §6.1):
  1. Tách 40% ô LO+ (từng nhãn, phân tầng theo neg_pattern) -> MNLI-HC (đánh giá RG trên văn bản tự nhiên).
  2. n* = số (LO+, C) còn lại; lấy n* (LO+, E) phân tầng theo (neg_pattern × hyp_len_bin).
  3. Ghép LO−: mỗi nhãn y và tầng s = (neg_pattern, hyp_len_bin, genre), lấy đúng số LO+ của (y, s);
     thiếu -> gộp genre -> chỉ neg_pattern -> toàn nhãn (ghi lại mọi lần gộp).
     TRONG mỗi tầng: ghép láng giềng gần nhất (tham lam, không lặp lại) theo (prem_len, hyp_len) đã chuẩn hoá
     (chia độ lệch chuẩn của LO− trong nhãn); hoà khoảng cách -> chọn ngẫu nhiên (ứng viên đã xáo theo seed).
  4. SMD giữa LO+ và LO− trong từng nhãn (yêu cầu |SMD| < 0.1), báo cáo TRƯỚC (LO+ vs toàn bộ LO−) và SAU ghép.
  5. Chia fit/held-out 75/25 phân tầng theo (z, y).
  6. Độ ổn định (§6.1 bước 7): lặp bước 2–5 với N_RESAMPLE seed lấy mẫu (mẫu 0 = mẫu chính); MNLI-HC cố định.
c_LABEL (§6.4): 10,000 ví dụ MNLI-POOL ngoài mọi mẫu c_LO và MNLI-HC, E/C 50/50, chia 75/25.
c_LABEL_40K (chẩn đoán, thêm sau pilot 1): 40,000 ví dụ cùng quy tắc; c_LABEL là tập con lồng trong nó (cùng split).
HT-B (§6.3): c_SUB, c_CON, c_JOINT, c_CTRL_H; chia 75/25 phân tầng theo (subcase, z);
  thêm cờ held-out theo TEMPLATE (mỗi subcase có ≥ 2 template: template cuối theo thứ tự tên) — chỉ dùng
  để đo khái niệm có tổng quát ra ngoài template không (rủi ro R-10/R-23), không dùng để fit eraser.
Mỗi tập trả về DataFrame: uid, z, y, split ∈ {fit, held}, stratum (cho null hoán vị) + cột phụ.

SỬA SAU PILOT 1 (10/11/2026, sửa sau khi thấy dữ liệu — CHỈ dữ liệu cân bằng, KHÔNG nhìn kết quả erasure để chọn):
  - Pilot 1 ghép LO− ngẫu nhiên trong tầng -> |SMD| prem_len (nhãn E) = 0.232, vi phạm tiêu chí tiền đăng ký < 0.1.
    Thêm ghép láng giềng gần nhất theo (prem_len, hyp_len) BÊN TRONG tầng khớp chính xác (số lượng mỗi tầng giữ nguyên
    như §6.1 bước 4).
  - Thêm c_LABEL_40K làm chẩn đoán control dương (xem pilot.py); c_LABEL 10K tiền đăng ký giữ nguyên vai trò.
  - Cài §6.1 bước 7 (độ ổn định hướng, 10 mẫu) — đã có trong SPEC, pilot 1 chưa cài.
  - Các bộ sinh số ngẫu nhiên tách theo mục đích (chia HC / lấy mẫu c_LO thứ r / c_LABEL / HT-B) để mẫu chính không phụ
    thuộc số lần lặp. Hệ quả: mẫu chính KHÁC mẫu của pilot 1.
"""
import json

import numpy as np
import pandas as pd

HC_FRAC = 0.40
HELD_FRAC = 0.25
GENRES_KEY = "genre"
MATCH_LEVELS = [["neg_pattern", "hyp_len_bin", "genre"], ["neg_pattern", "hyp_len_bin"], ["neg_pattern"], []]
NN_VARS = ("prem_len", "hyp_len")
COVARS = ["hyp_len", "prem_len", "neg_p", "neg_h"]
N_RESAMPLE = 10


def _key(df, keys):
    if not keys:
        return pd.Series("all", index=df.index)
    out = df[keys[0]].astype(str)
    for k in keys[1:]:
        out = out + "|" + df[k].astype(str)
    return out


def _pick(df, k, rng):
    """k dòng ngẫu nhiên, tất định (sắp theo uid trước khi rút)."""
    if k >= len(df):
        return df
    d = df.sort_values("uid")
    return d.iloc[np.sort(rng.choice(len(d), size=k, replace=False))]


def strat_sample(df, n, by, rng):
    """Lấy n dòng, phân bổ theo tỉ lệ các tầng `by` (phần dư lớn nhất), tất định."""
    if n >= len(df):
        return df.copy()
    key = _key(df, by)
    sizes = key.value_counts().sort_index()
    quota = sizes / sizes.sum() * n
    base = np.floor(quota).astype(int)
    rem = int(n - base.sum())
    extra = (quota - base).sort_values(ascending=False, kind="stable").index[:rem]
    base.loc[extra] += 1
    parts = [_pick(df[key == s], int(k), rng) for s, k in base.items() if k > 0]
    return pd.concat(parts) if parts else df.iloc[:0]


def strat_split(df, frac, by, rng):
    """Tách (giữ, tách_ra) với tỉ lệ frac trong từng tầng `by`."""
    key = _key(df, by)
    out = []
    for s in sorted(key.unique()):
        g = df[key == s].sort_values("uid")
        k = int(round(len(g) * frac))
        idx = rng.permutation(len(g))[:k]
        out.append(g.iloc[np.sort(idx)])
    taken = pd.concat(out) if out else df.iloc[:0]
    return df[~df.uid.isin(taken.uid)], taken


def match_counts(source, target, keys, rng):
    """Lấy từ source đúng số dòng theo từng tầng `keys` của target; thiếu thì bù ngẫu nhiên từ phần còn lại."""
    sk, tk = _key(source, keys), _key(target, keys)
    picks, short = [], 0
    for s, k in tk.value_counts().sort_index().items():
        cand = source[sk == s]
        p = _pick(cand, k, rng)
        short += k - len(p)
        picks.append(p)
    got = pd.concat(picks) if picks else source.iloc[:0]
    if short:
        got = pd.concat([got, _pick(source[~source.uid.isin(got.uid)], short, rng)])
    return got


def nn_pick(targets, cand, nn_vars, scale, rng):
    """Ghép láng giềng gần nhất tham lam, không lặp lại: mỗi dòng của `targets` (thứ tự ngẫu nhiên) lấy ứng viên chưa dùng
    gần nhất theo khoảng cách Euclid trên `nn_vars` / `scale`. Ứng viên được xáo trước -> hoà khoảng cách chọn ngẫu nhiên.
    Trả về các dòng ứng viên đã chọn (≤ len(targets))."""
    if len(cand) <= len(targets):
        return cand
    cand = cand.sort_values("uid")
    cand = cand.iloc[rng.permutation(len(cand))]
    C = cand[list(nn_vars)].to_numpy(float) / scale
    T = targets[list(nn_vars)].to_numpy(float) / scale
    free = np.ones(len(cand), bool)
    chosen = []
    for i in rng.permutation(len(T)):
        d = ((C - T[i]) ** 2).sum(1)
        d[~free] = np.inf
        j = int(np.argmin(d))
        free[j] = False
        chosen.append(j)
    return cand.iloc[sorted(chosen)]


def match_minus(plus, minus, rng, levels=MATCH_LEVELS, nn_vars=NN_VARS):
    """Ghép LO− cho LO+ theo nhãn và tầng hiệp biến (đúng số lượng mỗi tầng), gộp dần tầng khi thiếu.
    nn_vars: ghép láng giềng gần nhất trong tầng (None -> rút ngẫu nhiên như pilot 1)."""
    out, log = [], []
    for y in (0, 1):
        deficit = plus[plus.y == y]
        cand_all = minus[minus.y == y]
        scale = None
        if nn_vars:
            scale = cand_all[list(nn_vars)].to_numpy(float).std(0, ddof=1)
            scale = np.where(scale > 0, scale, 1.0)
        used = set()
        for lv, keys in enumerate(levels):
            if not len(deficit):
                break
            cand = cand_all[~cand_all.uid.isin(used)]
            dk, ck = _key(deficit, keys), _key(cand, keys)
            new_def = []
            n_matched = 0
            for s in sorted(dk.unique()):
                drows = deficit[dk == s].sort_values("uid")
                pool_s = cand[ck == s]
                pick = nn_pick(drows, pool_s, nn_vars, scale, rng) if nn_vars else _pick(pool_s, len(drows), rng)
                used.update(pick.uid)
                out.append(pick)
                n_matched += len(pick)
                if len(pick) < len(drows):
                    new_def.append(drows.iloc[len(pick):])
            deficit = pd.concat(new_def) if new_def else deficit.iloc[:0]
            log.append({"y": y, "level": lv, "keys": keys, "matched": n_matched, "unmatched_after": len(deficit)})
        if len(deficit):
            raise ValueError(f"[DỪNG] không ghép đủ LO− cho nhãn {y}: thiếu {len(deficit)}")
    return pd.concat(out), log


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def smd_pairs(plus, minus, genres):
    """SMD (LO+ − LO−) trong từng nhãn cho hiệp biến liên tục và tỉ lệ từng genre."""
    out = {}
    for y, name in ((0, "E"), (1, "C")):
        p, m = plus[plus.y == y], minus[minus.y == y]
        for v in COVARS:
            out[f"y={name}|{v}"] = round(smd(p[v], m[v]), 3)
        for g in genres:
            out[f"y={name}|genre={g}"] = round(smd(p[GENRES_KEY] == g, m[GENRES_KEY] == g), 3)
    return out


def smd_table(fit):
    genres = sorted(fit[GENRES_KEY].unique())
    return smd_pairs(fit[fit.z == 1], fit[fit.z == 0], genres)


def _max_abs(d):
    v = [abs(x) for x in d.values() if x == x]
    return float(max(v)) if v else float("nan")


def _draw_c_lo(keep_c, keep_e, lo_m, rng, nn_vars):
    """Bước 2–5 của §6.1 cho một seed lấy mẫu."""
    n_star = len(keep_c)
    sel_e = strat_sample(keep_e, n_star, ["neg_pattern", "hyp_len_bin"], rng)
    plus = pd.concat([keep_c, sel_e])
    minus, mlog = match_minus(plus, lo_m, rng, nn_vars=nn_vars)
    lo_fit = pd.concat([plus.assign(z=1), minus.assign(z=0)])
    lo_fit["stratum"] = _key(lo_fit, ["y", "neg_pattern", "hyp_len_bin"]).values    # ô cho null hoán vị
    _, held = strat_split(lo_fit, HELD_FRAC, ["z", "y"], rng)
    lo_fit["split"] = np.where(lo_fit.uid.isin(held.uid), "held", "fit")
    return lo_fit, plus, mlog, len(sel_e)


def build_mnli_sets(pool, seed=1234, n_label=10000, n_label_big=40000, n_resample=N_RESAMPLE, nn_vars=NN_VARS):
    """pool: MNLI-POOL (in_pool == 1) có cột lo, y, neg_pattern, hyp_len_bin, genre, hiệp biến.
    Bộ sinh ngẫu nhiên: [seed, 0] chia HC; [seed, 1, r] lấy mẫu c_LO thứ r (r = 0 là mẫu chính); [seed, 2] c_LABEL."""
    if isinstance(seed, np.random.Generator):
        raise TypeError("build_mnli_sets nhận seed (int), không nhận Generator (đổi từ pilot 1)")
    lo_p, lo_m = pool[pool.lo == 1], pool[pool.lo == 0]
    rng_split = np.random.default_rng([seed, 0])
    keep_c, hold_c = strat_split(lo_p[lo_p.y == 1], HC_FRAC, ["neg_pattern"], rng_split)
    keep_e, hold_e = strat_split(lo_p[lo_p.y == 0], HC_FRAC, ["neg_pattern"], rng_split)
    hc_e = match_counts(hold_e, hold_c, ["neg_pattern"], rng_split)
    hc = pd.concat([hold_c, hc_e]).assign(z=1, split="hc", stratum="hc")
    genres = sorted(pool[GENRES_KEY].unique())
    lo_fit, plus, mlog, n_e = _draw_c_lo(keep_c, keep_e, lo_m, np.random.default_rng([seed, 1, 0]), nn_vars)
    resamples, rs_smd = [], []
    for r in range(1, n_resample):
        d, _, _, _ = _draw_c_lo(keep_c, keep_e, lo_m, np.random.default_rng([seed, 1, r]), nn_vars)
        resamples.append(d.assign(resample=r))
        rs_smd.append(_max_abs(smd_table(d)))
    resamples = pd.concat(resamples) if resamples else lo_fit.iloc[:0].assign(resample=0)
    used = set(lo_fit.uid) | set(hc.uid) | set(resamples.uid)
    avail = pool[~pool.uid.isin(used)]
    rng_l = np.random.default_rng([seed, 2])
    k_big = max(n_label, n_label_big or 0) // 2
    parts = []
    for y in (0, 1):
        g = _pick(avail[avail.y == y], k_big, rng_l)
        parts.append(g.iloc[rng_l.permutation(len(g))].assign(order=np.arange(len(g))))   # thứ tự lồng nhau
    big = pd.concat(parts)
    big = big.assign(z=big.y.values, stratum="all")
    _, held_l = strat_split(big, HELD_FRAC, ["y"], rng_l)
    big["split"] = np.where(big.uid.isin(held_l.uid), "held", "fit")
    lab = big[big.order < n_label // 2].copy()
    smd_after = smd_table(lo_fit)
    smd_before = smd_pairs(plus, lo_m, genres)
    report = {"n_star_LO+C_fit": int(len(keep_c)), "LO+E_fit": int(n_e), "LO-_matched": int((lo_fit.z == 0).sum()),
              "c_LO_total": int(len(lo_fit)), "c_LO_fit": int((lo_fit.split == "fit").sum()),
              "matching": f"tầng khớp chính xác {MATCH_LEVELS[0]} + láng giềng gần nhất theo {list(nn_vars or [])}",
              "MNLI_HC": {"LO+C": int(len(hold_c)), "LO+E": int(len(hc_e))},
              "c_LABEL": {"n": int(len(lab)), "E": int((lab.y == 0).sum()), "C": int((lab.y == 1).sum())},
              "c_LABEL_40K": {"n": int(len(big)), "E": int((big.y == 0).sum()), "C": int((big.y == 1).sum())},
              "match_log": mlog, "SMD_before_matching_LOplus_vs_all_LOminus": smd_before,
              "SMD_c_LO_within_label": smd_after, "SMD_max_abs": _max_abs(smd_after),
              "SMD_max_abs_before": _max_abs(smd_before),
              "resamples": {"n_total_incl_main": int(n_resample), "SMD_max_abs_each": [round(x, 3) for x in rs_smd],
                            "uid_jaccard_with_main_LOminus": [
                                round(len(set(lo_fit.uid[lo_fit.z == 0]) & set(g.uid[g.z == 0]))
                                      / len(set(lo_fit.uid[lo_fit.z == 0]) | set(g.uid[g.z == 0])), 3)
                                for _, g in resamples.groupby("resample")]}}
    sets = {"c_LO": lo_fit, "MNLI_HC": hc, "c_LABEL": lab.drop(columns="order"), "c_LABEL_40K": big.drop(columns="order"),
            "c_LO_resamples": resamples}
    return sets, report


def build_htb_sets(htb, rng):
    """htb: HT-B (cột uid, y, heuristic, subcase, template, c4, subj_num)."""
    lo_only, sub, con = (htb[htb.c4 == c] for c in ("LO_only", "SUB_only", "CON"))
    n0 = len(lo_only)
    s1 = strat_sample(sub, n0 // 2, ["subcase"], rng)
    c1 = strat_sample(con, n0 - n0 // 2, ["subcase"], rng)
    sets = {"c_SUB": pd.concat([lo_only.assign(z=0), s1.assign(z=1), c1.assign(z=1)]),
            "c_CON": pd.concat([sub.assign(z=0), con.assign(z=1)]),
            "c_JOINT": htb.assign(z=htb.c4.map({"LO_only": 0, "SUB_only": 1, "CON": 2}))}
    d = htb[htb.subj_num.isin(["NN", "NNS"])]
    parts = []
    for (_h, _y), g in d.groupby(["heuristic", "y"], sort=True):
        nn, nns = g[g.subj_num == "NN"], g[g.subj_num == "NNS"]
        k = min(len(nn), len(nns))
        parts += [_pick(nn, k, rng).assign(z=0), _pick(nns, k, rng).assign(z=1)]
    sets["c_CTRL_H"] = pd.concat(parts)
    report = {}
    tmpl_last = (htb.groupby("subcase").template.agg(lambda t: sorted(set(t))[-1] if t.nunique() >= 2 else None))
    for name, df in sets.items():
        df = df.copy()
        if df.z.isna().any():
            raise ValueError(f"[DỪNG] {name}: có ví dụ không gán được z")
        df["z"] = df.z.astype(int)
        df["stratum"] = df.y.astype(str).values                   # null hoán vị trong từng nhãn E/NE
        _, held = strat_split(df, HELD_FRAC, ["subcase", "z"], rng)
        df["split"] = np.where(df.uid.isin(held.uid), "held", "fit")
        df["tmpl_held"] = [tmpl_last.get(s) == t for s, t in zip(df.subcase, df.template)]
        sets[name] = df
        report[name] = {"n": int(len(df)), "z_counts": {str(k): int(v) for k, v in df.z.value_counts().sort_index().items()},
                        "n_fit": int((df.split == "fit").sum()), "n_tmpl_held": int(df.tmpl_held.sum())}
    return sets, report


def build_all(data_dir, seed=1234, n_label=10000, n_label_big=40000, n_resample=N_RESAMPLE):
    import os
    tr = pd.read_parquet(os.path.join(data_dir, "mnli_bin_train_all.parquet"))
    pool = tr[tr.in_pool == 1].copy()
    need = {"uid", "y", "lo", "neg_pattern", "hyp_len_bin", "genre", *COVARS}
    miss = need - set(pool.columns)
    if miss:
        raise ValueError(f"[DỪNG] mnli_bin_train_all.parquet thiếu cột {miss}")
    htb = pd.read_json(os.path.join(data_dir, "hans_train_B.jsonl"), lines=True)
    mnli_sets, rep_m = build_mnli_sets(pool, seed=seed, n_label=n_label, n_label_big=n_label_big, n_resample=n_resample)
    htb_sets, rep_h = build_htb_sets(htb, np.random.default_rng([seed, 3]))
    keep = ("c_LO", "MNLI_HC", "c_LABEL_40K", "c_LO_resamples")
    union = pd.concat([mnli_sets[k] for k in keep]).drop_duplicates("uid")
    union = pool.set_index("uid").loc[sorted(union.uid)].reset_index()[["uid", "premise", "hypothesis", "y", "lo"]]
    return {**mnli_sets, **htb_sets}, union, {"seed": seed, "mnli": rep_m, "htb": rep_h,
                                               "pool_union_n": int(len(union))}


def save_sets(sets, union, report, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    keep = ["uid", "z", "y", "split", "stratum"]
    for name, df in sets.items():
        cols = keep + [c for c in ("heuristic", "subcase", "template", "tmpl_held", "neg_pattern", "resample")
                       if c in df.columns]
        df[cols].to_parquet(os.path.join(out_dir, f"set_{name}.parquet"))
    union[["uid", "y", "lo"]].to_parquet(os.path.join(out_dir, "pool_union.parquet"))
    with open(os.path.join(out_dir, "sets_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
