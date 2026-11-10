"""
sets.py — dựng tập fit khái niệm cho erasure (SPEC_DULIEU_KHAINIEM v1.2 §5–§6), tất định theo seed.

c_LO (MNLI-POOL, §6.1):
  1. Tách 40% ô LO+ (từng nhãn, phân tầng theo neg_pattern) -> MNLI-HC (đánh giá RG trên văn bản tự nhiên).
  2. n* = số (LO+, C) còn lại; lấy n* (LO+, E) phân tầng theo (neg_pattern × hyp_len_bin).
  3. Ghép LO−: mỗi nhãn y và tầng s = (neg_pattern, hyp_len_bin, genre), lấy đúng số LO+ của (y, s);
     thiếu -> gộp genre -> chỉ neg_pattern -> toàn nhãn (ghi lại mọi lần gộp).
  4. SMD giữa LO+ và LO− trong từng nhãn (yêu cầu |SMD| < 0.1).
  5. Chia fit/held-out 75/25 phân tầng theo (z, y).
c_LABEL (§6.4): 10,000 ví dụ MNLI-POOL ngoài tập c_LO và MNLI-HC, E/C 50/50, chia 75/25.
HT-B (§6.3): c_SUB, c_CON, c_JOINT, c_CTRL_H; chia 75/25 phân tầng theo (subcase, z);
  thêm cờ held-out theo TEMPLATE (mỗi subcase có ≥ 2 template: template cuối theo thứ tự tên) — chỉ dùng
  để đo khái niệm có tổng quát ra ngoài template không (rủi ro R-10/R-23), không dùng để fit eraser.
Mỗi tập trả về DataFrame: uid, z, y, split ∈ {fit, held}, stratum (cho null hoán vị) + cột phụ.
"""
import json

import numpy as np
import pandas as pd

HC_FRAC = 0.40
HELD_FRAC = 0.25
GENRES_KEY = "genre"
MATCH_LEVELS = [["neg_pattern", "hyp_len_bin", "genre"], ["neg_pattern", "hyp_len_bin"], ["neg_pattern"], []]
COVARS = ["hyp_len", "prem_len", "neg_p", "neg_h"]


def _key(df, keys):
    if not keys:
        return pd.Series("all", index=df.index)
    return df[keys].astype(str).agg("|".join, axis=1)


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


def match_minus(plus, minus, rng, levels=MATCH_LEVELS):
    """Ghép LO− cho từng dòng LO+ theo nhãn và tầng hiệp biến, gộp dần tầng khi thiếu."""
    out, log = [], []
    for y in (0, 1):
        deficit = plus[plus.y == y]
        cand_all = minus[minus.y == y]
        used = set()
        for lv, keys in enumerate(levels):
            if not len(deficit):
                break
            cand = cand_all[~cand_all.uid.isin(used)]
            dk, ck = _key(deficit, keys), _key(cand, keys)
            new_def = []
            n_matched = 0
            for s in sorted(dk.unique()):
                drows = deficit[dk == s]
                pick = _pick(cand[ck == s], len(drows), rng)
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


def smd_table(fit):
    out = {}
    for y, name in ((0, "E"), (1, "C")):
        sub = fit[fit.y == y]
        p, m = sub[sub.z == 1], sub[sub.z == 0]
        for v in COVARS:
            out[f"y={name}|{v}"] = round(smd(p[v], m[v]), 3)
        for g in sorted(fit[GENRES_KEY].unique()):
            out[f"y={name}|genre={g}"] = round(smd(p[GENRES_KEY] == g, m[GENRES_KEY] == g), 3)
    return out


def build_mnli_sets(pool, rng, n_label=10000):
    """pool: MNLI-POOL (in_pool == 1) có cột lo, y, neg_pattern, hyp_len_bin, genre, hiệp biến."""
    lo_p, lo_m = pool[pool.lo == 1], pool[pool.lo == 0]
    keep_c, hold_c = strat_split(lo_p[lo_p.y == 1], HC_FRAC, ["neg_pattern"], rng)
    keep_e, hold_e = strat_split(lo_p[lo_p.y == 0], HC_FRAC, ["neg_pattern"], rng)
    n_star = len(keep_c)
    sel_e = strat_sample(keep_e, n_star, ["neg_pattern", "hyp_len_bin"], rng)
    plus = pd.concat([keep_c, sel_e])
    minus, mlog = match_minus(plus, lo_m, rng)
    lo_fit = pd.concat([plus.assign(z=1), minus.assign(z=0)])
    lo_fit["stratum"] = _key(lo_fit, ["y", "neg_pattern", "hyp_len_bin"]).values    # ô cho null hoán vị
    rest, held = strat_split(lo_fit, HELD_FRAC, ["z", "y"], rng)
    lo_fit["split"] = np.where(lo_fit.uid.isin(held.uid), "held", "fit")
    hc_e = match_counts(hold_e, hold_c, ["neg_pattern"], rng)
    hc = pd.concat([hold_c, hc_e]).assign(z=1, split="hc", stratum="hc")
    used = set(lo_fit.uid) | set(hc.uid)
    avail = pool[~pool.uid.isin(used)]
    k = n_label // 2
    lab = pd.concat([_pick(avail[avail.y == 0], k, rng), _pick(avail[avail.y == 1], k, rng)])
    lab = lab.assign(z=lab.y.values, stratum="all")
    _, held_l = strat_split(lab, HELD_FRAC, ["y"], rng)
    lab["split"] = np.where(lab.uid.isin(held_l.uid), "held", "fit")
    report = {"n_star_LO+C_fit": int(n_star), "LO+E_fit": int(len(sel_e)), "LO-_matched": int(len(minus)),
              "c_LO_total": int(len(lo_fit)), "c_LO_fit": int((lo_fit.split == "fit").sum()),
              "MNLI_HC": {"LO+C": int(len(hold_c)), "LO+E": int(len(hc_e))},
              "c_LABEL": {"n": int(len(lab)), "E": int((lab.y == 0).sum()), "C": int((lab.y == 1).sum())},
              "match_log": mlog, "SMD_c_LO_within_label": smd_table(lo_fit),
              "SMD_max_abs": float(np.nanmax(np.abs(list(smd_table(lo_fit).values()))))}
    return {"c_LO": lo_fit, "MNLI_HC": hc, "c_LABEL": lab}, report


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


def build_all(data_dir, seed=1234, n_label=10000):
    import os
    tr = pd.read_parquet(os.path.join(data_dir, "mnli_bin_train_all.parquet"))
    pool = tr[tr.in_pool == 1].copy()
    need = {"uid", "y", "lo", "neg_pattern", "hyp_len_bin", "genre", *COVARS}
    miss = need - set(pool.columns)
    if miss:
        raise ValueError(f"[DỪNG] mnli_bin_train_all.parquet thiếu cột {miss}")
    htb = pd.read_json(os.path.join(data_dir, "hans_train_B.jsonl"), lines=True)
    rng = np.random.default_rng(seed)
    mnli_sets, rep_m = build_mnli_sets(pool, rng, n_label=n_label)
    htb_sets, rep_h = build_htb_sets(htb, rng)
    union = pd.concat([mnli_sets["c_LO"], mnli_sets["MNLI_HC"], mnli_sets["c_LABEL"]]).drop_duplicates("uid")
    union = pool.set_index("uid").loc[sorted(union.uid)].reset_index()[["uid", "premise", "hypothesis", "y", "lo"]]
    return {**mnli_sets, **htb_sets}, union, {"seed": seed, "mnli": rep_m, "htb": rep_h,
                                               "pool_union_n": int(len(union))}


def save_sets(sets, union, report, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    keep = ["uid", "z", "y", "split", "stratum"]
    for name, df in sets.items():
        cols = keep + [c for c in ("heuristic", "subcase", "template", "tmpl_held", "neg_pattern") if c in df.columns]
        df[cols].to_parquet(os.path.join(out_dir, f"set_{name}.parquet"))
    union.to_parquet(os.path.join(out_dir, "pool_union.parquet"))
    with open(os.path.join(out_dir, "sets_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
