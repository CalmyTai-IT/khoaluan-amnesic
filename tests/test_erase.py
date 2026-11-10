"""Unit test cho erasure (src/khoaluan/erase/): khớp thư viện concept-erasure, bảo đảm LEACE, tập fit khái niệm,
và hai phép kiểm đầu-cuối:
  (1) dữ liệu tổng hợp có CÀI SẴN hiệu ứng -> pilot phải phát hiện (độ nhạy), control dương phải đạt, khái niệm nhiễu
      phải KHÔNG có ý nghĩa (độ đặc hiệu) — kiểm phần thống kê, không cần BERT;
  (2) pipeline thật với BERT 2 tầng (train -> cache P2 -> trích MNLI-POOL -> erasure) chạy hết và sinh đủ đầu ra.
    python -m pytest -q tests/test_erase.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch = pytest.importorskip("torch")
pytest.importorskip("concept_erasure")

from khoaluan.data.feats import hyp_len_bin  # noqa: E402
from khoaluan.erase import sets as SETS  # noqa: E402
from khoaluan.erase.leace_fast import FitStats, one_hot, stratified_permutation, var_removed  # noqa: E402
from khoaluan.erase.pilot import hans_metrics, holm  # noqa: E402

HEURS = ["lexical_overlap", "subsequence", "constituent"]
C4_OF = {"lexical_overlap": "LO_only", "subsequence": "SUB_only", "constituent": "CON"}


# ------------------------------------------------------------------ LEACE nhanh == thư viện
@pytest.mark.parametrize("method,k", [("leace", 2), ("leace", 3), ("orth", 2), ("orth", 3)])
def test_fast_leace_matches_library(method, k):
    from concept_erasure import LeaceFitter
    rng = np.random.default_rng(1)
    n, d = 500, 24
    X = rng.normal(size=(n, d)) @ rng.normal(size=(d, d)) * 0.4
    z = rng.integers(0, k, n)
    X[:, 1] += 1.5 * z
    lib = LeaceFitter(d, k, method=method, affine=True, shrinkage=True, constrain_cov_trace=False, svd_tol=0.01,
                      dtype=torch.float64)
    lib.update(torch.tensor(X), torch.tensor(one_hot(z, k)))
    mine = FitStats(X, method=method).eraser(z)
    assert np.allclose(lib.eraser(torch.tensor(X)).numpy(), mine(X), atol=1e-10)
    assert mine.rank == k - 1


def test_guarantee_logit_shift_and_varremoved():
    rng = np.random.default_rng(2)
    n, d = 800, 16
    X = rng.normal(size=(n, d))
    z = rng.integers(0, 2, n)
    X[:, 0] += 2 * z
    st = FitStats(X)
    er = st.eraser(z)
    Xe = er(X)
    assert np.abs(Xe[z == 1].mean(0) - Xe[z == 0].mean(0)).max() < 1e-10          # Cov(X', Z) = 0 trên tập fit
    W = torch.tensor(rng.normal(size=(2, d)))
    b = rng.normal(size=2)
    explicit = Xe @ W.numpy().T + b
    shortcut = X @ W.numpy().T + b + er.logit_shift(torch.tensor(X) - st.mu, W).numpy()
    assert np.allclose(explicit, shortcut, atol=1e-10)
    assert 0 < var_removed(st, er) < 0.5
    r = st.random_eraser(2, torch.Generator().manual_seed(0))
    assert np.linalg.matrix_rank(X - r(X), tol=1e-8) == 2


def test_stratified_permutation_and_holm():
    rng = np.random.default_rng(3)
    z = np.array([0, 1] * 50)
    s = np.array(["a"] * 30 + ["b"] * 70)
    p = stratified_permutation(z, s, rng)
    for g in ("a", "b"):
        assert sorted(p[s == g]) == sorted(z[s == g])
    assert not np.array_equal(p, z)
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    from khoaluan.erase.pilot import bh, p_one_sided
    assert np.allclose(bh([0.01, 0.04, 0.03, 0.2]), [0.04, 0.16 / 3, 0.16 / 3, 0.2])
    assert p_one_sided(np.arange(99) / 100, -1) == 0.01 and p_one_sided(np.zeros(9), 0.0) == 1.0


def test_hans_metrics_matches_eval_metrics():
    from khoaluan.eval.metrics import hans_table, preds_frame
    rng = np.random.default_rng(4)
    n = 600
    meta = pd.DataFrame({"uid": [f"u{i}" for i in range(n)], "y": rng.integers(0, 2, n),
                         "heuristic": rng.choice(HEURS, n), "subcase": "s"})
    lg = rng.normal(size=(n, 2))
    ref, _ = hans_table(preds_frame(meta, lg))
    mine = hans_metrics(lg, meta.heuristic.to_numpy(), meta.y.to_numpy())
    for t in ("LO", "SUB", "CON"):
        assert abs(ref[f"RG_{t}"] - mine[f"RG_{t}"]) < 1e-12


# ------------------------------------------------------------------ tập fit
def _synthetic_pool(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    lo = (rng.random(n) < np.where(y == 0, 0.06, 0.006)).astype(int)        # LO+ hiếm ở C, như MNLI
    neg_p = (rng.random(n) < np.where((lo == 1) & (y == 1), 0.65, 0.2)).astype(int)   # confound phủ định
    neg_h = (rng.random(n) < np.where(y == 1, 0.35, 0.12)).astype(int)
    hyp = rng.integers(3, 20, n) - 2 * lo
    df = pd.DataFrame({"uid": [f"mnli_train:r{i}" for i in range(n)], "y": y, "lo": lo, "neg_p": neg_p, "neg_h": neg_h,
                       "hyp_len": hyp, "prem_len": rng.integers(8, 30, n), "genre": rng.choice(list("abcde"), n)})
    df["neg_pattern"] = "P" + df.neg_p.astype(str) + "H" + df.neg_h.astype(str)
    df["hyp_len_bin"] = df.hyp_len.map(hyp_len_bin)
    return df


def test_mnli_sets_balance_matching_disjoint_deterministic():
    pool = _synthetic_pool()
    kw = dict(seed=7, n_label=2000, n_label_big=4000, n_resample=3)
    a, rep = SETS.build_mnli_sets(pool, **kw)
    b, _ = SETS.build_mnli_sets(pool, **kw)
    assert a["c_LO"].uid.tolist() == b["c_LO"].uid.tolist()                    # tất định
    assert a["c_LO_resamples"].uid.tolist() == b["c_LO_resamples"].uid.tolist()
    lo = a["c_LO"]
    n_star = rep["n_star_LO+C_fit"]

    def balanced(df):
        cnt = df.groupby(["z", "y"]).size()
        return cnt.loc[(1, 1)] == cnt.loc[(1, 0)] == cnt.loc[(0, 1)] == cnt.loc[(0, 0)] == n_star   # 2×2 cân bằng
    assert balanced(lo)
    rs = a["c_LO_resamples"]
    assert sorted(rs["resample"].unique()) == [1, 2] and all(balanced(g) for _, g in rs.groupby("resample"))
    r1 = rs[rs["resample"] == 1]
    assert set(r1.uid[r1.z == 0]) != set(lo.uid[lo.z == 0])   # mẫu khác nhau
    lab, big = a["c_LABEL"], a["c_LABEL_40K"]
    for other in (lo, rs, a["MNLI_HC"]):
        assert set(other.uid).isdisjoint(big.uid)
    assert set(lo.uid).isdisjoint(a["MNLI_HC"].uid) and set(rs.uid).isdisjoint(a["MNLI_HC"].uid)
    # ghép đúng neg_pattern trong từng nhãn -> SMD của neg_p, neg_h = 0 (trước ghép: confound mạnh ở ô C)
    for y in ("E", "C"):
        assert abs(rep["SMD_c_LO_within_label"][f"y={y}|neg_p"]) < 1e-9
        assert abs(rep["SMD_c_LO_within_label"][f"y={y}|neg_h"]) < 1e-9
    assert rep["SMD_max_abs"] < 0.1 and rep["SMD_max_abs_before"] > rep["SMD_max_abs"]
    assert abs((lo.split == "held").mean() - 0.25) < 0.03
    assert (lab.y == 0).sum() == (lab.y == 1).sum() == 1000 and (lab.z == lab.y).all()
    assert (big.y == 0).sum() == (big.y == 1).sum() == 2000
    assert set(lab.uid) <= set(big.uid)                                          # c_LABEL lồng trong c_LABEL_40K, cùng split
    assert (big.set_index("uid").loc[lab.uid, "split"].to_numpy() == lab.split.to_numpy()).all()
    # mẫu chính và MNLI-HC không phụ thuộc số lần lấy mẫu lặp
    c, _ = SETS.build_mnli_sets(pool, seed=7, n_label=2000, n_label_big=4000, n_resample=1)
    assert c["c_LO"].uid.tolist() == lo.uid.tolist() and c["MNLI_HC"].uid.tolist() == a["MNLI_HC"].uid.tolist()
    with pytest.raises(TypeError):
        SETS.build_mnli_sets(pool, np.random.default_rng(7))


def test_nn_matching_fixes_length_confound():
    """prem_len lệch theo LO (như MNLI thật: LO+ có premise dài hơn) -> ghép ngẫu nhiên trong tầng để lại |SMD| lớn;
    ghép láng giềng gần nhất trong tầng phải đưa |SMD| < 0.1 mà vẫn giữ đúng số lượng mỗi tầng."""
    pool = _synthetic_pool(40000, seed=3)
    rng = np.random.default_rng(0)
    pool["prem_len"] = pool.prem_len + np.where(pool.lo == 1, 4, 0) + rng.integers(0, 3, len(pool))
    kw = dict(seed=7, n_label=1000, n_label_big=1000, n_resample=1)
    _, rep_rand = SETS.build_mnli_sets(pool, nn_vars=None, **kw)
    sets_nn, rep_nn = SETS.build_mnli_sets(pool, **kw)
    assert rep_rand["SMD_max_abs"] > 0.2 and rep_nn["SMD_max_abs"] < 0.1
    lo = sets_nn["c_LO"]
    k = SETS._key(lo, SETS.MATCH_LEVELS[0])
    for (y, s), g in lo.assign(k=k).groupby(["y", "k"]):
        assert (g.z == 1).sum() == (g.z == 0).sum()                         # số lượng mỗi tầng khớp chính xác (level 0)


def _hans_frame(source, per_subcase, start, seed):
    from test_p2 import _sent
    rng = np.random.default_rng(seed)
    rows, k = [], start
    for h in HEURS:
        for j in range(10):
            for i in range(per_subcase):
                rows.append({"uid": f"{source}:ex{k}", "y": int(j >= 5), "heuristic": h, "subcase": f"{h[:2]}_sc{j}",
                             "template": f"temp{i % 2}", "c4": C4_OF[h], "subj_num": rng.choice(["NN", "NNS"]),
                             "sentence1": _sent(rng), "sentence2": _sent(rng)})
                k += 1
    return pd.DataFrame(rows)


def test_htb_sets():
    htb = _hans_frame("hans_train", 12, 0, 5)
    sets, rep = SETS.build_htb_sets(htb, np.random.default_rng(1))
    assert sets["c_SUB"].z.value_counts().tolist() == [120, 120]
    assert set(sets["c_SUB"][sets["c_SUB"].z == 1].c4) == {"SUB_only", "CON"}
    assert sorted(sets["c_JOINT"].z.unique()) == [0, 1, 2]
    ct = sets["c_CTRL_H"]
    for (_h, _y), g in ct.groupby(["heuristic", "y"]):
        assert (g.z == 0).sum() == (g.z == 1).sum()
    for df in sets.values():
        assert abs((df.split == "held").mean() - 0.25) < 0.05 and df.tmpl_held.any() and not df.tmpl_held.all()


# ------------------------------------------------------------------ môi trường dữ liệu dùng chung (fixture)
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    import test_p2 as T

    class F:
        def mktemp(self, n):
            return str(tmp_path_factory.mktemp(n))
    tiny = T.tiny_model.__wrapped__(F())
    dd = T.data_dir.__wrapped__(F())
    rng = np.random.default_rng(0)
    tr = pd.read_parquet(os.path.join(dd, "mnli_bin_train_all.parquet"))
    syn = _synthetic_pool(len(tr), seed=1).drop(columns=["uid", "y"])
    tr = pd.concat([tr.reset_index(drop=True), syn.reset_index(drop=True)], axis=1)
    tr["c4"] = np.where(tr.lo == 1, "LO_only", "NONE")
    tr["in_pool"] = (tr.ft_rank >= 1200).astype(int)
    tr.to_parquet(os.path.join(dd, "mnli_bin_train_all.parquet"))
    for s in ("dev_m", "dev_mm"):
        dv = pd.read_parquet(os.path.join(dd, f"mnli_bin_{s}.parquet"))
        dv["lo"] = rng.integers(0, 2, len(dv))
        dv.to_parquet(os.path.join(dd, f"mnli_bin_{s}.parquet"))
    he = _hans_frame("hans_eval", 6, 0, 2)
    hb = _hans_frame("hans_train", 8, 5000, 3)
    he.to_json(os.path.join(dd, "hans_eval.jsonl"), orient="records", lines=True)
    hb.to_json(os.path.join(dd, "hans_train_B.jsonl"), orient="records", lines=True)
    return {"tiny": tiny, "data": dd}


def _planted_run(env, tmp_path, readout):
    """Biểu diễn tổng hợp (d = 16): hướng e1 = 'liên tiếp' (SUB/CON) đẩy về E bất kể nhãn; e2 mã hoá nhãn thật;
    e3 = số chủ ngữ (không vào head). readout=False: cache pooler = X, head tuyến tính trên X (kiểm L_out).
    readout=True: cache [CLS]_24 = X, pooler = tanh(I·X), head trên tanh(X) (kiểm L_ℓ tại ℓ = L qua pooler phi tuyến)."""
    from safetensors.torch import save_file
    from khoaluan.data import load as Ld
    from khoaluan.erase.pilot import _uid_hash
    rng = np.random.default_rng(11)
    d = 16
    dd = env["data"]
    runs, out = str(tmp_path / "runs"), str(tmp_path / "erase")
    rid = "bertL-mnliB-16k-s99"
    rdir = os.path.join(runs, rid)
    os.makedirs(os.path.join(rdir, "model"))
    W = np.zeros((2, d))
    W[0, 0], W[0, 1] = 1.5, 1.0                       # logit_E = 1.5·e1 + 1.0·e2 ; logit_NE = 0  (trên pooler)
    tens = {"classifier.weight": torch.tensor(W, dtype=torch.float32), "classifier.bias": torch.zeros(2)}
    if readout:
        tens["bert.pooler.dense.weight"] = torch.eye(d)
        tens["bert.pooler.dense.bias"] = torch.zeros(d)
    save_file(tens, os.path.join(rdir, "model", "model.safetensors"))
    json.dump({"num_hidden_layers": 24, "hidden_size": d}, open(os.path.join(rdir, "model", "config.json"), "w"))
    json.dump({"cls_layers": [24]}, open(os.path.join(rdir, "std_layers.json"), "w"))
    pool_fn = np.tanh if readout else (lambda x: x)

    def reps(n, contiguous, label_sign, subj=None):
        X = rng.normal(scale=0.3, size=(n, d))
        X[:, 0] += np.where(contiguous, 3.0, -3.0)
        X[:, 1] += label_sign
        if subj is not None:
            X[:, 2] += np.where(subj == "NN", 2.0, -2.0)
        return X

    for s in ("hans_eval", "hans_train_B", "dev_m"):
        fr = Ld.eval_frame(dd, s)
        if s == "dev_m":
            X = reps(len(fr), np.zeros(len(fr), bool), np.where(fr.y == 0, 3.0, -3.0))
            X[:, 0] = rng.normal(scale=0.3, size=len(fr))            # MNLI: không có tín hiệu e1
        else:
            raw = pd.read_json(os.path.join(dd, f"{s}.jsonl"), lines=True).set_index("uid").loc[fr.uid]
            X = reps(len(fr), fr.heuristic.isin(["subsequence", "constituent"]).to_numpy(),
                     np.where(fr.y == 0, 1.0, -1.0), raw.subj_num.to_numpy())
        P = pool_fn(X)
        lg = P @ W.T
        np.save(os.path.join(rdir, f"std_{s}_pooler.npy"), P.astype(np.float16))
        np.save(os.path.join(rdir, f"std_{s}_cls.npy"), X[:, None, :].astype(np.float16))
        with open(os.path.join(rdir, f"std_{s}_uids.txt"), "w") as f:
            f.write("\n".join(fr.uid))
        pd.DataFrame({"uid": fr.uid, "pred": (lg[:, 1] > lg[:, 0]).astype(int)}).to_parquet(
            os.path.join(rdir, f"preds_{s}.parquet"))
    sets, union, _ = SETS.build_all(dd, n_label=400, n_label_big=800, n_resample=3)
    Xp = rng.normal(scale=0.3, size=(len(union), d))
    Xp[:, 1] += np.where(union.y == 0, 3.0, -3.0)
    od = os.path.join(out, rid)
    os.makedirs(od)
    np.save(os.path.join(od, "pool_pooler.npy"), pool_fn(Xp).astype(np.float16))
    np.save(os.path.join(od, "pool_logits.npy"), pool_fn(Xp) @ W.T)
    np.save(os.path.join(od, "pool_cls.npy"), Xp[:, None, :].astype(np.float16))
    json.dump({"n": len(union), "uid_sha": _uid_hash(union.uid.tolist()), "cls_layers": [24]},
              open(os.path.join(od, "pool_meta.json"), "w"))
    args = ["--data-root", dd, "--runs-root", runs, "--out", out, "--run-ids", rid, "--n-label", "400",
            "--n-label-big", "800", "--n-resample", "3", "--cls-layers", "24", "--allow-unofficial-data"]
    return out, rid, args


def test_pilot_detects_planted_effect_and_controls(env, tmp_path):
    """L_out. Kỳ vọng: c_SUB/c_JOINT làm RG cột SUB, CON giảm mạnh, p nhỏ; c_CTRL_H và cột LO không có ý nghĩa;
    c_LABEL đưa DEV-M về chance; c_LO (không mang tín hiệu nào trong biểu diễn) ổn định ≈ 0."""
    from khoaluan.erase.pilot import main
    out, rid, args = _planted_run(env, tmp_path, readout=False)
    models, cells = main(args + ["--n-null", "200", "--stages", "lout"])
    m = models[0]["concepts"]
    assert m["c_LABEL"]["positive_control_ok"] and m["c_LABEL_40K"]["positive_control_ok"]
    c = cells.set_index(["concept", "column"])
    for col in ("SUB", "CON"):
        assert c.loc[("c_SUB", col), "dRG"] < -0.5 and c.loc[("c_SUB", col), "p_N3"] < 0.01
        assert c.loc[("c_SUB", col), "dRG"] < c.loc[("c_SUB", col), "q05_N1"]
        assert c.loc[("c_SUB", col), "cell_robust_this_seed"]
        assert c.loc[("c_SUB", col), "dPE_NE"] < -0.3 and c.loc[("c_SUB", col), "p_N3_PE_NE"] < 0.01
        assert c.loc[("c_SUB", col), "flip_NE"] > 0.5 and c.loc[("c_SUB", col), "dMargin_NE"] < 0
        # c_JOINT (3 lớp, 2 lớp cùng mang e1): LEACE affine kéo về trung bình tập fit (e1 = +1) -> hiệu ứng yếu hơn
        assert c.loc[("c_JOINT", col), "dRG"] < 0
    for col in ("LO", "SUB", "CON"):
        assert c.loc[("c_CTRL_H", col), "p_N3"] > 0.05 and abs(c.loc[("c_CTRL_H", col), "dRG"]) < 0.05
    assert m["c_SUB"]["decode_balacc_held"] > 0.95 and m["c_SUB"]["probe_after_balacc_held"] < 0.6
    assert m["c_SUB"]["logit_gap_between_classes_after_fit"] < 1e-6 and m["c_SUB"]["mean_shift_rel"] > 0
    st = models[0]["c_LO_stability"]
    assert st["n_samples"] == 3 and len(st["per_sample"]) == 3 and abs(st["mean_over_samples"]["dRG_SUB"]) < 0.05
    text = open(os.path.join(out, "erase_summary.md")).read()
    assert "Ma trận cross-effect" in text and "ĐẠT" in text and "Độ ổn định hướng c_LO" in text


def test_layers_planted_effect_at_last_layer_via_pooler(env, tmp_path):
    """L_ℓ tại ℓ = L (xoá trước pooler tanh): hiệu ứng cài sẵn phải được phát hiện qua pooler phi tuyến, null đủ, BH."""
    from khoaluan.erase.pilot import main
    out, rid, args = _planted_run(env, tmp_path, readout=True)
    main(args + ["--stages", "lmid", "--layers", "24", "--null-layers", "24", "--all-token-layers", "",
                 "--n-null-last", "200"])
    res = json.load(open(os.path.join(out, rid, "lmid_model.json")))
    assert res["checks"]["readout_L24_reproduces_preds_hans_eval"] >= 0.99
    pc = res["configs"]["c_LABEL|24|cls"]                     # xoá nhãn trước pooler -> DEV-M (60 ví dụ) tụt mạnh
    assert pc["devm_acc_base"] > 0.95 and pc["after"]["devm_acc"] <= pc["devm_acc_base"] - 0.3
    cells = pd.read_csv(os.path.join(out, "lmid_cells_all.csv")).set_index(["concept", "column"])
    for col in ("SUB", "CON"):
        r = cells.loc[("c_SUB", col)]
        assert r.dRG < -0.5 and r.p_N3 < 0.01 and r.q_bh < 0.05 and bool(r.cell_robust_this_seed)
    for col in ("LO", "SUB", "CON"):
        assert abs(cells.loc[("c_CTRL_H", col), "dRG"]) < 0.05
        assert cells.loc[("c_CON", col), "p_N3"] > 0.01 or cells.loc[("c_CON", col), "dRG"] > -0.05
    assert os.path.exists(os.path.join(out, rid, "lmid_nulls.npz"))
    text = open(os.path.join(out, "lmid_summary.md")).read()
    assert "Control dương theo tầng" in text and "Ma trận ΔRG theo tầng" in text


# ------------------------------------------------------------------ can thiệp trong mô hình (SPEC_ERASURE §11.4)
def _tiny_bert(layers=3, seed=0):
    from transformers import BertConfig, BertForSequenceClassification
    torch.manual_seed(seed)
    cfg = BertConfig(vocab_size=60, hidden_size=32, num_hidden_layers=layers, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64, num_labels=2)
    return BertForSequenceClassification(cfg).eval()


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, 60, (5, 11), generator=g)
    am = torch.ones_like(ids)
    am[0, 7:] = 0
    am[3, 4:] = 0
    return {"input_ids": ids, "attention_mask": am, "token_type_ids": torch.zeros_like(ids)}


def _rand_eraser(d, rank=1, seed=0):
    from khoaluan.erase.leace_fast import TorchEraser
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, rank, generator=g))
    A = torch.randn(d, rank, generator=g)                  # chiếu xiên (P_L ≠ P_Rᵀ) như LEACE
    return TorchEraser(torch.randn(d, generator=g), A, (Q.T @ A).inverse() @ Q.T, rank)


def test_continuer_identity_and_states_match_forward():
    from khoaluan.erase.layers import Continuer
    m = _tiny_bert()
    b = _batch()
    with torch.no_grad():
        ref = m(**b, output_hidden_states=True)
    with Continuer(m, [0, 1, 2]) as c:
        lg0, _ = c.prefix(b)
        assert torch.equal(lg0, ref.logits)
        for l in (0, 1, 2):
            assert torch.allclose(c.state(l), ref.hidden_states[l], atol=1e-6)     # hidden_states[ℓ] = đầu vào layer[ℓ]
            lg, _ = c.finish(l, c.state(l))
            assert torch.allclose(lg, ref.logits, atol=1e-6)                       # eraser đồng nhất == forward gốc
    assert all(len(blk._forward_pre_hooks) == 0 for blk in m.bert.encoder.layer)   # gỡ hook khi xong


@pytest.mark.parametrize("mode", ["cls", "all"])
def test_continuer_matches_forward_hook(mode):
    """Eraser KHÁC đồng nhất: chạy tiếp từ trạng thái đã ghi == sửa đầu ra layer[ℓ−1] bằng forward hook rồi forward lại."""
    from khoaluan.erase.layers import Continuer, apply_eraser
    m = _tiny_bert()
    b = _batch(1)
    er = _rand_eraser(32, rank=2, seed=3)
    with torch.no_grad():
        ref = m(**b).logits
    with Continuer(m, [1, 2]) as c:
        c.prefix(b)
        for l in (1, 2):
            lg, _ = c.finish(l, apply_eraser(c.state(l), er, mode, b["attention_mask"]))

            def hk(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h2 = apply_eraser(h, er, mode, b["attention_mask"])
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            hdl = m.bert.encoder.layer[l - 1].register_forward_hook(hk)
            with torch.no_grad():
                lh = m(**b).logits
            hdl.remove()
            assert torch.allclose(lg, lh, atol=1e-6)
            assert (lg - ref).abs().max() > 1e-4                                    # can thiệp có tác dụng


def test_all_token_mode_ignores_padding():
    """mode 'all' chỉ sửa token thật: kết quả của một ví dụ không đổi khi thêm padding."""
    from khoaluan.erase.layers import Continuer, apply_eraser
    m = _tiny_bert()
    b = _batch(2)
    er = _rand_eraser(32, seed=4)
    with Continuer(m, [1]) as c:
        c.prefix(b)
        lg_pad, _ = c.finish(1, apply_eraser(c.state(1), er, "all", b["attention_mask"]))
        i = 3
        n = int(b["attention_mask"][i].sum())
        bi = {k: v[i:i + 1, :n] for k, v in b.items()}
        c.prefix(bi)
        lg_one, _ = c.finish(1, apply_eraser(c.state(1), er, "all", bi["attention_mask"]))
    assert torch.allclose(lg_pad[i], lg_one[0], atol=1e-5)


def test_readout_matches_model(env):
    """ℓ = L tính từ cache [CLS]_L qua pooler + head đọc từ checkpoint == logits của mô hình."""
    from transformers import AutoModelForSequenceClassification
    from khoaluan.erase.layers import Readout
    m = AutoModelForSequenceClassification.from_pretrained(env["tiny"]).eval()
    b = _batch(0)
    b["input_ids"] = b["input_ids"] % m.config.vocab_size
    with torch.no_grad():
        o = m(**b, output_hidden_states=True)
    ro = Readout(env["tiny"], torch.device("cpu"))
    lg, _ = ro.from_pre(ro.pre(o.hidden_states[-1][:, 0].numpy()))
    assert torch.allclose(lg, o.logits, atol=1e-5)
    er = _rand_eraser(m.config.hidden_size, seed=5)
    X = o.hidden_states[-1][:, 0]
    a = ro.from_pre(ro.erased_pre(ro.pre(X.numpy()), X, er))[0]
    assert torch.allclose(a, ro.from_pre(ro.pre(er(X).numpy()))[0], atol=1e-5)    # cập nhật hạng thấp == tính thẳng


def test_pilot_end_to_end_with_tiny_bert(env, tmp_path):
    from khoaluan.erase.pilot import main
    from khoaluan.p2 import main as p2_main
    dd, tiny = env["data"], env["tiny"]
    runs = str(tmp_path / "runs")
    p2_main(["--data-root", dd, "--out-root", runs, "--allow-unofficial-data", "--data", "mnliB", "--size", "100",
             "--seed", "11", "--stages", "train,infer", "--std-reps", "yes", "--save-model", "yes",
             "--set", f"model_name={tiny}", "--set", "fp16=false", "--set", "use_cpu=true",
             "--set", "num_train_epochs=1", "--std-layers", "1-2"])
    out = str(tmp_path / "erase")
    rid = "bertL-mnliB-100-s11"
    common = ["--data-root", dd, "--runs-root", runs, "--out", out, "--run-ids", rid, "--n-null", "30",
              "--n-label", "400", "--n-label-big", "800", "--n-resample", "3", "--allow-unofficial-data",
              "--batch", "16", "--cls-layers", "1,2", "--layers", "1,2", "--all-token-layers", "1",
              "--null-layers", "1,2", "--n-null-last", "20", "--n-null-mid-n3", "8", "--n-null-mid-n1", "4",
              "--null-subset-per-subcase", "3"]
    models, cells = main(common + ["--stages", "lout,lmid"])
    for f in ["pool_pooler.npy", "pool_logits.npy", "pool_cls.npy", "pool_meta.json", "erase_cells.csv", "nulls.npz",
              "erase_model.json", "lmid_model.json", "lmid_cells.csv", "lmid_nulls.npz"]:
        assert os.path.exists(os.path.join(out, rid, f)), f
    for f in ["erase_summary.md", "erase_cells_all.csv", "lmid_summary.md", "lmid_cells_all.csv",
              "sets/sets_report.json", "sets/set_c_LO.parquet", "sets/set_c_LO_resamples.parquet"]:
        assert os.path.exists(os.path.join(out, f)), f
    assert len(cells) == 5 * 3 and cells.p_N3.between(0, 1).all()
    res = json.load(open(os.path.join(out, rid, "erase_model.json")))
    assert res["checks"]["head_reproduces_preds_hans_eval"] >= 0.99
    assert set(res["concepts"]) == {"c_LABEL", "c_LABEL_40K", "c_LO", "c_SUB", "c_CON", "c_JOINT", "c_CTRL_H"}
    lm = json.load(open(os.path.join(out, rid, "lmid_model.json")))
    assert lm["checks"]["identity_continuation_max_abs_logit_diff"] < 1e-5
    assert lm["checks"]["live_reproduces_preds_hans_eval"] >= 0.99
    assert lm["checks"]["readout_L2_reproduces_preds_hans_eval"] >= 0.99
    lc = pd.read_csv(os.path.join(out, rid, "lmid_cells.csv"))
    # tầng 1: cls + all cho c_LO/c_SUB/c_CON, cls cho c_JOINT/c_CTRL_H; tầng 2 (= L): chỉ cls
    assert len(lc[(lc.layer == 1) & (lc["mode"] == "cls")]) == 15 and len(lc[(lc.layer == 1) & (lc["mode"] == "all")]) == 9
    assert len(lc[lc.layer == 2]) == 15 and set(lc[lc.layer == 2]["mode"]) == {"cls"}
    withnull = lc[lc.p_N3.notna()]
    assert set(withnull.concept) == {"c_LO", "c_SUB", "c_CON"} and len(withnull) == 2 * 9
    assert withnull.q_bh.between(0, 1).all()
    assert (withnull[withnull.layer == 1].n_N3 == 8).all() and (withnull[withnull.layer == 2].n_N3 == 20).all()
    assert "c_SUB|1|all" in lm["configs"] and "recovery_pooler_held" in lm["configs"]["c_SUB|1|cls"]
    # lần chạy thứ hai dùng lại cache MNLI-POOL đã trích, kết quả quan sát giống hệt
    _, cells2 = main(common + ["--stages", "lout"])
    assert np.allclose(cells.dRG, cells2.dRG)


def test_summary_groups_by_family_and_cross_session_script(env, tmp_path):
    """Báo cáo tách theo họ mô hình (16K vs MIX); script gộp nhiều phiên đọc đúng đầu ra."""
    import copy
    import subprocess
    from khoaluan.erase.pilot import family, summarize
    assert family("bertL-mix-16k-s22") == "bertL-mix-16k" and family("bertL-mnliB-16k-s111") == "bertL-mnliB-16k"
    rng = np.random.default_rng(0)
    concepts = {c: {"rank": 1, "rank_expected": 1, "decode_balacc_held": 0.9, "probe_after_balacc_held": 0.5,
                    "var_removed": 0.01, "delta": {"devm_acc": 0.0, "devm_acc_LOminus": 0.0, "RG_HC": 0.0},
                    "after": {"devm_acc": 0.5}, "positive_control_ok": True, "transfer_HANS_LOplus_rate": 0.9}
                for c in ["c_LABEL", "c_LO", "c_SUB", "c_CON", "c_JOINT", "c_CTRL_H"]}
    base = {"run_id": "bertL-mnliB-16k-s1", "checks": {"head_reproduces_preds_hans_eval": 1.0,
                                                      "head_reproduces_preds_dev_m": 1.0},
            "base": {k: 0.5 for k in ["RG_LO", "RG_SUB", "RG_CON", "devm_acc", "RG_HC"]}, "concepts": concepts}
    models, rows = [], []
    for rid in ["bertL-mnliB-16k-s1", "bertL-mnliB-16k-s2", "bertL-mix-16k-s1"]:
        m = copy.deepcopy(base)
        m["run_id"] = rid
        models.append(m)
        for c in ["c_LO", "c_SUB", "c_CON", "c_JOINT", "c_CTRL_H"]:
            for col in ("LO", "SUB", "CON"):
                rows.append({"run_id": rid, "concept": c, "column": col, "dRG": rng.normal(), "p_N3": 0.5,
                             "q05_N3": -0.1, "q05_N1": -0.1, "dAccE": 0, "dAccNE": 0, "dAUROC": 0,
                             "cell_robust_this_seed": False})
    rep = {"mnli": {"n_star_LO+C_fit": 370, "c_LO_total": 1480, "c_LO_fit": 1110, "SMD_max_abs": 0.05,
                    "MNLI_HC": {}}}
    text = summarize(models, pd.DataFrame(rows), rep, 100, "leace")
    assert "# bertL-mix-16k:" in text and "# bertL-mnliB-16k:" in text and "| 0/2 " in text
    for i, sub in enumerate(([models[0], models[1]], [models[2]])):
        d = tmp_path / f"s{i}"
        (d / "sets").mkdir(parents=True)
        json.dump(rep, open(d / "sets" / "sets_report.json", "w"))
        for m in sub:
            (d / m["run_id"]).mkdir()
            json.dump(m, open(d / m["run_id"] / "erase_model.json", "w"))
            pd.DataFrame([r for r in rows if r["run_id"] == m["run_id"]]).to_csv(
                d / m["run_id"] / "erase_cells.csv", index=False)
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "summarize_erase.py")
    r = subprocess.run([sys.executable, script, "--in", str(tmp_path / "s0"), str(tmp_path / "s1"),
                        "--out", str(tmp_path / "all"), "--n-null", "100"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "# bertL-mix-16k:" in open(tmp_path / "all" / "erase_summary.md").read()
