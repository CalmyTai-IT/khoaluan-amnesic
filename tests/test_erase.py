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
    a, rep = SETS.build_mnli_sets(pool, np.random.default_rng(7), n_label=2000)
    b, _ = SETS.build_mnli_sets(pool, np.random.default_rng(7), n_label=2000)
    assert a["c_LO"].uid.tolist() == b["c_LO"].uid.tolist()                    # tất định
    lo = a["c_LO"]
    n_star = rep["n_star_LO+C_fit"]
    cnt = lo.groupby(["z", "y"]).size()
    assert cnt.loc[(1, 1)] == cnt.loc[(1, 0)] == cnt.loc[(0, 1)] == cnt.loc[(0, 0)] == n_star   # 2×2 cân bằng
    assert set(lo.uid).isdisjoint(a["MNLI_HC"].uid) and set(lo.uid).isdisjoint(a["c_LABEL"].uid)
    assert set(a["MNLI_HC"].uid).isdisjoint(a["c_LABEL"].uid)
    # ghép đúng neg_pattern trong từng nhãn -> SMD của neg_p, neg_h = 0 (trước ghép: confound mạnh ở ô C)
    for y in ("E", "C"):
        assert abs(rep["SMD_c_LO_within_label"][f"y={y}|neg_p"]) < 1e-9
        assert abs(rep["SMD_c_LO_within_label"][f"y={y}|neg_h"]) < 1e-9
    assert abs((lo.split == "held").mean() - 0.25) < 0.03
    lab = a["c_LABEL"]
    assert (lab.y == 0).sum() == (lab.y == 1).sum() == 1000 and (lab.z == lab.y).all()


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


def test_pilot_detects_planted_effect_and_controls(env, tmp_path):
    """Biểu diễn tổng hợp (d = 16): head dùng hướng e1 = 'liên tiếp' (SUB/CON) để đẩy về E bất kể nhãn; e2 mã hoá nhãn
    thật; e3 = số chủ ngữ (không vào head). Kỳ vọng: c_SUB/c_JOINT làm RG cột SUB, CON giảm mạnh, p nhỏ; c_CTRL_H và
    cột LO không có ý nghĩa; c_LABEL đưa DEV-M về chance."""
    from safetensors.torch import save_file
    from khoaluan.data import load as Ld
    from khoaluan.erase.pilot import main
    rng = np.random.default_rng(11)
    d = 16
    dd = env["data"]
    runs, out = str(tmp_path / "runs"), str(tmp_path / "erase")
    rid = "bertL-mnliB-16k-s99"
    rdir = os.path.join(runs, rid)
    os.makedirs(os.path.join(rdir, "model"))
    W = np.zeros((2, d))
    W[0, 0], W[0, 1] = 1.5, 1.0                       # logit_E = 1.5·e1 + 1.0·e2 ; logit_NE = 0
    save_file({"classifier.weight": torch.tensor(W, dtype=torch.float32),
               "classifier.bias": torch.zeros(2, dtype=torch.float32)}, os.path.join(rdir, "model", "model.safetensors"))

    def reps(frame, contiguous, label_sign, subj=None):
        X = rng.normal(scale=0.3, size=(len(frame), d))
        X[:, 0] += np.where(contiguous, 3.0, -3.0)
        X[:, 1] += label_sign
        if subj is not None:
            X[:, 2] += np.where(subj == "NN", 2.0, -2.0)
        return X

    for s in ("hans_eval", "hans_train_B", "dev_m"):
        fr = Ld.eval_frame(dd, s)
        if s == "dev_m":
            X = reps(fr, np.zeros(len(fr), bool), np.where(fr.y == 0, 3.0, -3.0))
            X[:, 0] = rng.normal(scale=0.3, size=len(fr))            # MNLI: không có tín hiệu e1
        else:
            raw = pd.read_json(os.path.join(dd, f"{s}.jsonl"), lines=True).set_index("uid").loc[fr.uid]
            X = reps(fr, fr.heuristic.isin(["subsequence", "constituent"]).to_numpy(),
                     np.where(fr.y == 0, 1.0, -1.0), raw.subj_num.to_numpy())
        lg = X @ W.T
        np.save(os.path.join(rdir, f"std_{s}_pooler.npy"), X.astype(np.float16))
        with open(os.path.join(rdir, f"std_{s}_uids.txt"), "w") as f:
            f.write("\n".join(fr.uid))
        pd.DataFrame({"uid": fr.uid, "pred": (lg[:, 1] > lg[:, 0]).astype(int)}).to_parquet(
            os.path.join(rdir, f"preds_{s}.parquet"))
    sets, union, _ = SETS.build_all(dd, n_label=400)
    Xp = rng.normal(scale=0.3, size=(len(union), d))
    Xp[:, 1] += np.where(union.y == 0, 3.0, -3.0)
    os.makedirs(os.path.join(out, rid))
    np.save(os.path.join(out, rid, "pool_pooler.npy"), Xp.astype(np.float16))
    models, cells = main(["--data-root", dd, "--runs-root", runs, "--out", out, "--run-ids", rid, "--n-null", "200",
                          "--n-label", "400", "--allow-unofficial-data"])
    m = models[0]["concepts"]
    assert models[0]["concepts"]["c_LABEL"]["positive_control_ok"]
    c = cells.set_index(["concept", "column"])
    for col in ("SUB", "CON"):
        assert c.loc[("c_SUB", col), "dRG"] < -0.5 and c.loc[("c_SUB", col), "p_N3"] < 0.01
        assert c.loc[("c_SUB", col), "dRG"] < c.loc[("c_SUB", col), "q05_N1"]
        assert c.loc[("c_SUB", col), "cell_robust_this_seed"]
        # c_JOINT (3 lớp, 2 lớp cùng mang e1): LEACE affine kéo về trung bình tập fit (e1 = +1) -> hiệu ứng yếu hơn
        assert c.loc[("c_JOINT", col), "dRG"] < 0
    for col in ("LO", "SUB", "CON"):
        assert c.loc[("c_CTRL_H", col), "p_N3"] > 0.05 and abs(c.loc[("c_CTRL_H", col), "dRG"]) < 0.05
    assert m["c_SUB"]["decode_balacc_held"] > 0.95 and m["c_SUB"]["probe_after_balacc_held"] < 0.6
    assert m["c_SUB"]["logit_gap_between_classes_after_fit"] < 1e-6
    text = open(os.path.join(out, "erase_summary.md")).read()
    assert "Ma trận cross-effect" in text and "ĐẠT" in text


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
    models, cells = main(["--data-root", dd, "--runs-root", runs, "--out", out, "--run-ids", "bertL-mnliB-100-s11",
                          "--n-null", "30", "--n-label", "400", "--allow-unofficial-data", "--batch", "16"])
    rid = "bertL-mnliB-100-s11"
    for f in ["pool_pooler.npy", "pool_logits.npy", "erase_cells.csv", "nulls.npz", "erase_model.json"]:
        assert os.path.exists(os.path.join(out, rid, f)), f
    for f in ["erase_summary.md", "erase_cells_all.csv", "sets/sets_report.json", "sets/set_c_LO.parquet"]:
        assert os.path.exists(os.path.join(out, f)), f
    assert len(cells) == 5 * 3 and cells.p_N3.between(0, 1).all()
    res = json.load(open(os.path.join(out, rid, "erase_model.json")))
    assert res["checks"]["head_reproduces_preds_hans_eval"] >= 0.99
    assert set(res["concepts"]) == {"c_LABEL", "c_LO", "c_SUB", "c_CON", "c_JOINT", "c_CTRL_H"}
    # lần chạy thứ hai dùng lại cache MNLI-POOL đã trích, kết quả quan sát giống hệt
    _, cells2 = main(["--data-root", dd, "--runs-root", runs, "--out", out, "--run-ids", rid, "--n-null", "30",
                      "--n-label", "400", "--allow-unofficial-data"])
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
