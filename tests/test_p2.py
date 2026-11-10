"""Unit test cho code P2 (load / metrics / forward / lr / p2 đầu-cuối), chạy offline trên CPU với BERT 2 tầng tự tạo.
Dữ liệu giả mô phỏng ĐÚNG định dạng đầu ra prepare_data.py v1.2 (tên file, cột, uid có tiền tố nguồn,
HANS-train/eval trùng pairID). Không cần Internet, không cần GPU.

    python -m pytest -q tests/test_p2.py
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # test chỉ chạy CPU (trên Kaggle T4 x2 tránh chốt chặn >1 GPU)
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from khoaluan.data import load as L  # noqa: E402
from khoaluan.eval import metrics as M  # noqa: E402
from khoaluan.probe import lr as P  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
NOUNS = ["cat", "dog", "doctor", "lawyer", "artist", "judge"]
VERBS = ["saw", "helped", "called", "met"]
WORDS = ("[PAD] [UNK] [CLS] [SEP] [MASK] given that does it follow yes or no ? . the and if not "
         + " ".join(NOUNS + VERBS)).split()
HEURS = ["lexical_overlap", "subsequence", "constituent"]


# ------------------------------------------------------------------ fixture
def _sent(rng):
    a, b = rng.choice(NOUNS, 2, replace=False)
    return f"the {a} {rng.choice(VERBS)} the {b} ."


def _fingerprint(values):
    return L.fingerprint(values)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
    d = str(tmp_path_factory.mktemp("tiny_bert"))
    tok = BertTokenizer(vocab={w: i for i, w in enumerate(WORDS)}, do_lower_case=True)
    tok.model_max_length = 128
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(WORDS), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=128, num_labels=2)
    BertForSequenceClassification(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("data"))
    rng = np.random.default_rng(0)
    n_tr = 8100
    tr = pd.DataFrame({"premise": [_sent(rng) for _ in range(n_tr)], "hypothesis": [_sent(rng) for _ in range(n_tr)],
                       "y": rng.integers(0, 2, n_tr), "row_idx": np.arange(n_tr) * 3})
    tr["uid"] = "mnli_train:r" + tr.row_idx.astype(str)
    tr["ft_rank"] = np.random.default_rng(1234).permutation(n_tr)
    for n in L.FT_SIZES:
        with open(os.path.join(d, f"ft_uids_{n}.txt"), "w") as f:
            f.write("\n".join(tr.loc[tr.ft_rank < n].sort_values("ft_rank").uid))
    tr["in_pool"] = (tr.ft_rank >= 16000).astype(int)
    tr.to_parquet(os.path.join(d, "mnli_bin_train_all.parquet"))
    for s in ("dev_m", "dev_mm"):
        n = 60
        dv = pd.DataFrame({"premise": [_sent(rng) for _ in range(n)], "hypothesis": [_sent(rng) for _ in range(n)],
                           "y": np.array([0, 1] * (n // 2)), "row_idx": np.arange(n)})
        dv["uid"] = f"mnli_{s}:r" + dv.row_idx.astype(str)
        dv.to_parquet(os.path.join(d, f"mnli_bin_{s}.parquet"))

    def hans(source, per_subcase, start):
        rows, k = [], start
        for h in HEURS:
            for j in range(10):
                lab = "entailment" if j < 5 else "non-entailment"
                for _ in range(per_subcase):
                    rows.append({"sentence1": _sent(rng), "sentence2": _sent(rng), "gold_label": lab, "heuristic": h,
                                 "subcase": f"{h[:2]}_sc{j}", "template": f"temp{j % 2}", "pairID": f"ex{k}"})
                    k += 1
        df = pd.DataFrame(rows)
        df["uid"] = source + ":" + df.pairID
        df["y"] = (df.gold_label != "entailment").astype(int)
        return df

    he = hans("hans_eval", 4, 0)
    ht = hans("hans_train", 4, 0)              # cùng dải pairID với eval, như dữ liệu thật
    ha = ht.groupby("subcase", group_keys=False).head(2)
    hb = ht.drop(index=ha.index)
    he.to_json(os.path.join(d, "hans_eval.jsonl"), orient="records", lines=True)
    ha.to_json(os.path.join(d, "hans_train_A.jsonl"), orient="records", lines=True)
    hb.to_json(os.path.join(d, "hans_train_B.jsonl"), orient="records", lines=True)
    with open(os.path.join(d, "mix_uids.txt"), "w") as f:
        f.write("\n".join(list(tr.loc[tr.ft_rank < 8000].sort_values("ft_rank").uid) + list(ha.uid)))
    man = {"script_version": "1.2", "data_seed": 1234,
           "fingerprint_ft16k": _fingerprint(tr.loc[tr.ft_rank < 16000].sort_values("ft_rank").uid),
           "fingerprint_hans_A": _fingerprint(ha.uid),
           "mnli_source": {"hf_revision_resolved": "deadbeef"}}
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(man, f)
    return d


def _expected(data_dir):
    man = L.load_manifest(data_dir)
    return {"script_version": "1.2", "data_seed": 1234, "fingerprint_ft16k": man["fingerprint_ft16k"],
            "fingerprint_hans_A": man["fingerprint_hans_A"], "hf_revision_resolved": "deadbeef"}


# ------------------------------------------------------------------ load
def test_fingerprint_matches_prepare_data():
    from khoaluan.data.prepare_data import fingerprint as fp_prep
    vals = ["mnli_train:r1", "mnli_train:r7", "hans_train:ex3"]
    assert L.fingerprint(vals) == fp_prep(vals)


def test_verify_data_pass_and_fail(data_dir, tmp_path):
    assert L.verify_data(data_dir, _expected(data_dir))["ok"]
    with pytest.raises(ValueError):
        L.verify_data(data_dir)                       # hằng số chính thức khác dữ liệu giả -> phải dừng
    bad = dict(_expected(data_dir), fingerprint_ft16k="0" * 16)
    with pytest.raises(ValueError):
        L.verify_data(data_dir, bad)


def test_find_data_dir(data_dir, tmp_path):
    assert L.find_data_dir(os.path.dirname(data_dir)) == data_dir


def test_train_frames(data_dir):
    f25 = L.train_frame(data_dir, "mnliB", 25)
    assert len(f25) == 25 and list(f25.columns) == L.STD_COLS
    f100 = L.train_frame(data_dir, "mnliB", 100)
    assert set(f25.uid) <= set(f100.uid)              # lồng nhau
    mix = L.train_frame(data_dir, "mix", 16000)
    assert len(mix) == 8000 + 60 and mix.uid.is_unique
    assert mix.uid.str.startswith("hans_train:").sum() == 60
    with pytest.raises(ValueError):
        L.train_frame(data_dir, "mnliB", 500)


def test_eval_frames_and_hans_subset(data_dir):
    he = L.eval_frame(data_dir, "hans_eval")
    assert len(he) == 120 and he.uid.str.startswith("hans_eval:").all()
    sub1 = L.hans_contrast_subset(he, n_total=60, seed=1234)
    sub2 = L.hans_contrast_subset(he, n_total=60, seed=1234)
    assert list(sub1.uid) == list(sub2.uid)                            # tất định
    assert sub1.subcase.value_counts().tolist() == [2] * 30          # phân tầng theo subcase
    odd = L.hans_contrast_subset(he, n_total=65, seed=1234).subcase.value_counts()
    assert odd.max() == 3 and odd.min() == 2 and odd.sum() == 65


# ------------------------------------------------------------------ metrics
def test_hans_table_and_q4():
    rows = []
    for h in HEURS:
        rows += [{"uid": f"{h}e{i}", "y": 0, "pred": 0, "heuristic": h, "subcase": "a", "p_E": .9} for i in range(10)]
        rows += [{"uid": f"{h}n{i}", "y": 1, "pred": int(i < 2), "heuristic": h, "subcase": "b", "p_E": .8}
                 for i in range(10)]
    t, sub = M.hans_table(pd.DataFrame(rows))
    assert t["hans_E_LO"] == 1.0 and abs(t["hans_NE_LO"] - 0.2) < 1e-12 and abs(t["RG_LO"] - 0.8) < 1e-12
    assert abs(t["hans_NE_mean3"] - 0.2) < 1e-12 and len(sub) == 6
    q = M.q4_check(dict(t, devm_acc=0.9), "mnli")
    assert q["Q4_pass"][1] is True
    q = M.q4_check(dict(t, devm_acc=0.84), "mnli")
    assert q["Q4_pass"][1] is False
    assert M.is_degenerate(1000, 0.55) and not M.is_degenerate(100, 0.5) and not M.is_degenerate(16000, 0.8)


def test_preds_frame():
    fr = pd.DataFrame({"uid": ["a", "b"], "y": [0, 1], "premise": ["x", "y"], "hypothesis": ["x", "y"]})
    p = M.preds_frame(fr, np.array([[2.0, 0.0], [0.0, 1.0]]))
    assert list(p.pred) == [0, 1] and abs(p.p_E[0] - 1 / (1 + math.exp(-2))) < 1e-6
    with pytest.raises(ValueError):
        M.preds_frame(fr, np.zeros((3, 2)))


# ------------------------------------------------------------------ lr
def test_lr_splits_by_example_and_disjoint():
    y = np.array([0, 1] * 3346)
    for tr, te in P.splits_mnli(y):
        assert len(tr) == len(te) == 1000 and not set(tr) & set(te)
        assert abs(y[tr].mean() - y.mean()) < 0.01
    strata = np.array([f"{h}|{l}" for h in HEURS for l in (0, 1) for _ in range(20)])
    sp = P.splits_hans(strata)
    used = [set(tr) | set(te) for tr, te in sp]
    for i in range(5):
        tr, te = sp[i]
        assert not set(tr) & set(te)
        assert pd.Series(strata[tr]).value_counts().tolist() == [2] * 6
        for j in range(i + 1, 5):
            assert not used[i] & used[j]                 # 5 phần rời nhau


def test_lr_learns_signal_and_sign_convention():
    rng = np.random.default_rng(0)
    n, d = 400, 8
    y = rng.integers(0, 2, n)
    h_no = rng.normal(size=(n, 1, d)).astype(np.float32)
    h_yes = h_no + (2 * y - 1)[:, None, None] * 1.0 * np.eye(d, dtype=np.float32)[0]
    assert np.allclose(P.dlk_x(h_yes[:, 0], h_no[:, 0])[:, 0], 2 * y - 1)
    sp = P.splits_mnli(y, n_runs=2, n_sample=400)
    df = P.run_lr(h_yes, h_no, y, sp, layers=[5], layer_ids=[5], dataset="toy")
    assert (df.acc > 0.95).all() and set(df.variant) == {"faithful", "robust"}


def test_fp16_diag():
    a = np.ones((5, 2, 4), np.float32)
    diag = P.fp16_storage_diagnostic(a, a.copy(), [0, 1])
    assert (diag.frac_zero_diff == 1.0).all()


# ------------------------------------------------------------------ forward
def test_contrast_text_format():
    y, n = __import__("khoaluan.eval.forward", fromlist=["x"]).contrast_texts("The doctor saw the cat .",
                                                                                 "The cat saw the doctor .")
    assert y == "Given that The doctor saw the cat . Does it follow that The cat saw the doctor . Yes or no? Yes"
    assert n.endswith("Yes or no? No") and n[:-3] == y[:-4]


def test_run_pairs_order_and_logits(tiny_model, data_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from khoaluan.eval.forward import run_pairs
    tok = AutoTokenizer.from_pretrained(tiny_model)
    model = AutoModelForSequenceClassification.from_pretrained(tiny_model).eval()
    fr = L.eval_frame(data_dir, "dev_m")
    a = run_pairs(model, tok, fr, 128, batch_size=7, fp16=False, cls_layers=[1, 2])
    b = run_pairs(model, tok, fr, 128, batch_size=1, fp16=False, cls_layers=[1, 2])
    assert np.allclose(a["logits"], b["logits"], atol=1e-5) and np.allclose(a["cls"], b["cls"], atol=1e-5)
    with torch.no_grad():  # đối chiếu từng ví dụ với forward chuẩn của model
        e = tok(fr.premise[3], fr.hypothesis[3], return_tensors="pt")
        o = model(**e, output_hidden_states=True)
    assert np.allclose(a["logits"][3], o.logits[0].numpy(), atol=1e-5)
    assert np.allclose(a["cls"][3, 1], o.hidden_states[2][0, 0].numpy(), atol=1e-5)


def test_run_contrast_layer0_identity(tiny_model, data_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from khoaluan.eval.forward import run_contrast
    tok = AutoTokenizer.from_pretrained(tiny_model)
    model = AutoModelForSequenceClassification.from_pretrained(tiny_model).eval()
    fr = L.eval_frame(data_dir, "dev_m").head(10)
    arr = {f"{k}_{a}": np.zeros((10, 3 if k == "cls" else 1, 32), np.float32)
           for k in ("cls", "last", "mean") for a in ("yes", "no")}
    st = run_contrast(model, tok, fr, 4, False, [0, 1, 2], [2], arr)
    assert st["n_truncated"] == 0 and st["n_identical_pairs"] == 0
    # tầng 0 (embedding) của [CLS] giống hệt nhau giữa Yes/No -> LR tầng 0 phải ở mức chance
    assert np.array_equal(arr["cls_yes"][:, 0], arr["cls_no"][:, 0])
    assert not np.allclose(arr["cls_yes"][:, 2], arr["cls_no"][:, 2])
    assert not np.allclose(arr["last_yes"], arr["last_no"])


# ------------------------------------------------------------------ đầu-cuối
def _args(data_dir, out, tiny_model, *extra):
    return ["--data-root", data_dir, "--out-root", out, "--allow-unofficial-data",
            "--set", f"model_name={tiny_model}", "--set", "fp16=false", "--set", "use_cpu=true",
            "--set", "num_train_epochs=2", "--set", "logging_steps=1", "--std-layers", "1-2",
            "--cp-layers", "all", "--hans-cp-n", "120", "--dr1-layers", "1-2", *extra]


def test_end_to_end_finetune_pipeline(tiny_model, data_dir, tmp_path):
    from khoaluan.p2 import main
    out = str(tmp_path / "out")
    rec = main(_args(data_dir, out, tiny_model, "--data", "mnliB", "--size", "25", "--seed", "11",
                     "--account", "TV-B", "--std-reps", "yes", "--save-model", "yes"))
    d = os.path.join(out, "bertL-mnliB-25-s11")
    assert rec["train"]["steps"] == math.ceil(25 / 8) * 2 == rec["train"]["expected_steps"]
    assert rec["train"]["train_batch_size"] == 8
    for f in ["run_record.json", "p2_summary.txt", "p2_metrics.json", "preds_dev_m.parquet", "preds_hans_eval.parquet",
              "preds_hans_train_B.parquet", "std_hans_eval_pooler.npy", "std_hans_eval_cls.npy", "hans_subcase.csv",
              "cp_dev_m_cls_yes.npy", "cp_hans_eval_meta.parquet", "lr_raw.csv", "lr_summary.csv",
              "cp_fp16_diag.csv", "model/config.json", "train_log.json", "ccs_summary.csv", "dr1_summary.csv"]:
        assert os.path.exists(os.path.join(d, f)), f
    rr = json.load(open(os.path.join(d, "run_record.json")))
    assert rr["run_id"] == "bertL-mnliB-25-s11" and rr["seed"] == 11 and rr["data_seed"] == 1234
    assert rr["notes"].startswith("UNOFFICIAL-DATA") and rr["devm_acc"] is not None
    assert np.load(os.path.join(d, "std_hans_eval_cls.npy")).shape == (120, 2, 32)
    assert np.load(os.path.join(d, "cp_dev_m_cls_yes.npy")).shape == (60, 3, 32)
    preds = pd.read_parquet(os.path.join(d, "preds_hans_eval.parquet"))
    assert preds.uid.is_unique and len(preds) == 120
    lr = pd.read_csv(os.path.join(d, "lr_raw.csv"))
    assert set(lr.layer) == {0, 1, 2} and set(lr.dataset) == {"dev_m", "hans_eval"} and lr.run.max() == 5
    # pooler cache -> classifier phải tái tạo logits đã lưu (điều kiện cho baseline ở P4)
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(os.path.join(d, "model")).eval()
    pool = torch.tensor(np.load(os.path.join(d, "std_dev_m_pooler.npy")).astype(np.float32))
    with torch.no_grad():
        lg = m.classifier(pool).numpy()
    pdm = pd.read_parquet(os.path.join(d, "preds_dev_m.parquet"))
    assert np.allclose(lg, pdm[["logit_0", "logit_1"]].to_numpy(), atol=2e-2)
    assert (lg.argmax(1) == pdm.pred.to_numpy()).mean() > 0.95


def test_training_is_seed_deterministic(tiny_model, data_dir, tmp_path):
    from khoaluan.p2 import main
    r1 = main(_args(data_dir, str(tmp_path / "a"), tiny_model, "--data", "mnliB", "--size", "25", "--seed", "22",
                    "--stages", "train"))
    r2 = main(_args(data_dir, str(tmp_path / "b"), tiny_model, "--data", "mnliB", "--size", "25", "--seed", "22",
                    "--stages", "train"))
    r3 = main(_args(data_dir, str(tmp_path / "c"), tiny_model, "--data", "mnliB", "--size", "25", "--seed", "33",
                    "--stages", "train"))
    assert r1["train"]["train_loss_mean"] == r2["train"]["train_loss_mean"]
    assert r1["train"]["train_loss_mean"] != r3["train"]["train_loss_mean"]


def test_pretrained_no_head_path(tiny_model, data_dir, tmp_path):
    from khoaluan.p2 import main
    rec = main(_args(data_dir, str(tmp_path / "pt"), tiny_model, "--data", "pt", "--std-reps", "yes",
                     "--aux-layers", "2"))
    d = os.path.join(str(tmp_path / "pt"), "bertL-pt")
    assert "train" not in rec["stages"] and rec["metrics"] == {"degenerate": False}
    assert not os.path.exists(os.path.join(d, "preds_dev_m.parquet"))
    for f in ["lr_aux_summary.csv", "ccs_aux_summary.csv", "cp_dev_m_last_yes.npy", "dr1_summary.csv"]:
        assert os.path.exists(os.path.join(d, f)), f
    aux = pd.read_csv(os.path.join(d, "lr_aux_summary.csv"))
    assert set(aux.token) == {"last", "mean"} and set(aux.layer) == {2}
    assert os.path.exists(os.path.join(d, "std_dev_m_pooler.npy")) and os.path.exists(os.path.join(d, "lr_summary.csv"))


def test_guards(tiny_model, data_dir, tmp_path):
    from khoaluan.p2 import main
    with pytest.raises(SystemExit):
        main(_args(data_dir, str(tmp_path), tiny_model, "--data", "mnliB", "--size", "25", "--seed", "11",
                   "--stages", "train,bogus"))
    with pytest.raises(SystemExit):
        main(_args(data_dir, str(tmp_path), tiny_model, "--data", "mnliB", "--size", "500", "--seed", "11"))
    with pytest.raises(ValueError):   # không có --allow-unofficial-data -> kiểm vân tay chính thức phải dừng
        main(["--data-root", data_dir, "--out-root", str(tmp_path), "--data", "mnliB", "--size", "25",
              "--seed", "11", "--stages", "train"])


def test_is_key_model():
    from khoaluan.p2 import is_key_model
    assert is_key_model("pt", 0, 0)
    assert is_key_model("mnliB", 16000, 11) and is_key_model("mnliB", 16000, 111)   # seed+100 thay seed 11 thoái hoá
    assert is_key_model("mnliB", 8000, 11) and is_key_model("mix", 16000, 211)
    assert not is_key_model("mnliB", 16000, 22) and not is_key_model("mnliB", 1000, 11)


def test_pipeline_learns_learnable_labels(tiny_model, data_dir, tmp_path):
    """Lỗ hổng của bộ test cũ: chỉ dùng nhãn ngẫu nhiên nên không kiểm được là pipeline HỌC được.
    Ở đây y = 1 khi hypothesis có từ 'not' -> DEV acc phải gần 1 (nhãn gắn đúng input, Trainer cập nhật trọng số,
    inference đọc đúng mô hình đã train)."""
    import shutil
    from khoaluan.p2 import main
    d = str(tmp_path / "learn_data")
    shutil.copytree(data_dir, d)
    rng = np.random.default_rng(5)
    for f in ["mnli_bin_train_all.parquet", "mnli_bin_dev_m.parquet", "mnli_bin_dev_mm.parquet"]:
        df = pd.read_parquet(os.path.join(d, f))
        y = rng.integers(0, 2, len(df))
        df["hypothesis"] = [h.replace(" the ", " not the ", 1) if t else h for h, t in zip(df.hypothesis, y)]
        df["y"] = y
        df.to_parquet(os.path.join(d, f))
    rec = main(_args(d, str(tmp_path / "out"), tiny_model, "--data", "mnliB", "--size", "1000", "--seed", "11",
                     "--stages", "train,infer", "--set", "learning_rate=5.0e-4", "--set", "num_train_epochs=3",
                     "--set", "logging_steps=50"))
    assert rec["metrics"]["devm_acc"] >= 0.9 and rec["train"]["train_loss_mean"] < 0.5


# ------------------------------------------------------------------ dừng sớm run thoái hoá + bỏ stage sau
def test_abort_stuck_run_skips_later_stages(tiny_model, data_dir, tmp_path):
    from khoaluan.p2 import main
    out = str(tmp_path / "ab")
    # lr = 0 -> loss kẹt ~ln2; hạ ngưỡng áp dụng (n ≥ 20) để thử trên FT-25; 4 bước/epoch, log mỗi bước
    rec = main(_args(data_dir, out, tiny_model, "--data", "mnliB", "--size", "25", "--seed", "11",
                     "--save-model", "yes", "--set", "learning_rate=0.0", "--set", "abort_min_n=20",
                     "--set", "abort_loss=0.5"))
    d = os.path.join(out, "bertL-mnliB-25-s11")
    assert rec["degenerate"] and rec["train"]["aborted"] and rec["train"]["steps"] == 4
    assert rec["skipped_stages"] == ["infer", "contrast", "lr", "ccs", "dr1"]
    assert not os.path.exists(os.path.join(d, "model")) and not os.path.exists(os.path.join(d, "lr_raw.csv"))
    rr = json.load(open(os.path.join(d, "run_record.json")))
    assert rr["degenerate"] is True and "THOÁI HOÁ" in rr["notes"]


def test_abort_rule_not_triggered_when_threshold_high(tiny_model, data_dir, tmp_path):
    from khoaluan.p2 import main
    rec = main(_args(data_dir, str(tmp_path / "nb"), tiny_model, "--data", "mnliB", "--size", "25", "--seed", "11",
                     "--stages", "train", "--set", "abort_min_n=20", "--set", "abort_loss=5.0"))
    assert rec["train"]["abort_rule"]["enabled"] and not rec["train"]["aborted"] and rec["train"]["steps"] == 8
    assert rec["train"]["abort_rule"]["epoch1_second_half_mean_loss"] is not None


def test_stuck_after_epoch1_window():
    from khoaluan.train.finetune import stuck_after_epoch1
    hist = [{"loss": 0.2, "step": 100}, {"loss": 0.70, "step": 600}, {"loss": 0.72, "step": 1000},
            {"loss": 0.1, "step": 1100}]
    assert stuck_after_epoch1(hist, 1000, 0.65) == (True, pytest.approx(0.71))
    assert stuck_after_epoch1(hist, 1000, 0.75)[0] is False
    assert stuck_after_epoch1([], 1000) == (False, None)


# ------------------------------------------------------------------ hàng đợi tự chạy lại seed+100
def _fake_runner(out_root, degenerate_seeds, fail_seeds=()):
    from khoaluan.p2_queue import run_id_for
    calls = []

    def runner(argv):
        a = dict(zip(argv[::2], argv[1::2]))
        data, size, seed = a["--data"], int(a["--size"]), int(a["--seed"])
        calls.append((data, size, seed))
        if seed in fail_seeds:
            return 3
        rid = run_id_for(data, size, seed)
        os.makedirs(os.path.join(out_root, rid), exist_ok=True)
        deg = seed in degenerate_seeds
        json.dump({"run_id": rid, "degenerate": deg, "degenerate_reason": "giả lập" if deg else None},
                  open(os.path.join(out_root, rid, "p2_metrics.json"), "w"))
        return 0
    return runner, calls


def test_queue_retries_degenerate_and_stops_on_error(tmp_path):
    from khoaluan.p2_queue import run_queue
    out = str(tmp_path)
    runner, calls = _fake_runner(out, degenerate_seeds={11, 22, 122, 222})
    rep = run_queue(["mnliB-16000-11", "mnliB-16000-22", "pt-0-0"], [], out, runner=runner, log=lambda m: None)
    assert [c[2] for c in calls] == [11, 111, 22, 122, 222, 0]          # 11 -> 111 đạt; 22 hỏng cả 3 lần; PT 1 lần
    assert rep["stopped_on_error"] is None and sum(r["degenerate"] for r in rep["runs"]) == 4
    runner, calls = _fake_runner(out, degenerate_seeds=set(), fail_seeds={33})
    rep = run_queue(["mnliB-16000-33", "mnliB-16000-44"], [], out, runner=runner, log=lambda m: None)
    assert rep["stopped_on_error"] == "bertL-mnliB-16k-s33" and len(calls) == 1


def test_plan_covers_spec_models():
    import yaml
    plan = yaml.safe_load(open(os.path.join(REPO, "configs", "p2", "plan.yaml")))
    specs = [s for sess in plan.values() for q in sess.values() for s in q]
    assert len(specs) == len(set(specs)) == 21
    want = ({f"mnliB-16000-{s}" for s in (11, 22, 33, 44, 55)} | {f"mix-16000-{s}" for s in (11, 22, 33)}
            | {f"mnliB-{n}-{s}" for n in (25, 100, 1000, 8000) for s in (11, 22, 33)} | {"pt-0-0"})
    assert set(specs) == want


# ------------------------------------------------------------------ CCS
def test_ccs_finds_unsupervised_truth_direction():
    from khoaluan.probe.ccs import fit_eval_ccs
    rng = np.random.default_rng(0)
    n, d = 600, 16
    y = rng.integers(0, 2, n)
    base = rng.normal(size=(n, d)).astype(np.float32)
    v = np.zeros(d, np.float32); v[0] = 3.0
    s = (2 * y - 1)[:, None]
    x0, x1 = base + s * v, base - s * v      # "Yes" và "No" lệch ngược chiều theo sự thật — CCS không thấy y khi train
    acc, loss = fit_eval_ccs(x0[:300], x1[:300], x0[300:], x1[300:], y[300:], nepochs=300, seed=1, device="cpu")
    assert acc > 0.9 and loss < 0.2
    acc_mlp, _ = fit_eval_ccs(x0[:300], x1[:300], x0[300:], x1[300:], y[300:], linear=False, nepochs=300, seed=1,
                              device="cpu")
    assert acc_mlp > 0.85


def test_ccs_loss_matches_dlk_formula_and_chance_floor():
    from khoaluan.probe.ccs import ccs_loss, fit_eval_ccs
    p0 = torch.tensor([[[0.9], [0.2]]]); p1 = torch.tensor([[[0.3], [0.6]]])
    want = ((0.3 ** 2 + 0.2 ** 2) / 2) + (((0.9 - 0.7) ** 2 + (0.2 - 0.4) ** 2) / 2)
    assert abs(float(ccs_loss(p0, p1)[0]) - want) < 1e-6
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 8)).astype(np.float32)
    acc, _ = fit_eval_ccs(x[:200], x[:200] + 0.01, x[200:], x[200:] + 0.01, rng.integers(0, 2, 200), nepochs=50,
                          device="cpu")
    assert 0.5 <= acc <= 0.65                       # max(acc, 1-acc) >= 0.5 theo định nghĩa


# ------------------------------------------------------------------ D-R1
def test_dr1_folds_hold_out_one_subcase_per_heuristic_label(data_dir):
    he = L.eval_frame(data_dir, "hans_eval")
    folds = P.dr1_folds(he)
    assert len(folds) == 5
    seen = set()
    for tr, te, held in folds:
        assert len(held) == 6 and not set(tr) & set(te)
        sub = he.iloc[te]
        assert sub.groupby(["heuristic", "y"]).subcase.nunique().tolist() == [1] * 6
        assert not set(he.iloc[tr].subcase) & set(held)
        seen |= set(held)
    assert seen == set(he.subcase)                   # mỗi subcase được giữ lại đúng một lần


# ------------------------------------------------------------------ báo cáo G2 (quy tắc viết trước)
def _write_run(root, run_id, *, degenerate=False, is_key=False, devm=0.92, E=0.99, NE=0.03, total=0.51,
               lr=None, ccs=None):
    d = os.path.join(root, run_id)
    os.makedirs(d, exist_ok=True)
    m = {"devm_acc": devm, "devmm_acc": devm, "hans_E_all": E, "hans_NE_mean3": NE, "hans_acc_total": total,
         "hans_NE_LO": NE, "hans_NE_SUB": NE, "hans_NE_CON": NE, "degenerate": degenerate}
    json.dump({"run_id": run_id, "degenerate": degenerate, "is_key_model": is_key, "metrics": m, "train": {}},
              open(os.path.join(d, "p2_metrics.json"), "w"))
    rows = []
    for (ds, layer), acc in (lr or {}).items():
        rows.append({"run_id": run_id, "dataset": ds, "layer": layer, "variant": "faithful", "acc_mean": acc,
                     "acc_sd": 0.01, "n_runs": 5, "conv_warn": 0})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(d, "lr_summary.csv"), index=False)
    rows = [{"run_id": run_id, "dataset": ds, "layer": layer, "variant": "linear", "acc_mean": acc, "acc_sd": 0.01,
             "n_runs": 5} for (ds, layer), acc in (ccs or {}).items()]
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(d, "ccs_summary.csv"), index=False)


def _vdl_like(root, shift=0, q4_fail=False):
    """Dựng kết quả giả GIỐNG HỆT bảng V&dL; shift = k: hàng ℓ của Bảng 6/7 nằm ở hidden_states[ℓ + k]."""
    from khoaluan.analysis import p2_report as R
    def table_curve(T):
        c = {}
        for l, v in T.items():
            if 0 <= l + shift <= 24:
                c[("dev_m", l + shift)], c[("hans_eval", l + shift)] = v[0], v[1]
        return c
    L23 = 23 + shift
    def at23(model):
        m, h = R.REF_LR23[model]
        return {("dev_m", L23): m, ("hans_eval", L23): h}
    def ccs23(model):
        if model not in R.REF_CCS23:
            return {}
        m, h = R.REF_CCS23[model]
        return {("dev_m", L23): m, ("hans_eval", L23): h}
    _write_run(root, "bertL-pt", is_key=True, lr=at23("pt"))
    for size in ("25", "100", "1k"):
        for s in (11, 22, 33):
            _write_run(root, f"bertL-mnliB-{size}-s{s}", lr=at23(size), ccs=ccs23(size))
    _write_run(root, "bertL-mnliB-8k-s11", is_key=True, lr={**table_curve(R.T6), **at23("8k")}, ccs=ccs23("8k"))
    for s in (22, 33):
        _write_run(root, f"bertL-mnliB-8k-s{s}", lr=at23("8k"), ccs=ccs23("8k"))
    _write_run(root, "bertL-mnliB-16k-s11", degenerate=True, is_key=True, devm=0.52)
    _write_run(root, "bertL-mnliB-16k-s111", is_key=True, lr={**table_curve(R.T7), **at23("16k")}, ccs=ccs23("16k"),
               NE=0.5 if q4_fail else 0.03)
    for s in (22, 33, 44, 55):
        _write_run(root, f"bertL-mnliB-16k-s{s}", lr=at23("16k"), ccs=ccs23("16k"), NE=0.5 if q4_fail else 0.03)
    for s in (11, 22, 33):
        _write_run(root, f"bertL-mix-16k-s{s}", lr=at23("mix"), ccs=ccs23("mix"), total=0.97, NE=0.95)


def test_report_g2_pass_when_results_match_vdl(tmp_path):
    from khoaluan.analysis.p2_report import build_report
    _vdl_like(str(tmp_path / "in"), shift=0)
    text, res, primary = build_report(str(tmp_path / "in"), str(tmp_path / "out"))
    assert primary == 23 and res[23]["G2"] == "ĐẠT"
    assert res[23]["Q3"]["pass"] and res[23]["anchors_pass"] and res[23]["seed_majority"]["16k"] == "5/5"
    assert "thoái hoá: 1" in text and os.path.exists(str(tmp_path / "out" / "P2_REPORT.md"))


def test_report_dr2_detects_shift_and_q4_failure(tmp_path):
    from khoaluan.analysis.p2_report import build_report
    _vdl_like(str(tmp_path / "in1"), shift=1)
    _, res, primary = build_report(str(tmp_path / "in1"), str(tmp_path / "o1"))
    assert primary == 24 and res[24]["G2"] == "ĐẠT"          # D-R2 chọn đúng quy ước, G2 đánh giá ở tầng đó
    _vdl_like(str(tmp_path / "in2"), shift=0, q4_fail=True)
    _, res, _ = build_report(str(tmp_path / "in2"), str(tmp_path / "o2"))
    assert res[23]["G2"].startswith("KHÔNG ĐẠT")
