"""
load.py — đọc đầu ra của prepare_data.py v1.2 cho P2 (fine-tune, inference, contrast pair).
Tương thích: SPEC_DULIEU_KHAINIEM v1.2 §1–§2, SPEC_REPRODUCE_VDL v1.1 §3–§5.

Nguyên tắc:
- Mọi ví dụ định danh bằng `uid` (KHÔNG join theo pairID).
- Trước khi train, kiểm dữ liệu đúng là bản của lần chạy A-06 chính thức: vân tay FT-16K và HT-A
  tính lại từ file phải khớp manifest.json VÀ khớp hằng số OFFICIAL_A06 dưới đây.
  Lệch -> dừng (không train trên dữ liệu khác bản đã ghi vào sổ dự án).
"""
import json
import os

import numpy as np
import pandas as pd

# Lần chạy A-06 chính thức (A06_summary.txt nhóm gửi 09/11/2026, prepare_data v1.2).
OFFICIAL_A06 = {
    "script_version": "1.2",
    "data_seed": 1234,
    "fingerprint_ft16k": "2e48372834c0e490",
    "fingerprint_hans_A": "08cd3fff63776257",
    "hf_revision_resolved": "da70db2af9d09693783c3320c4249840212ee221",
}

FT_SIZES = (25, 100, 1000, 8000, 16000)
EVAL_SETS = ("dev_m", "dev_mm", "hans_eval", "hans_train_B")
STD_COLS = ["uid", "premise", "hypothesis", "y"]


def fingerprint(values):
    """Giống hệt prepare_data.fingerprint (sha256 của chuỗi giá trị, ngăn bởi \\x00, lấy 16 ký tự đầu)."""
    import hashlib
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def find_data_dir(root):
    """Tìm thư mục chứa manifest.json của prepare_data (Kaggle mount input ở độ sâu khác nhau)."""
    if os.path.exists(os.path.join(root, "manifest.json")):
        return root
    hits = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "manifest.json" in filenames and "ft_uids_16000.txt" in filenames:
            hits.append(dirpath)
    if len(hits) != 1:
        raise FileNotFoundError(f"cần đúng 1 thư mục dữ liệu dưới {root}, tìm thấy {len(hits)}: {hits}")
    return hits[0]


def load_manifest(data_dir):
    with open(os.path.join(data_dir, "manifest.json")) as f:
        return json.load(f)


def read_uids(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def verify_data(data_dir, expected=OFFICIAL_A06):
    """Kiểm dữ liệu khớp lần chạy A-06 chính thức. Trả về dict kết quả; lỗi -> ValueError."""
    man = load_manifest(data_dir)
    rev = (man.get("mnli_source") or {}).get("hf_revision_resolved")
    got = {
        "script_version": man.get("script_version"),
        "data_seed": man.get("data_seed"),
        "fingerprint_ft16k_manifest": man.get("fingerprint_ft16k"),
        "fingerprint_ft16k_file": fingerprint(read_uids(os.path.join(data_dir, "ft_uids_16000.txt"))),
        "fingerprint_hans_A_manifest": man.get("fingerprint_hans_A"),
        "fingerprint_hans_A_file": fingerprint(
            pd.read_json(os.path.join(data_dir, "hans_train_A.jsonl"), lines=True)["uid"]),
        "hf_revision_resolved": rev,
    }
    problems = []
    if got["script_version"] != expected["script_version"]:
        problems.append(f"script_version {got['script_version']} != {expected['script_version']}")
    if got["data_seed"] != expected["data_seed"]:
        problems.append(f"data_seed {got['data_seed']} != {expected['data_seed']}")
    for k in ("ft16k", "hans_A"):
        e = expected[f"fingerprint_{k}"]
        if not (got[f"fingerprint_{k}_manifest"] == got[f"fingerprint_{k}_file"] == e):
            problems.append(f"fingerprint_{k}: manifest={got[f'fingerprint_{k}_manifest']} "
                            f"file={got[f'fingerprint_{k}_file']} expected={e}")
    if rev != expected["hf_revision_resolved"]:
        problems.append(f"hf_revision {rev} != {expected['hf_revision_resolved']}")
    if problems:
        raise ValueError("[DỪNG] dữ liệu không khớp lần chạy A-06 chính thức: " + "; ".join(problems))
    got["ok"] = True
    return got


# ----------------------------------------------------------------- bảng chuẩn hoá
def _mnli_std(df):
    return df[["uid", "premise", "hypothesis", "y"]].copy()


def _hans_std(df):
    out = df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis"})
    return out[["uid", "premise", "hypothesis", "y", "heuristic", "subcase", "template"]].copy()


def _check_frame(df, name):
    if not df.uid.is_unique:
        raise ValueError(f"[DỪNG] uid trùng trong {name}")
    if not set(df.y.unique()) <= {0, 1}:
        raise ValueError(f"[DỪNG] nhãn ngoài {{0,1}} trong {name}")
    if df[["premise", "hypothesis"]].isna().any().any():
        raise ValueError(f"[DỪNG] premise/hypothesis rỗng trong {name}")
    return df


def train_frame(data_dir, data_name, size):
    """Tập fine-tune theo SPEC_REPRODUCE §3.1. data_name: 'mnliB' (FT-size) | 'mix' (FT-8K ∪ HT-A, size=16000).
    Thứ tự dòng = thứ tự trong file uid (Trainer tự xáo theo seed huấn luyện)."""
    tr = pd.read_parquet(os.path.join(data_dir, "mnli_bin_train_all.parquet"),
                         columns=["uid", "premise", "hypothesis", "y", "ft_rank", "in_pool"])
    if data_name == "mnliB":
        if size not in FT_SIZES:
            raise ValueError(f"size {size} không thuộc {FT_SIZES}")
        uids = read_uids(os.path.join(data_dir, f"ft_uids_{size}.txt"))
        by_rank = tr.loc[tr.ft_rank < size].sort_values("ft_rank").uid.tolist()
        if uids != by_rank:
            raise ValueError(f"[DỪNG] ft_uids_{size}.txt lệch với ft_rank trong parquet")
        if tr.set_index("uid").loc[uids, "in_pool"].any():
            raise ValueError("[DỪNG] tập fine-tune chứa ví dụ của MNLI-POOL")
        df = tr.set_index("uid").loc[uids].reset_index()
        return _check_frame(_mnli_std(df), f"FT-{size}")
    if data_name == "mix":
        if size != 16000:
            raise ValueError("M-MIX chỉ có size 16000 (FT-8K ∪ HT-A)")
        uids = read_uids(os.path.join(data_dir, "mix_uids.txt"))
        ha = pd.read_json(os.path.join(data_dir, "hans_train_A.jsonl"), lines=True)
        m = tr.set_index("uid")
        mn = [u for u in uids if u.startswith("mnli_train:")]
        hs = [u for u in uids if u.startswith("hans_train:")]
        if len(mn) != 8000 or len(hs) != len(ha) or len(mn) + len(hs) != len(uids):
            raise ValueError(f"[DỪNG] mix_uids.txt sai cấu trúc: mnli={len(mn)} hans={len(hs)} tổng={len(uids)}")
        if mn != read_uids(os.path.join(data_dir, "ft_uids_8000.txt")):
            raise ValueError("[DỪNG] phần MNLI của MIX khác FT-8K")
        part_m = _mnli_std(m.loc[mn].reset_index())
        part_h = _hans_std(ha.set_index("uid").loc[hs].reset_index())[STD_COLS]
        df = pd.concat([part_m, part_h], ignore_index=True)
        return _check_frame(df, "MIX")
    raise ValueError(f"data_name không hợp lệ: {data_name}")


def eval_frame(data_dir, name):
    """Tập đánh giá / trích representation. Cột: uid, premise, hypothesis, y (+ heuristic, subcase, template với HANS)."""
    if name == "dev_m":
        df = _mnli_std(pd.read_parquet(os.path.join(data_dir, "mnli_bin_dev_m.parquet")))
    elif name == "dev_mm":
        df = _mnli_std(pd.read_parquet(os.path.join(data_dir, "mnli_bin_dev_mm.parquet")))
    elif name == "hans_eval":
        df = _hans_std(pd.read_json(os.path.join(data_dir, "hans_eval.jsonl"), lines=True))
    elif name == "hans_train_B":
        df = _hans_std(pd.read_json(os.path.join(data_dir, "hans_train_B.jsonl"), lines=True))
    else:
        raise ValueError(f"tập không hợp lệ: {name}")
    return _check_frame(df.reset_index(drop=True), name)


def hans_contrast_subset(hans_eval, n_total=10000, seed=1234):
    """SPEC_REPRODUCE §5.3: 10,000 ví dụ HANS-EVAL phân tầng theo subcase (333–334/subcase), cố định qua mọi mô hình.
    Tất định: sắp theo uid trong mỗi subcase rồi rút bằng default_rng(seed)."""
    subcases = sorted(hans_eval.subcase.unique())
    base, extra = divmod(n_total, len(subcases))
    rng = np.random.default_rng(seed)
    extra_sc = set(rng.choice(subcases, size=extra, replace=False)) if extra else set()
    picks = []
    for sc in subcases:
        pool = sorted(hans_eval.loc[hans_eval.subcase == sc, "uid"])
        k = base + (1 if sc in extra_sc else 0)
        picks += list(rng.choice(pool, size=k, replace=False))
    picks = sorted(picks)
    return hans_eval.set_index("uid").loc[picks].reset_index()
