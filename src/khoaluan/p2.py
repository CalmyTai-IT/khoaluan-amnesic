"""
p2.py — chạy MỘT mô hình của P2 (S2 + S3) theo SPEC_REPRODUCE_VDL v1.1.

    CUDA_VISIBLE_DEVICES=0 python scripts/run_p2.py --data-root /kaggle/input --out-root /kaggle/working/out \
        --data mnliB --size 16000 --seed 11 --account TV-B

Stage (--stages, mặc định train,infer,contrast,lr,ccs,dr1; M-PT tự bỏ train):
  train    : fine-tune 5 epoch cố định (base.yaml); lưu checkpoint cho M-16K / M-MIX (D-15, SPEC §2.2);
             DỪNG SỚM nếu run thoái hoá (finetune.py: n ≥ 8K, loss nửa sau epoch 1 ≥ 0.65)
  infer    : DEV-M, DEV-MM, HANS-EVAL (+ HT-B cho mô hình giữ checkpoint): logits theo uid, chỉ số, Q4;
             cache pooler_output (= L_out) + [CLS] tầng 20–24 (fp16) cho S4–S5 (SPEC §5.4)
  contrast : contrast pair DEV-M (toàn bộ) + HANS-EVAL 10K phân tầng: [CLS] mọi tầng (mô hình then chốt) hoặc 20–24;
             token cuối + mean ở tầng 23–24 (mô hình then chốt, cho D-R4). Tính và lưu fp32.
  lr       : LR faithful + robust, 5 lần, mọi tầng đã cache; chẩn đoán lưu fp16 — SPEC §6.1
  ccs      : CCS tuyến tính (DLK) mọi tầng đã cache; thêm MLP ở tầng 23–24 — SPEC §6.2
  dr1      : D-R1 (LR-HANS khi test là subcase chưa thấy, so với chia ngẫu nhiên cùng cỡ) tầng 20–24 — SPEC §8
Run thoái hoá (dừng sớm, hoặc DEV-M < 0.60 với n ≥ 1K): ghi bản ghi, BỎ các stage sau; hàng đợi chạy lại seed+100.
Đầu ra: <out-root>/<run_id>/ ; p2_summary.txt; p2_metrics.json; run_record.json cho registry.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # phải đặt trước khi khởi tạo CUDA
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from khoaluan.data import load as L  # noqa: E402
from khoaluan.eval import metrics as M  # noqa: E402
from khoaluan.utils.registry import now_utc, sha256_file, write_run_record  # noqa: E402

ALL_STAGES = ["train", "infer", "contrast", "lr", "ccs", "dr1"]
SIZE_TAG = {25: "25", 100: "100", 1000: "1k", 8000: "8k", 16000: "16k"}
# Mốc LR tầng 23 (bảng phụ lục V&dL, Bảng 2–3; M-PT theo văn bản §4.3.1) — SPEC_REPRODUCE §7.1
REF_LR23 = {"pt": (0.59, 0.61), "25": (0.59, 0.67), "100": (0.58, 0.68), "1k": (0.71, 0.70),
            "8k": (0.73, 0.58), "16k": (0.86, 0.72), "mix": (0.83, 0.83)}
T0 = time.time()


def is_key_model(data, size, seed):
    """Mô hình then chốt (SPEC §5.4): M-PT và seed 11 của M-8K, M-16K, M-MIX — KỂ CẢ bản chạy lại seed+100
    (111, 211, …) khi seed 11 thoái hoá (SPEC §3.3). Mô hình then chốt: contrast mọi tầng + token cuối/mean."""
    if data == "pt":
        return True
    return (data == "mix" or size in (8000, 16000)) and seed % 100 == 11


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} phút] {msg}", flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="thư mục chứa (ở đâu đó bên dưới) manifest.json của prepare_data")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--data", choices=["mnliB", "mix", "pt"], required=True)
    ap.add_argument("--size", type=int, default=0, help="25|100|1000|8000|16000 (mix: 16000; pt: bỏ qua)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "..", "configs", "finetune",
                                                      "base.yaml"))
    ap.add_argument("--set", action="append", default=[], help="ghi đè cấu hình key=value (CHỈ dùng cho test)")
    ap.add_argument("--account", default=None, help="TV-A | TV-B")
    ap.add_argument("--run-suffix", default="", help="ví dụ -rep2 cho chạy lặp kiểm S0")
    ap.add_argument("--model-dir", default=None, help="checkpoint đã fine-tune (khi không có stage train)")
    ap.add_argument("--cache-dir", default=None, help="đọc cache contrast từ thư mục khác (lr/ccs/dr1 chạy riêng)")
    ap.add_argument("--save-model", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--std-reps", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--std-layers", default="20-24")
    ap.add_argument("--cp-layers", choices=["auto", "all", "20-24"], default="auto")
    ap.add_argument("--aux-layers", default="23-24", help="tầng cho token cuối/mean (mô hình then chốt)")
    ap.add_argument("--cp-dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--hans-cp-n", type=int, default=10000)
    ap.add_argument("--lr-layers", default="all", help="all | danh sách, ví dụ 20-24 hoặc 23,24")
    ap.add_argument("--dr1-layers", default="20-24")
    ap.add_argument("--infer-batch", type=int, default=64)
    ap.add_argument("--cp-batch", type=int, default=64)
    ap.add_argument("--allow-unofficial-data", action="store_true",
                    help="bỏ kiểm vân tay A-06 chính thức (CHỈ cho unit test)")
    return ap.parse_args(argv)


def parse_layers(spec, n_layers):
    if spec == "all":
        return list(range(n_layers + 1))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [l for l in out if 0 <= l <= n_layers]


def load_cfg(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def cfg_sha(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def run_id_of(a):
    if a.data == "pt":
        return "bertL-pt" + a.run_suffix
    tag = "mix-16k" if a.data == "mix" else f"mnliB-{SIZE_TAG[a.size]}"
    return f"bertL-{tag}-s{a.seed}{a.run_suffix}"


def git_commit():
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(__file__), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def save_npy(path, arr, dtype):
    np.save(path, np.asarray(arr).astype(dtype, copy=False))


def write_uids(path, uids):
    with open(path, "w") as f:
        f.write("\n".join(uids))


def load_cp(cdir, s):
    """Đọc cache contrast của tập s: (meta, h_yes, h_no, y, splits) — splits giống hệt nhau cho LR và CCS."""
    from khoaluan.probe import lr as P
    meta = pd.read_parquet(os.path.join(cdir, f"cp_{s}_meta.parquet"))
    hy = np.load(os.path.join(cdir, f"cp_{s}_cls_yes.npy"), mmap_mode="r")
    hn = np.load(os.path.join(cdir, f"cp_{s}_cls_no.npy"), mmap_mode="r")
    y = meta.y.to_numpy()
    splits = P.splits_mnli(y) if s == "dev_m" else P.splits_hans(meta.heuristic + "|" + meta.y.astype(str))
    return meta, hy, hn, y, splits


def layer_report(summ, variant, layers):
    """Bảng gọn ở tầng 23/24 cho p2_summary."""
    out = {}
    for l in [x for x in (23, 24) if x in layers]:
        f = summ[(summ.layer == l) & (summ.variant == variant)].set_index("dataset")
        if {"dev_m", "hans_eval"} <= set(f.index):
            out[f"L{l}"] = {"MNLI": round(float(f.loc["dev_m", "acc_mean"]), 4),
                            "HANS": round(float(f.loc["hans_eval", "acc_mean"]), 4),
                            "SD_MNLI_HANS": [round(float(f.loc["dev_m", "acc_sd"]), 4),
                                             round(float(f.loc["hans_eval", "acc_sd"]), 4)]}
    curve = summ[summ.variant == variant].pivot(index="layer", columns="dataset", values="acc_mean")
    out["curve"] = {int(l): [round(float(v), 3) for v in row] for l, row in curve.iterrows()}
    out["curve_columns"] = list(curve.columns)
    return out


def main(argv=None):
    a = parse_args(argv)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        sys.exit(f"[DỪNG] stage không hợp lệ: {unknown}")
    if a.data == "mix":
        a.size = 16000
    if a.data == "pt" and "train" in stages:
        stages = [s for s in stages if s != "train"]          # M-PT không fine-tune
    if a.data != "pt" and (a.size not in SIZE_TAG or a.seed <= 0):
        sys.exit("[DỪNG] cần --size thuộc {25,100,1000,8000,16000} và --seed > 0")
    import torch
    import transformers
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        sys.exit("[DỪNG] thấy >1 GPU. Đặt CUDA_VISIBLE_DEVICES=0 (hoặc 1) cho mỗi tiến trình.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    if not transformers.__version__.startswith("5."):
        log(f"CẢNH BÁO: transformers {transformers.__version__} (D-20 yêu cầu 5.x như image Kaggle)")

    run_id = run_id_of(a)
    out = os.path.join(a.out_root, run_id)
    os.makedirs(out, exist_ok=True)
    cfg = load_cfg(a.config, a.set)
    is_key = is_key_model(a.data, a.size, a.seed)
    keep_ckpt = a.data == "mix" or (a.data == "mnliB" and a.size == 16000)
    save_model = {"auto": keep_ckpt, "yes": True, "no": False}[a.save_model]
    std_reps = {"auto": keep_ckpt or a.data == "pt", "yes": True, "no": False}[a.std_reps]
    fp16 = bool(cfg.get("fp16")) and device.type == "cuda"
    rec = {"run_id": run_id, "stages": stages, "is_key_model": is_key, "device": str(device), "gpu_name": gpu,
           "torch": torch.__version__, "transformers": transformers.__version__, "start_utc": now_utc(),
           "git_commit": git_commit(), "config_sha256": cfg_sha(cfg), "config_file_sha256": sha256_file(a.config),
           "config_overrides": a.set, "fp16_inference": fp16, "degenerate": False, "degenerate_reason": None,
           "skipped_stages": []}
    log(f"run_id={run_id} | thiết bị={device} {gpu or ''} | torch {torch.__version__} | "
        f"transformers {transformers.__version__} | stages={stages} | then chốt={is_key}")

    data_dir = L.find_data_dir(a.data_root)
    rec["data_dir"] = data_dir
    rec["data_manifest_sha256"] = sha256_file(os.path.join(data_dir, "manifest.json"))
    if a.allow_unofficial_data:
        rec["data_check"] = "BỎ QUA (unit test)"
        log("CẢNH BÁO: bỏ kiểm vân tay dữ liệu — kết quả KHÔNG được vào registry")
    else:
        rec["data_check"] = L.verify_data(data_dir)
        log(f"dữ liệu khớp lần chạy A-06 chính thức: {data_dir}")

    model = tok = None
    n_layers = None
    metrics = {}

    def degenerate(reason):
        rec["degenerate"], rec["degenerate_reason"] = True, reason
        later = [s for s in stages if ALL_STAGES.index(s) > ALL_STAGES.index(current)]
        rec["skipped_stages"] = later
        log(f"[THOÁI HOÁ] {reason} -> bỏ các stage {later}; giữ bản ghi (SPEC §3.3), hàng đợi sẽ chạy seed+100")

    # ------------------------------------------------------------ train
    current = "train"
    if "train" in stages:
        from khoaluan.train.finetune import finetune
        frame = L.train_frame(data_dir, a.data, a.size)
        log(f"fine-tune {run_id}: {len(frame)} ví dụ")
        model, tok, st = finetune(frame, cfg, a.seed, out, model_dir=os.path.join(out, "model") if save_model else None,
                                  log=log)
        model.to(device)
        import gc
        gc.collect()                      # giải phóng optimizer state (~2.7 GB) trước khi trích contrast fp32
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rec["train"] = {k: v for k, v in st.items() if k != "train_loss_curve"}
        with open(os.path.join(out, "train_log.json"), "w") as f:
            json.dump({"loss_every_logging_steps": st["train_loss_curve"],
                       "logging_steps": cfg.get("logging_steps")}, f)
        log(f"fine-tune xong: {st['wall_minutes_train']} phút, loss TB {st['train_loss_mean']:.4f}, "
            f"dừng sớm={st['aborted']}")
        if st["aborted"]:
            degenerate(f"loss TB nửa sau epoch 1 = {st['abort_rule']['epoch1_second_half_mean_loss']:.4f} "
                       f"≥ {st['abort_rule']['threshold']}")
    elif {"infer", "contrast"} & set(stages):
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        if a.data == "pt":
            tok = AutoTokenizer.from_pretrained(cfg["model_name"])
            model = AutoModel.from_pretrained(cfg["model_name"])
        else:
            if not a.model_dir:
                sys.exit("[DỪNG] không có stage train -> cần --model-dir của checkpoint đã fine-tune")
            tok = AutoTokenizer.from_pretrained(a.model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(a.model_dir)
            rec["model_dir_loaded"] = a.model_dir
        model.to(device).eval()
    if model is not None:
        n_layers = model.config.num_hidden_layers

    # ------------------------------------------------------------ infer
    current = "infer"
    if "infer" in stages and not rec["degenerate"]:
        from khoaluan.eval.forward import run_pairs
        sets = ["dev_m", "dev_mm", "hans_eval"] + (["hans_train_B"] if std_reps else [])
        std_layers = parse_layers(a.std_layers, n_layers) if std_reps else []
        preds = {}
        t = time.time()
        for s in sets:
            fr = L.eval_frame(data_dir, s)
            r = run_pairs(model, tok, fr, cfg["max_length"], a.infer_batch, fp16, cls_layers=std_layers)
            if r["logits"] is not None:
                preds[s] = M.preds_frame(fr, r["logits"])
                preds[s].to_parquet(os.path.join(out, f"preds_{s}.parquet"))
            if std_reps:
                save_npy(os.path.join(out, f"std_{s}_pooler.npy"), r["pooler"], np.float16)
                save_npy(os.path.join(out, f"std_{s}_cls.npy"), r["cls"], np.float16)
                write_uids(os.path.join(out, f"std_{s}_uids.txt"), fr.uid.tolist())
            log(f"inference {s}: n={len(fr)}" + (f" acc={M.accuracy(preds[s]):.4f}" if s in preds else ""))
        if std_reps:
            with open(os.path.join(out, "std_layers.json"), "w") as f:
                json.dump({"cls_layers": std_layers, "pooler": "pooler_output (L_out)", "dtype": "float16"}, f)
        rec["wall_minutes_infer"] = round((time.time() - t) / 60, 2)
        if preds:
            metrics = M.summarize_sets(preds)
            M.subcase_table(preds).to_csv(os.path.join(out, "hans_subcase.csv"), index=False)
            kind = "mix" if a.data == "mix" else "mnli"
            q4 = M.q4_check(metrics, kind)
            metrics["Q4"] = {k: [None if v[0] is None else round(float(v[0]), 4), bool(v[1])] for k, v in q4.items()}
            log(f"Q4 ({kind}, ngưỡng tiền đăng ký): {metrics['Q4']}")
            if a.size < 16000 and a.data == "mnliB":
                log("(Q4 chỉ áp dụng cho M-16K / M-MIX; với kích thước khác chỉ để tham khảo)")
            if M.is_degenerate(a.size, metrics["devm_acc"]):
                degenerate(f"DEV-M acc = {metrics['devm_acc']:.4f} < 0.60")
    metrics["degenerate"] = rec["degenerate"]

    # ------------------------------------------------------------ contrast
    current = "contrast"
    if "contrast" in stages and not rec["degenerate"]:
        from khoaluan.eval.forward import run_contrast
        cp_spec = a.cp_layers if a.cp_layers != "auto" else ("all" if is_key else "20-24")
        cls_layers = parse_layers(cp_spec, n_layers)
        aux_layers = parse_layers(a.aux_layers, n_layers) if is_key else []
        if not cls_layers:
            sys.exit(f"[DỪNG] không có tầng nào để trích (cp-layers={cp_spec}, model có {n_layers} tầng)")
        # Trích contrast ở fp32 trừ khi --set fp16_contrast=true: hiệu h(Yes) − h(No) nhạy với sai số số học.
        cp_fp16 = bool(cfg.get("fp16_contrast", False)) and device.type == "cuda"
        cp_meta = {"cls_layers": cls_layers, "aux_layers": aux_layers, "dtype": a.cp_dtype,
                   "convention": "x_LR = h(Yes) - h(No) (DLK: neg = answer_choices[0] = 'Yes')",
                   "compute": "fp16 autocast" if cp_fp16 else "fp32"}
        t = time.time()
        frames = {"dev_m": L.eval_frame(data_dir, "dev_m"),
                  "hans_eval": L.hans_contrast_subset(L.eval_frame(data_dir, "hans_eval"), a.hans_cp_n, seed=1234)}
        cp_meta["hans_subset_fingerprint"] = L.fingerprint(frames["hans_eval"].uid)
        cp_stats = {}
        for s, fr in frames.items():
            n, hdim = len(fr), model.config.hidden_size
            arrs = {}
            for ans in ("yes", "no"):
                arrs[f"cls_{ans}"] = np.lib.format.open_memmap(os.path.join(out, f"cp_{s}_cls_{ans}.npy"), "w+",
                                                               a.cp_dtype, (n, len(cls_layers), hdim))
                if aux_layers:
                    for kind in ("last", "mean"):
                        arrs[f"{kind}_{ans}"] = np.lib.format.open_memmap(
                            os.path.join(out, f"cp_{s}_{kind}_{ans}.npy"), "w+", a.cp_dtype,
                            (n, len(aux_layers), hdim))
            st = run_contrast(model, tok, fr, a.cp_batch, cp_fp16, cls_layers, aux_layers, arrs)
            for v in arrs.values():
                v.flush()
            del arrs
            keep = [c for c in ["uid", "y", "heuristic", "subcase", "template"] if c in fr.columns]
            fr[keep].to_parquet(os.path.join(out, f"cp_{s}_meta.parquet"))
            cp_stats[s] = st
            if st["n_truncated"] or st["n_identical_pairs"]:
                sys.exit(f"[DỪNG] contrast {s}: {st} — có ví dụ bị cắt cụt hoặc cặp Yes/No trùng token")
            log(f"contrast {s}: n={n}, token tối đa {st['max_tokens']}")
        cp_meta["stats"] = cp_stats
        with open(os.path.join(out, "cp_layers.json"), "w") as f:
            json.dump(cp_meta, f, indent=2)
        rec["wall_minutes_contrast"] = round((time.time() - t) / 60, 2)

    # ------------------------------------------------------------ lr / ccs / dr1 (CPU/GPU nhỏ, từ cache)
    cdir = a.cache_dir or out
    ref_key = "pt" if a.data == "pt" else ("mix" if a.data == "mix" else SIZE_TAG[a.size])
    lr_report, ccs_report, dr1_report = {}, {}, {}
    probe_stages = [s for s in ("lr", "ccs", "dr1") if s in stages]
    if probe_stages and not rec["degenerate"]:
        with open(os.path.join(cdir, "cp_layers.json")) as f:
            cpm = json.load(f)
        layer_ids = cpm["cls_layers"]

    current = "lr"
    if "lr" in stages and not rec["degenerate"]:
        from khoaluan.probe import lr as P
        layers = layer_ids if a.lr_layers == "all" else [l for l in parse_layers(a.lr_layers, max(layer_ids))
                                                         if l in layer_ids]
        t = time.time()
        res = []
        for s in ("dev_m", "hans_eval"):
            meta, hy, hn, y, splits = load_cp(cdir, s)
            res.append(P.run_lr(hy, hn, y, splits, layers, layer_ids, dataset=s, log=log))
            if s == "dev_m" and cpm["dtype"] == "float32":
                P.fp16_storage_diagnostic(hy, hn, layer_ids).to_csv(os.path.join(out, "cp_fp16_diag.csv"), index=False)
        raw = pd.concat(res, ignore_index=True)
        raw.insert(0, "run_id", run_id)
        raw.to_csv(os.path.join(out, "lr_raw.csv"), index=False)
        summ = P.summarize(raw)
        summ.insert(0, "run_id", run_id)
        summ.to_csv(os.path.join(out, "lr_summary.csv"), index=False)
        if cpm.get("aux_layers"):                       # D-R4: token cuối / mean pooling (so Bảng 1 của V&dL)
            aux = []
            for s in ("dev_m", "hans_eval"):
                _meta, _hy, _hn, y, splits = load_cp(cdir, s)
                for kind in ("last", "mean"):
                    ay = np.load(os.path.join(cdir, f"cp_{s}_{kind}_yes.npy"), mmap_mode="r")
                    an = np.load(os.path.join(cdir, f"cp_{s}_{kind}_no.npy"), mmap_mode="r")
                    d = P.run_lr(ay, an, y, splits, cpm["aux_layers"], cpm["aux_layers"], variants=("faithful",),
                                 dataset=s)
                    d["token"] = kind
                    aux.append(d)
            aux = pd.concat(aux, ignore_index=True)
            aux.insert(0, "run_id", run_id)
            (aux.groupby(["run_id", "token", "dataset", "layer", "variant"])
                .agg(acc_mean=("acc", "mean"), acc_sd=("acc", "std"), n_runs=("acc", "size")).reset_index()
                .to_csv(os.path.join(out, "lr_aux_summary.csv"), index=False))
        rec["wall_minutes_lr"] = round((time.time() - t) / 60, 2)
        lr_report = {"faithful": layer_report(summ, "faithful", layers),
                     "robust": {k: v for k, v in layer_report(summ, "robust", layers).items() if k.startswith("L")},
                     "conv_warn_faithful_total": int(summ[summ.variant == "faithful"].conv_warn.sum()),
                     "ref_VdL_L23_MNLI_HANS": REF_LR23[ref_key]}

    current = "ccs"
    if "ccs" in stages and not rec["degenerate"]:
        from khoaluan.probe import ccs as C
        t = time.time()
        res = []
        mlp_layers = [l for l in (23, 24) if l in layer_ids]
        for s in ("dev_m", "hans_eval"):
            meta, hy, hn, y, splits = load_cp(cdir, s)
            res.append(C.run_ccs(hy, hn, y, splits, layer_ids, layer_ids, mlp_layers=mlp_layers, dataset=s, log=log))
        if cpm.get("aux_layers"):                       # D-R4: CCS tuyến tính trên token cuối / mean pooling
            aux = []
            for s in ("dev_m", "hans_eval"):
                _meta, _hy, _hn, y, splits = load_cp(cdir, s)
                for kind in ("last", "mean"):
                    ay = np.load(os.path.join(cdir, f"cp_{s}_{kind}_yes.npy"), mmap_mode="r")
                    an = np.load(os.path.join(cdir, f"cp_{s}_{kind}_no.npy"), mmap_mode="r")
                    d = C.run_ccs(ay, an, y, splits, cpm["aux_layers"], cpm["aux_layers"], dataset=s)
                    d["token"] = kind
                    aux.append(d)
            aux = pd.concat(aux, ignore_index=True)
            aux.insert(0, "run_id", run_id)
            (aux.groupby(["run_id", "token", "dataset", "layer", "variant"])
                .agg(acc_mean=("acc", "mean"), acc_sd=("acc", "std"), n_runs=("acc", "size")).reset_index()
                .to_csv(os.path.join(out, "ccs_aux_summary.csv"), index=False))
        raw = pd.concat(res, ignore_index=True)
        raw.insert(0, "run_id", run_id)
        raw.to_csv(os.path.join(out, "ccs_raw.csv"), index=False)
        summ = (raw.groupby(["run_id", "dataset", "layer", "variant"])
                   .agg(acc_mean=("acc", "mean"), acc_sd=("acc", "std"), n_runs=("acc", "size")).reset_index())
        summ.to_csv(os.path.join(out, "ccs_summary.csv"), index=False)
        rec["wall_minutes_ccs"] = round((time.time() - t) / 60, 2)
        ccs_report = {"linear": layer_report(summ, "linear", layer_ids),
                      "mlp": {k: v for k, v in layer_report(summ, "mlp", layer_ids).items() if k.startswith("L")},
                      "chance_note": "acc = max(acc, 1-acc): mức chance ≈ 0.513, phân vị 95% ≈ 0.531 với 1,000 ví dụ test"}

    current = "dr1"
    if "dr1" in stages and not rec["degenerate"]:
        from khoaluan.probe import lr as P
        layers = [l for l in parse_layers(a.dr1_layers, max(layer_ids)) if l in layer_ids]
        if layers:
            t = time.time()
            meta, hy, hn, _y, _ = load_cp(cdir, "hans_eval")
            raw = P.run_dr1(hy, hn, meta, layers, layer_ids, robust_layers=[l for l in (23, 24) if l in layers],
                            log=log)
            raw.insert(0, "run_id", run_id)
            raw.to_csv(os.path.join(out, "dr1_raw.csv"), index=False)
            summ = raw.groupby(["run_id", "layer", "split", "variant"]).acc.agg(["mean", "std"]).reset_index()
            summ.to_csv(os.path.join(out, "dr1_summary.csv"), index=False)
            rec["wall_minutes_dr1"] = round((time.time() - t) / 60, 2)
            f = summ[summ.variant == "faithful"]
            dr1_report = {f"L{int(l)}": {r["split"]: round(float(r["mean"]), 4) for _, r in g.iterrows()}
                          for l, g in f.groupby("layer")}

    # ------------------------------------------------------------ ghi kết quả
    rec["wall_minutes_total"] = round((time.time() - T0) / 60, 2)
    rec["metrics"] = metrics
    rec["lr"], rec["ccs"], rec["dr1"] = lr_report, ccs_report, dr1_report
    with open(os.path.join(out, "p2_metrics.json"), "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False, default=str)
    if {"train", "infer"} & set(stages):
        tr = rec.get("train", {})
        write_run_record(
            out, run_id=run_id, kind="p2:" + "+".join(stages), git_commit=rec["git_commit"],
            config_sha256=rec["config_sha256"], data_manifest_sha256=rec["data_manifest_sha256"],
            seed=a.seed if a.data != "pt" else None, data_seed=1234, kaggle_account=a.account,
            kaggle_notebook_version=os.environ.get("KAGGLE_NOTEBOOK_VERSION"), gpu_name=gpu,
            torch=torch.__version__, transformers=transformers.__version__, start_utc=rec["start_utc"],
            wall_minutes=rec["wall_minutes_total"], train_loss_final=tr.get("train_loss_last_logged"),
            devm_acc=metrics.get("devm_acc"), devmm_acc=metrics.get("devmm_acc"),
            hans_E_LO=metrics.get("hans_E_LO"), hans_E_SUB=metrics.get("hans_E_SUB"),
            hans_E_CON=metrics.get("hans_E_CON"), hans_NE_LO=metrics.get("hans_NE_LO"),
            hans_NE_SUB=metrics.get("hans_NE_SUB"), hans_NE_CON=metrics.get("hans_NE_CON"),
            truncation_rate=tr.get("truncation_rate"), degenerate=rec["degenerate"],
            notes=("UNOFFICIAL-DATA (test) " if a.allow_unofficial_data else "") + f"stages={','.join(stages)}"
                  + (f"; THOÁI HOÁ: {rec['degenerate_reason']}" if rec["degenerate"] else ""),
            train_loss_mean=tr.get("train_loss_mean"), steps=tr.get("steps"), optim=tr.get("optim"),
            fp16_train=tr.get("fp16"), aborted=tr.get("aborted"), Q4=metrics.get("Q4"),
            hans_acc_total=metrics.get("hans_acc_total"),
        )
    write_summary(rec, os.path.join(out, "p2_summary.txt"))
    log(f"XONG {run_id}. Tóm tắt: {os.path.join(out, 'p2_summary.txt')}")
    return rec


def write_summary(rec, path):
    m = rec.get("metrics", {})
    blocks = {
        "run": {k: rec.get(k) for k in ["run_id", "stages", "is_key_model", "git_commit", "gpu_name", "torch",
                                         "transformers", "fp16_inference", "data_manifest_sha256"]},
        "data_check_ok": (rec.get("data_check") or {}).get("ok") if isinstance(rec.get("data_check"), dict)
        else rec.get("data_check"),
        "degenerate": {"degenerate": rec.get("degenerate"), "reason": rec.get("degenerate_reason"),
                       "skipped_stages": rec.get("skipped_stages")},
        "train": rec.get("train"),
        "wall_minutes": {k: rec.get(k) for k in ["wall_minutes_infer", "wall_minutes_contrast", "wall_minutes_lr",
                                                  "wall_minutes_ccs", "wall_minutes_dr1", "wall_minutes_total"]},
        "behaviour": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                      if k not in ("Q4", "degenerate")},
        "Q4": m.get("Q4"),
        "lr": rec.get("lr"), "ccs": rec.get("ccs"), "dr1": rec.get("dr1"),
    }
    with open(path, "w") as f:
        for k, v in blocks.items():
            f.write(f"## {k}\n{json.dumps(v, ensure_ascii=False, default=str)}\n")
    print(open(path).read())


if __name__ == "__main__":
    main()
