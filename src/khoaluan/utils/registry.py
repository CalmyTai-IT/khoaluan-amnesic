"""Registry kết quả (root §3.7; SPEC_REPRODUCE_VDL §2.4). Mỗi run ghi 1 run_record.json; gộp thành results/registry.csv."""
import csv
import datetime as dt
import hashlib
import json
import os

COLUMNS = [
    "run_id", "kind", "git_commit", "config_sha256", "data_manifest_sha256", "seed", "data_seed",
    "kaggle_account", "kaggle_notebook_version", "gpu_name", "torch", "transformers",
    "start_utc", "wall_minutes", "train_loss_final", "devm_acc", "devmm_acc",
    "hans_E_LO", "hans_E_SUB", "hans_E_CON", "hans_NE_LO", "hans_NE_SUB", "hans_NE_CON",
    "truncation_rate", "degenerate", "notes",
]


def sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_run_record(out_dir, **fields):
    """Ghi run_record.json; trường không biết -> None. Trường thừa -> giữ trong 'extra'."""
    rec = {k: fields.pop(k, None) for k in COLUMNS}
    if fields:
        rec["extra"] = fields
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "run_record.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    return path


def merge(record_paths, registry_csv):
    """Thêm các run_record.json vào registry.csv (bỏ qua run_id đã có)."""
    existing = set()
    if os.path.exists(registry_csv):
        with open(registry_csv) as f:
            existing = {r["run_id"] for r in csv.DictReader(f)}
    new_file = not os.path.exists(registry_csv)
    added = 0
    with open(registry_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for p in record_paths:
            rec = json.load(open(p))
            if rec.get("run_id") in existing:
                print(f"bỏ qua (đã có): {rec.get('run_id')}")
                continue
            w.writerow(rec)
            existing.add(rec.get("run_id"))
            added += 1
    return added
