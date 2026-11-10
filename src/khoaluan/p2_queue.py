"""
p2_queue.py — chạy TUẦN TỰ một danh sách mô hình trên MỘT GPU (mỗi mô hình một tiến trình riêng),
tự áp quy tắc tiền đăng ký SPEC_REPRODUCE §3.3: run thoái hoá -> giữ bản ghi, chạy lại với seed + 100, rồi seed + 200.

    CUDA_VISIBLE_DEVICES=0 python scripts/run_queue.py --data-root /kaggle/input --out-root /kaggle/working/out \
        --account TV-B --specs mnliB-16000-11 mnliB-16000-33

Spec: "<data>-<size>-<seed>" với data ∈ {mnliB, mix, pt}; M-PT viết "pt-0-0" (không fine-tune, không chạy lại).
Một mô hình lỗi (mã thoát ≠ 0) -> DỪNG hàng đợi (lỗi thường có tính hệ thống); kết quả đã có vẫn giữ.
"""
import json
import os
import subprocess
import sys
import time

MAX_RETRIES = 2          # seed, seed+100, seed+200
SIZE_TAG = {25: "25", 100: "100", 1000: "1k", 8000: "8k", 16000: "16k"}


def parse_spec(spec):
    data, size, seed = spec.split("-")
    if data not in ("mnliB", "mix", "pt"):
        raise ValueError(f"spec không hợp lệ: {spec}")
    return {"data": data, "size": int(size), "seed": int(seed)}


def run_id_for(data, size, seed, suffix=""):
    if data == "pt":
        return "bertL-pt" + suffix
    tag = "mix-16k" if data == "mix" else f"mnliB-{SIZE_TAG[size]}"
    return f"bertL-{tag}-s{seed}{suffix}"


def subprocess_runner(argv):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "run_p2.py")
    return subprocess.call([sys.executable, script] + argv)


def run_queue(specs, base_argv, out_root, runner=subprocess_runner, max_retries=MAX_RETRIES, log=print):
    """base_argv: tham số chung cho run_p2 (--data-root, --out-root, --account, ...). Trả về dict báo cáo."""
    report = {"specs": specs, "runs": [], "stopped_on_error": None}
    for spec in specs:
        s = parse_spec(spec)
        tries = 1 if s["data"] == "pt" else max_retries + 1
        for k in range(tries):
            seed = s["seed"] + 100 * k
            argv = base_argv + ["--data", s["data"], "--size", str(s["size"]), "--seed", str(seed)]
            rid = run_id_for(s["data"], s["size"], seed)
            log(f"=== [{time.strftime('%H:%M:%S')}] bắt đầu {rid} (spec {spec}, lần {k + 1}/{tries})")
            t0 = time.time()
            rc = runner(argv)
            entry = {"spec": spec, "run_id": rid, "seed": seed, "attempt": k + 1, "exit_code": rc,
                     "minutes": round((time.time() - t0) / 60, 2), "degenerate": None, "reason": None}
            if rc != 0:
                report["runs"].append(entry)
                report["stopped_on_error"] = rid
                log(f"=== LỖI {rid} (mã {rc}) -> dừng hàng đợi")
                return report
            mpath = os.path.join(out_root, rid, "p2_metrics.json")
            with open(mpath) as f:
                rec = json.load(f)
            entry["degenerate"], entry["reason"] = bool(rec.get("degenerate")), rec.get("degenerate_reason")
            report["runs"].append(entry)
            log(f"=== xong {rid}: {entry['minutes']} phút, thoái hoá={entry['degenerate']}")
            if not entry["degenerate"]:
                break
            if k + 1 < tries:
                log(f"=== {rid} thoái hoá ({entry['reason']}) -> chạy lại seed {seed + 100} (SPEC §3.3)")
            else:
                log(f"=== CẢNH BÁO: {spec} thoái hoá cả {tries} lần -> nghi vấn hệ thống, báo lại Claude")
    return report


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--account", default=None)
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--report", default=None, help="đường dẫn queue_report.json")
    a, extra = ap.parse_known_args(argv)
    base = ["--data-root", a.data_root, "--out-root", a.out_root] + (["--account", a.account] if a.account else []) \
        + extra
    rep = run_queue(a.specs, base, a.out_root)
    path = a.report or os.path.join(a.out_root, f"queue_report_{os.environ.get('CUDA_VISIBLE_DEVICES', 'x')}.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    sys.exit(1 if rep["stopped_on_error"] else 0)


if __name__ == "__main__":
    main()
