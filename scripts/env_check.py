"""
env_check.py — A-04: kiểm môi trường Kaggle + đo tốc độ fine-tune bert-large-cased thật (fp16).

    python scripts/env_check.py --out /kaggle/working/env                      # chỉ thông tin môi trường
    python scripts/env_check.py --out /kaggle/working/env --bench              # + benchmark 1 GPU
    python scripts/env_check.py --out /kaggle/working/env --bench --parallel   # + 2 GPU chạy đồng thời

Đầu ra: env.json, requirements.lock (pip freeze), bench_*.json, A04_summary.txt (gửi lại cho Claude).
Benchmark dùng input token ngẫu nhiên (không cần dữ liệu); đo giây/bước ở batch 8, độ dài 64 và 128,
với đúng optimizer/AMP như SPEC_REPRODUCE_VDL §3.2 → ước lượng thời gian M-16K (10,000 bước).
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from khoaluan.utils.env import collect_env, pip_freeze  # noqa: E402

MODEL = "google-bert/bert-large-cased"


def bench(device_index, steps=40, warmup=10, seq_lens=(64, 128), batch=8):
    import torch
    from transformers import AutoModelForSequenceClassification
    torch.manual_seed(0)
    dev = torch.device(f"cuda:{device_index}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    try:
        scaler = torch.amp.GradScaler("cuda")
        autocast = lambda: torch.amp.autocast("cuda", dtype=torch.float16)  # noqa: E731
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
        autocast = lambda: torch.cuda.amp.autocast(dtype=torch.float16)  # noqa: E731
    vocab = model.config.vocab_size
    res = {"gpu": torch.cuda.get_device_name(device_index), "batch": batch}
    for L in seq_lens:
        ids = torch.randint(1000, vocab, (batch, L), device=dev)
        labels = torch.randint(0, 2, (batch,), device=dev)
        torch.cuda.reset_peak_memory_stats(dev)
        for i in range(warmup + steps):
            if i == warmup:
                torch.cuda.synchronize(dev)
                t = time.time()
            with autocast():
                loss = model(input_ids=ids, labels=labels).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize(dev)
        sps = (time.time() - t) / steps
        res[f"sec_per_step_len{L}"] = round(sps, 4)
        res[f"peak_mem_gb_len{L}"] = round(torch.cuda.max_memory_allocated(dev) / 1e9, 2)
        res[f"est_minutes_10k_steps_len{L}"] = round(sps * 10000 / 60, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/env")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--parallel", action="store_true", help="chạy benchmark đồng thời trên GPU 0 và 1")
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.worker is not None:  # tiến trình con của --parallel
        r = bench(0)  # CUDA_VISIBLE_DEVICES đã giới hạn còn 1 GPU
        json.dump(r, open(os.path.join(args.out, f"bench_parallel_gpu{args.worker}.json"), "w"), indent=2)
        return

    env = collect_env()
    json.dump(env, open(os.path.join(args.out, "env.json"), "w"), indent=2, ensure_ascii=False)
    open(os.path.join(args.out, "requirements.lock"), "w").write(pip_freeze())
    summary = {"env": env}

    if args.bench:
        print("Benchmark 1 GPU (tải bert-large-cased ~1.3 GB lần đầu)...", flush=True)
        r1 = bench(0)
        json.dump(r1, open(os.path.join(args.out, "bench_single.json"), "w"), indent=2)
        summary["bench_single"] = r1
        if args.parallel and len(env.get("gpus", [])) >= 2:
            print("Benchmark 2 GPU đồng thời...", flush=True)
            procs = []
            for g in (0, 1):
                e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                procs.append(subprocess.Popen([sys.executable, __file__, "--out", args.out, "--worker", str(g)], env=e))
            for p in procs:
                p.wait()
            summary["bench_parallel"] = [json.load(open(os.path.join(args.out, f"bench_parallel_gpu{g}.json")))
                                         for g in (0, 1)]

    with open(os.path.join(args.out, "A04_summary.txt"), "w") as f:
        f.write(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
