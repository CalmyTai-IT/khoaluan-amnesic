"""
finetune.py — S2: fine-tune bert-large-cased trên MNLI nhị phân / MIX (SPEC_REPRODUCE_VDL v1.1 §3).

- Siêu tham số lấy từ configs/finetune/base.yaml (D-01); phần V&dL không nêu = mặc định HF Trainer
  của transformers 5.x đang chạy (ghi lại optim, max_grad_norm thực tế vào kết quả).
- Không eval/chọn checkpoint trong lúc train (eval_strategy="no"); 5 epoch cố định như V&dL; lưu một lần ở cuối.
  KHÔNG dừng sớm khi loss nhỏ (sẽ cho ra mô hình khác bài gốc).
- CHỐT CHẶN: Trainer tự bật DataParallel khi thấy >1 GPU -> batch thực = 8 × số GPU (khác V&dL).
  Vì vậy mỗi tiến trình chỉ được thấy 1 GPU (CUDA_VISIBLE_DEVICES), và kiểm batch thực = 8,
  số bước = ceil(n/8) × epoch (trừ khi run bị dừng vì thoái hoá, xem dưới).
- DỪNG SỚM RUN THOÁI HOÁ (quy tắc tiền đăng ký 10/11/2026, sau run bertL-mnliB-16k-s11 kẹt ở loss ≈ ln2 suốt 10,000 bước):
  chỉ áp cho tập train n ≥ ABORT_MIN_N (M-8K, M-16K, M-MIX). Hết epoch 1, nếu trung bình loss đã log ở NỬA SAU epoch 1
  ≥ ABORT_LOSS (0.65; chance = ln2 ≈ 0.693) -> dừng, gắn cờ thoái hoá, không lưu checkpoint.
  Hai run lành (s111, s22) có loss ~0.30–0.40 ở cùng thời điểm; run hỏng ~0.70.
"""
import math
import shutil
import time

import torch

from khoaluan.utils.seed import set_all_seeds

# Khoá của base.yaml được chuyển thẳng vào TrainingArguments
TA_KEYS = ["learning_rate", "per_device_train_batch_size", "num_train_epochs", "adam_beta1", "adam_beta2",
           "adam_epsilon", "weight_decay", "lr_scheduler_type", "warmup_steps", "fp16", "eval_strategy",
           "save_strategy", "logging_steps", "report_to", "use_cpu", "max_grad_norm", "optim"]
ABORT_MIN_N = 8000
ABORT_LOSS = 0.65


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.enc, self.labels = encodings, list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def tokenize_pairs(tokenizer, frame, max_length):
    """Mã hoá cặp (premise, hypothesis) đúng như lúc inference; trả về (encodings, truncation_rate)."""
    p, h = frame.premise.tolist(), frame.hypothesis.tolist()
    full = tokenizer(p, h, truncation=False)["input_ids"]
    n_trunc = sum(len(x) > max_length for x in full)
    enc = tokenizer(p, h, truncation=True, max_length=max_length)
    keep = [k for k in ("input_ids", "token_type_ids", "attention_mask") if k in enc]
    return {k: enc[k] for k in keep}, n_trunc / max(1, len(p))


def stuck_after_epoch1(log_history, steps_per_epoch, threshold=ABORT_LOSS):
    """Trung bình loss đã log có bước thuộc (E/2, E]. Trả về (bị_kẹt, giá_trị) — giá trị None nếu chưa có log nào."""
    vals = [r["loss"] for r in log_history
            if "loss" in r and steps_per_epoch / 2 < r.get("step", 0) <= steps_per_epoch]
    if not vals:
        return False, None
    m = sum(vals) / len(vals)
    return m >= threshold, m


def _callbacks(total_steps, steps_per_epoch, abort_enabled, threshold, every=500):
    from transformers import TrainerCallback

    class _Progress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.t0 = time.time()

        def on_step_end(self, args, state, control, **kw):
            s = state.global_step
            if s % every == 0 or s == total_steps:
                el = (time.time() - self.t0) / 60
                eta = el / s * (total_steps - s) if s else float("nan")
                loss = next((r["loss"] for r in reversed(state.log_history) if "loss" in r), float("nan"))
                print(f"  bước {s}/{total_steps} | {el:.1f} phút | còn ~{eta:.1f} phút | loss gần nhất {loss:.4f}",
                      flush=True)

    class _AbortIfStuck(TrainerCallback):
        def __init__(self):
            self.aborted, self.checked, self.value = False, False, None

        def on_log(self, args, state, control, **kw):
            # on_log chạy SAU khi log của bước hiện tại đã vào log_history; kiểm MỘT lần, ở lần log đầu tiên từ cuối epoch 1
            if abort_enabled and not self.checked and state.global_step >= steps_per_epoch:
                self.checked = True
                stuck, self.value = stuck_after_epoch1(state.log_history, steps_per_epoch, threshold)
                if stuck:
                    self.aborted = True
                    control.should_training_stop = True
                    print(f"  [THOÁI HOÁ] loss TB nửa sau epoch 1 = {self.value:.4f} ≥ {threshold} -> dừng run",
                          flush=True)

    return _Progress(), _AbortIfStuck()


def load_model_and_tokenizer(model_name, num_labels, seed):
    """Seed đặt TRƯỚC from_pretrained để khởi tạo classifier head tất định theo seed."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    set_all_seeds(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tok


def finetune(frame, cfg, seed, work_dir, model_dir=None, log=print):
    """Huấn luyện trên frame (uid, premise, hypothesis, y). Trả về (model, tokenizer, stats).
    model_dir != None -> lưu checkpoint (safetensors) + tokenizer vào đó (không lưu nếu run bị dừng vì thoái hoá).
    stats["aborted"] = True khi run bị dừng sớm vì kẹt ở mức chance."""
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        raise SystemExit("[DỪNG] tiến trình thấy >1 GPU -> Trainer sẽ dùng DataParallel (batch thực 16). "
                         "Đặt CUDA_VISIBLE_DEVICES=0 (hoặc 1) trước khi chạy.")
    model, tok = load_model_and_tokenizer(cfg["model_name"], cfg["num_labels"], seed)
    enc, trunc_rate = tokenize_pairs(tok, frame, cfg["max_length"])
    ds = PairDataset(enc, frame.y.tolist())

    ta = {k: cfg[k] for k in TA_KEYS if k in cfg}
    args = TrainingArguments(output_dir=f"{work_dir}/trainer_tmp", seed=seed, data_seed=seed,
                             disable_tqdm=True, save_only_model=True, **ta)
    bs = cfg["per_device_train_batch_size"]
    steps_per_epoch = math.ceil(len(ds) / bs)
    expected_steps = steps_per_epoch * int(cfg["num_train_epochs"])
    abort_min_n = int(cfg.get("abort_min_n", ABORT_MIN_N))
    abort_loss = float(cfg.get("abort_loss", ABORT_LOSS))
    abort_enabled = len(ds) >= abort_min_n and int(cfg["num_train_epochs"]) > 1
    progress, abort_cb = _callbacks(expected_steps, steps_per_epoch, abort_enabled, abort_loss)
    trainer = Trainer(model=model, args=args, train_dataset=ds, processing_class=tok,
                      data_collator=DataCollatorWithPadding(tok), callbacks=[progress, abort_cb])
    if trainer.args.n_gpu > 1 or trainer.args.train_batch_size != bs:
        raise SystemExit(f"[DỪNG] n_gpu={trainer.args.n_gpu}, batch thực={trainer.args.train_batch_size} "
                         f"(phải là 1 GPU, batch {bs})")
    optim = str(getattr(trainer.args.optim, "value", trainer.args.optim))
    log(f"train: n={len(ds)} | batch={trainer.args.train_batch_size} | bước dự kiến={expected_steps} | "
        f"optim={optim} | max_grad_norm={trainer.args.max_grad_norm} | fp16={trainer.args.fp16} | "
        f"cắt cụt {trunc_rate:.4%} | dừng-nếu-thoái-hoá={'bật' if abort_enabled else 'tắt'}")
    t0 = time.time()
    out = trainer.train()
    wall = (time.time() - t0) / 60
    steps = trainer.state.global_step
    aborted = abort_cb.aborted
    if not aborted and steps != expected_steps:
        raise SystemExit(f"[DỪNG] số bước {steps} != dự kiến {expected_steps}")
    losses = [r["loss"] for r in trainer.state.log_history if "loss" in r]
    stats = {
        "n_train": len(ds), "train_label_counts": {int(k): int(v) for k, v in frame.y.value_counts().items()},
        "steps": steps, "expected_steps": expected_steps, "train_batch_size": trainer.args.train_batch_size,
        "optim": optim, "max_grad_norm": trainer.args.max_grad_norm,
        "fp16": bool(trainer.args.fp16), "train_loss_mean": float(out.training_loss),
        "train_loss_last_logged": float(losses[-1]) if losses else None,
        "train_loss_curve": losses, "truncation_rate": trunc_rate, "wall_minutes_train": round(wall, 2),
        "aborted": aborted, "abort_rule": {"enabled": abort_enabled, "min_n": abort_min_n, "threshold": abort_loss,
                                           "epoch1_second_half_mean_loss": abort_cb.value},
    }
    if torch.cuda.is_available():
        stats["peak_mem_gb_train"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    shutil.rmtree(f"{work_dir}/trainer_tmp", ignore_errors=True)  # save_strategy="no" -> thư mục rỗng
    if model_dir and not aborted:
        trainer.save_model(model_dir)
        tok.save_pretrained(model_dir)
        stats["model_dir"] = model_dir
    model = trainer.model
    model.eval()
    return model, tok, stats
