"""
finetune.py — S2: fine-tune bert-large-cased trên MNLI nhị phân / MIX (SPEC_REPRODUCE_VDL v1.1 §3).

- Siêu tham số lấy từ configs/finetune/base.yaml (D-01); phần V&dL không nêu = mặc định HF Trainer
  của transformers 5.x đang chạy (ghi lại optim, max_grad_norm thực tế vào kết quả).
- Không eval/chọn checkpoint trong lúc train (eval_strategy="no"); lưu một lần ở cuối.
- CHỐT CHẶN: Trainer tự bật DataParallel khi thấy >1 GPU -> batch thực = 8 × số GPU (khác V&dL).
  Vì vậy mỗi tiến trình chỉ được thấy 1 GPU (CUDA_VISIBLE_DEVICES), và kiểm batch thực = 8,
  số bước = ceil(n/8) × epoch.
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


def _progress_callback(total_steps, every=500):
    from transformers import TrainerCallback

    class _P(TrainerCallback):
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
    return _P()


def load_model_and_tokenizer(model_name, num_labels, seed):
    """Seed đặt TRƯỚC from_pretrained để khởi tạo classifier head tất định theo seed."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    set_all_seeds(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tok


def finetune(frame, cfg, seed, work_dir, model_dir=None, log=print):
    """Huấn luyện trên frame (uid, premise, hypothesis, y). Trả về (model, tokenizer, stats).
    model_dir != None -> lưu checkpoint (safetensors) + tokenizer vào đó."""
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
    expected_steps = math.ceil(len(ds) / bs) * int(cfg["num_train_epochs"])
    trainer = Trainer(model=model, args=args, train_dataset=ds, processing_class=tok,
                      data_collator=DataCollatorWithPadding(tok),
                      callbacks=[_progress_callback(expected_steps)])
    if trainer.args.n_gpu > 1 or trainer.args.train_batch_size != bs:
        raise SystemExit(f"[DỪNG] n_gpu={trainer.args.n_gpu}, batch thực={trainer.args.train_batch_size} "
                         f"(phải là 1 GPU, batch {bs})")
    log(f"train: n={len(ds)} | batch={trainer.args.train_batch_size} | bước dự kiến={expected_steps} | "
        f"optim={trainer.args.optim} | max_grad_norm={trainer.args.max_grad_norm} | fp16={trainer.args.fp16} | "
        f"cắt cụt {trunc_rate:.4%}")
    t0 = time.time()
    out = trainer.train()
    wall = (time.time() - t0) / 60
    steps = trainer.state.global_step
    if steps != expected_steps:
        raise SystemExit(f"[DỪNG] số bước {steps} != dự kiến {expected_steps}")
    losses = [r["loss"] for r in trainer.state.log_history if "loss" in r]
    stats = {
        "n_train": len(ds), "train_label_counts": {int(k): int(v) for k, v in frame.y.value_counts().items()},
        "steps": steps, "expected_steps": expected_steps, "train_batch_size": trainer.args.train_batch_size,
        "optim": str(getattr(trainer.args.optim, "value", trainer.args.optim)), "max_grad_norm": trainer.args.max_grad_norm,
        "fp16": bool(trainer.args.fp16), "train_loss_mean": float(out.training_loss),
        "train_loss_last_logged": float(losses[-1]) if losses else None,
        "train_loss_curve": losses, "truncation_rate": trunc_rate, "wall_minutes_train": round(wall, 2),
    }
    if torch.cuda.is_available():
        stats["peak_mem_gb_train"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    shutil.rmtree(f"{work_dir}/trainer_tmp", ignore_errors=True)  # save_strategy="no" -> thư mục rỗng
    if model_dir:
        trainer.save_model(model_dir)
        tok.save_pretrained(model_dir)
        stats["model_dir"] = model_dir
    model = trainer.model
    model.eval()
    return model, tok, stats
