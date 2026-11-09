"""
forward.py — chạy encoder theo lô (sắp theo độ dài để giảm padding), trả representation theo đúng thứ tự gốc.

Dùng cho:
- inference đầu vào chuẩn NLI (cặp câu, max_length 128 như lúc train) -> logits + pooler_output (= L_out, D-06)
  + [CLS] các tầng chỉ định (SPEC_REPRODUCE §4, §5.4);
- contrast pair một đoạn có prompt (SPEC_REPRODUCE §5.1–5.2).

Quy ước tầng: hidden_states[0] = embedding, [1..L] = tầng Transformer (root §3.7).
"""
import contextlib

import numpy as np
import torch


def encoder_of(model):
    """BertForSequenceClassification -> model.bert; BertModel -> chính nó."""
    return getattr(model, "bert", model)


def _autocast(device, fp16):
    if fp16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _batches(lengths, batch_size):
    order = np.argsort(np.asarray(lengths), kind="stable")
    for i in range(0, len(order), batch_size):
        yield order[i:i + batch_size]


def _pad(tokenizer, enc, idx, device):
    feats = [{k: enc[k][j] for k in enc} for j in idx]
    b = tokenizer.pad(feats, return_tensors="pt")
    return {k: v.to(device) for k, v in b.items()}


@torch.no_grad()
def run_pairs(model, tokenizer, frame, max_length, batch_size, fp16, cls_layers=(), check_logits=True):
    """Inference đầu vào chuẩn. Trả về dict: logits (N,2) f32, pooler (N,H) f32, cls (N,len(cls_layers),H) f32."""
    device = next(model.parameters()).device
    model.eval()
    enc = tokenizer(frame.premise.tolist(), frame.hypothesis.tolist(), truncation=True, max_length=max_length)
    enc = {k: enc[k] for k in ("input_ids", "token_type_ids", "attention_mask") if k in enc}
    n, hdim = len(frame), model.config.hidden_size
    has_head = hasattr(model, "classifier")
    out = {"logits": np.zeros((n, 2), np.float32) if has_head else None,
           "pooler": np.zeros((n, hdim), np.float32),
           "cls": np.zeros((n, len(cls_layers), hdim), np.float32)}
    first = True
    for idx in _batches([len(x) for x in enc["input_ids"]], batch_size):
        b = _pad(tokenizer, enc, idx, device)
        with _autocast(device, fp16):
            o = encoder_of(model)(**b, output_hidden_states=True)
            pooled = o.pooler_output
            if has_head:
                logits = model.classifier(model.dropout(pooled))
                if check_logits and first:  # logits tự tính từ pooler phải trùng forward chuẩn của model
                    ref = model(**b).logits
                    if not torch.allclose(logits.float(), ref.float(), atol=1e-3, rtol=1e-3):
                        raise RuntimeError("[DỪNG] classifier(pooler_output) != model(...).logits — kiểm lại L_out")
                    first = False
        if has_head:
            out["logits"][idx] = logits.float().cpu().numpy()
        out["pooler"][idx] = pooled.float().cpu().numpy()
        if cls_layers:
            hs = o.hidden_states
            out["cls"][idx] = torch.stack([hs[l][:, 0] for l in cls_layers], 1).float().cpu().numpy()
    return out


def contrast_texts(premise, hypothesis):
    """PromptSource HANS 'does it follow that' (id 591a436a-…), ghép đáp án kiểu DLK: question + ' ' + answer.
    Trả về (text_yes, text_no). Giữ nguyên văn bản gốc (HANS có khoảng trắng trước dấu chấm)."""
    q = f"Given that {premise} Does it follow that {hypothesis} Yes or no?"
    return q + " Yes", q + " No"


@torch.no_grad()
def run_contrast(model, tokenizer, frame, batch_size, fp16, cls_layers, aux_layers, out_arrays):
    """Trích representation contrast pair (một đoạn, token_type_ids = 0).
    out_arrays: dict tên -> mảng ghi được (np.memmap) với shape:
      cls_yes/cls_no (N, len(cls_layers), H); last_yes/last_no, mean_yes/mean_no (N, len(aux_layers), H).
    'last' = token không-pad cuối cùng (với BERT là [SEP]); 'mean' = trung bình mọi token không-pad (gồm [CLS], [SEP]).
    Trả về thống kê kiểm tra (cắt cụt, cặp trùng)."""
    device = next(model.parameters()).device
    model.eval()
    enc_model = encoder_of(model)
    max_len = int(getattr(tokenizer, "model_max_length", 512) or 512)
    max_len = min(max_len, int(model.config.max_position_embeddings))
    stats = {"n": len(frame), "n_truncated": 0, "n_identical_pairs": 0, "max_tokens": 0}
    for answer in ("yes", "no"):
        texts = [contrast_texts(p, h)[0 if answer == "yes" else 1]
                 for p, h in zip(frame.premise, frame.hypothesis)]
        full = tokenizer(texts, truncation=False)["input_ids"]
        stats["n_truncated"] += sum(len(x) > max_len for x in full)
        stats["max_tokens"] = max(stats["max_tokens"], max(len(x) for x in full))
        if answer == "yes":
            ids_yes = full
        else:
            stats["n_identical_pairs"] = sum(a == b for a, b in zip(ids_yes, full))
        enc = tokenizer(texts, truncation=True, max_length=max_len)
        enc = {k: enc[k] for k in ("input_ids", "token_type_ids", "attention_mask") if k in enc}
        for idx in _batches([len(x) for x in enc["input_ids"]], batch_size):
            b = _pad(tokenizer, enc, idx, device)
            with _autocast(device, fp16):
                hs = enc_model(**b, output_hidden_states=True).hidden_states
            m = b["attention_mask"]
            out_arrays[f"cls_{answer}"][idx] = torch.stack([hs[l][:, 0] for l in cls_layers], 1).float().cpu().numpy()
            if aux_layers:
                last = m.sum(1) - 1
                ar = torch.arange(m.shape[0], device=m.device)
                mf = m.unsqueeze(-1).float()
                out_arrays[f"last_{answer}"][idx] = torch.stack(
                    [hs[l][ar, last] for l in aux_layers], 1).float().cpu().numpy()
                out_arrays[f"mean_{answer}"][idx] = torch.stack(
                    [(hs[l].float() * mf).sum(1) / mf.sum(1) for l in aux_layers], 1).cpu().numpy()
    return stats
