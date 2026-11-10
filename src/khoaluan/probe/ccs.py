"""
ccs.py — CCS (Burns et al., ICLR 2023) theo lớp `CCS` của mã DLK / CCS.ipynb (SPEC_REPRODUCE_VDL v1.1 §6.2).

Tham số mặc định của DLK: nepochs=1000, ntries=10, lr=1e-3, full batch, AdamW(weight_decay=0.01),
linear=True (probe tuyến tính + sigmoid), var_normalize=False.
- x0 = h_neg = h(Yes) (đáp án của nhãn 0), x1 = h_pos = h(No) — cùng quy ước với lr.py.
- Mỗi tập trừ trung bình CỦA CHÍNH NÓ; khi đánh giá, tập TEST cũng trừ trung bình của chính tập test (đúng get_acc của DLK).
- loss = mean(min(p0, p1)^2) + mean((p0 − (1 − p1))^2); chọn probe có loss CUỐI nhỏ nhất trong ntries lần thử.
- acc = max(acc, 1 − acc) (CCS không biết dấu) -> mức chance với 1,000 ví dụ test ≈ 0.513 (phân vị 95% ≈ 0.531).

Khác về cài đặt (không khác về thuật toán): ntries lần thử được huấn luyện ĐỒNG THỜI như ntries probe độc lập
(tham số riêng, loss cộng lại -> gradient của mỗi probe chỉ đến từ loss của nó; AdamW tác động theo từng phần tử),
khởi tạo giống nn.Linear mặc định. Tương đương phân phối với vòng lặp tuần tự của DLK, nhanh hơn ~ntries lần.
Biến thể MLP: 1 tầng ẩn 100 + ReLU (MLPProbe của DLK).
"""
import math

import numpy as np
import torch


def _uniform(shape, bound, gen, device):
    return (torch.rand(shape, generator=gen, device=device) * 2 - 1) * bound


class _Probes(torch.nn.Module):
    """ntries probe độc lập. Tuyến tính: W (t,d,1), b (t,1,1). MLP: W1 (t,d,100), b1, W2 (t,100,1), b2."""

    def __init__(self, d, ntries, linear, gen, device):
        super().__init__()
        self.linear = linear
        bd = 1 / math.sqrt(d)  # nn.Linear: kaiming_uniform(a=√5) -> bound = 1/√fan_in; bias cùng bound
        if linear:
            self.W = torch.nn.Parameter(_uniform((ntries, d, 1), bd, gen, device))
            self.b = torch.nn.Parameter(_uniform((ntries, 1, 1), bd, gen, device))
        else:
            h = 100
            self.W1 = torch.nn.Parameter(_uniform((ntries, d, h), bd, gen, device))
            self.b1 = torch.nn.Parameter(_uniform((ntries, 1, h), bd, gen, device))
            bh = 1 / math.sqrt(h)
            self.W2 = torch.nn.Parameter(_uniform((ntries, h, 1), bh, gen, device))
            self.b2 = torch.nn.Parameter(_uniform((ntries, 1, 1), bh, gen, device))

    def forward(self, x):  # x (n,d) -> (t,n,1)
        if self.linear:
            return torch.sigmoid(torch.einsum("nd,tdo->tno", x, self.W) + self.b)
        h = torch.relu(torch.einsum("nd,tdh->tnh", x, self.W1) + self.b1)
        return torch.sigmoid(torch.bmm(h, self.W2) + self.b2)


def ccs_loss(p0, p1):
    """Loss của DLK cho từng probe: p0, p1 (t,n,1) -> (t,)."""
    informative = (torch.minimum(p0, p1) ** 2).mean(dim=1)
    consistent = ((p0 - (1 - p1)) ** 2).mean(dim=1)
    return (informative + consistent)[:, 0]


def _center(x):
    return x - x.mean(axis=0, keepdims=True)


def fit_eval_ccs(x0_tr, x1_tr, x0_te, x1_te, y_te, linear=True, nepochs=1000, ntries=10, lr=1e-3,
                 weight_decay=0.01, seed=0, device=None):
    """Huấn luyện CCS trên (x0_tr, x1_tr) — KHÔNG dùng nhãn — rồi đo trên test. Trả về (acc, best_final_loss)."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gen = torch.Generator(device=device).manual_seed(int(seed))
    x0 = torch.tensor(_center(np.asarray(x0_tr, np.float32)), device=device)
    x1 = torch.tensor(_center(np.asarray(x1_tr, np.float32)), device=device)
    probes = _Probes(x0.shape[1], ntries, linear, gen, device)
    opt = torch.optim.AdamW(probes.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(nepochs):
        losses = ccs_loss(probes(x0), probes(x1))
        opt.zero_grad()
        losses.sum().backward()
        opt.step()
    final = losses.detach()      # như DLK.train(): loss của epoch cuối, tính TRƯỚC bước cập nhật cuối
    with torch.no_grad():
        best = int(torch.argmin(final))
        t0 = torch.tensor(_center(np.asarray(x0_te, np.float32)), device=device)
        t1 = torch.tensor(_center(np.asarray(x1_te, np.float32)), device=device)
        p0, p1 = probes(t0)[best, :, 0], probes(t1)[best, :, 0]
        avg_conf = 0.5 * (p0 + (1 - p1))
        pred = (avg_conf < 0.5).long().cpu().numpy()
    acc = float((pred == np.asarray(y_te)).mean())
    return max(acc, 1 - acc), float(final[best])


def run_ccs(h_yes, h_no, y, splits, layers, layer_ids, mlp_layers=(), dataset="", log=None, device=None):
    """Như lr.run_lr: cùng các lần chia train/test. Trả về DataFrame dài: dataset, layer, run, variant, acc, final_loss."""
    import pandas as pd
    pos = {l: i for i, l in enumerate(layer_ids)}
    rows = []
    for l in layers:
        a = np.asarray(h_yes[:, pos[l]], dtype=np.float32)
        b = np.asarray(h_no[:, pos[l]], dtype=np.float32)
        variants = [("linear", True)] + ([("mlp", False)] if l in mlp_layers else [])
        for r, (tr, te) in enumerate(splits, start=1):
            for name, lin in variants:
                acc, loss = fit_eval_ccs(a[tr], b[tr], a[te], b[te], y[te], linear=lin, seed=1000 * l + r,
                                         device=device)
                rows.append({"dataset": dataset, "layer": l, "run": r, "variant": name, "acc": acc,
                             "final_loss": loss})
        if log:
            sub = [x["acc"] for x in rows if x["layer"] == l and x["variant"] == "linear"]
            log(f"  CCS {dataset} tầng {l:2d}: linear {np.mean(sub):.3f}")
    return pd.DataFrame(rows)
