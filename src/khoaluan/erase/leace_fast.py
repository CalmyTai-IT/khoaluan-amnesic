"""
leace_fast.py — LEACE / Mean Projection khớp `concept-erasure` 0.2.4 (SPEC_ERASURE §4.1–4.2), viết lại để chạy null nhanh.

Toán học giống hệt LeaceFitter(method, affine=True, shrinkage=True, constrain_cov_trace=False, svd_tol=0.01):
  Σ_xx (co rút tuyến tính tối ưu) -> W = Σ^{-1/2}, W⁺ = Σ^{1/2} (cắt trị riêng như pinv);
  Σ_xz = X_cᵀ Z_c / (n − 1);  u, s = SVD(W Σ_xz), bỏ cột có s ≤ svd_tol;
  P_left = W⁺ u,  P_right = uᵀ W;  r(x) = x − ((x − μ) P_rightᵀ) P_leftᵀ.
Σ_xx, W, μ KHÔNG phụ thuộc nhãn khái niệm -> tính một lần cho mỗi tập fit; null hoán vị (N3) chỉ tính lại Σ_xz.
Unit test so khớp số học với LeaceFitter (tests/test_erase.py).

Vì head của BERT tuyến tính (logit = W_cls h + b), hiệu ứng của eraser lên logit:
  logit' = logit − ((h − μ) P_rightᵀ)(W_cls P_left)ᵀ   -> mỗi lần null chỉ tốn một phép nhân n×d×r.
"""
import numpy as np
import torch
from concept_erasure import optimal_linear_shrinkage

DT = torch.float64


def one_hot(z, k=None):
    z = np.asarray(z, dtype=int)
    k = k or int(z.max()) + 1
    out = np.zeros((len(z), k))
    out[np.arange(len(z)), z] = 1.0
    return out


class FitStats:
    """Thống kê không phụ thuộc nhãn của tập fit X (n×d)."""

    def __init__(self, X, method="leace", shrinkage=True):
        X = torch.as_tensor(np.asarray(X), dtype=DT)
        self.n, self.d = X.shape
        self.method = method
        self.mu = X.mean(0)
        self.Xc = X - self.mu
        eye = torch.eye(self.d, dtype=DT)
        if method == "leace":
            S = self.Xc.T @ self.Xc
            S = (S + S.T) / 2
            sigma = optimal_linear_shrinkage(S / self.n, self.n) if shrinkage else S / (self.n - 1)
            L, V = torch.linalg.eigh(sigma)
            mask = L > (L[-1] * sigma.shape[-1] * torch.finfo(L.dtype).eps)
            L = L.clamp_min(0.0)
            self.W = V * torch.where(mask, L.rsqrt(), torch.zeros_like(L)) @ V.T
            self.W_inv = V * torch.where(mask, L.sqrt(), torch.zeros_like(L)) @ V.T
            self.trace_x = float(torch.trace(sigma))
        elif method == "orth":
            self.W = self.W_inv = eye
            self.trace_x = float(torch.trace(self.Xc.T @ self.Xc) / (self.n - 1))
        else:
            raise ValueError(method)

    def eraser(self, Z, svd_tol=0.01):
        """Z: (n×k) one-hot (hoặc mảng nhãn int). Trả về Eraser."""
        Z = torch.as_tensor(one_hot(Z) if np.ndim(Z) == 1 else np.asarray(Z), dtype=DT)
        Zc = Z - Z.mean(0)
        sigma_xz = self.Xc.T @ Zc / (self.n - 1)
        u, s, _ = torch.linalg.svd(self.W @ sigma_xz, full_matrices=False)
        keep = s > svd_tol
        u = u * keep
        return Eraser(self.mu, self.W_inv @ u, u.T @ self.W, rank=int(keep.sum()))

    def random_eraser(self, rank, gen):
        """N1: chiếu trực giao affine quanh μ theo `rank` hướng ngẫu nhiên đẳng hướng."""
        Q, _ = torch.linalg.qr(torch.randn(self.d, rank, dtype=DT, generator=gen))
        return Eraser(self.mu, Q, Q.T, rank=rank)


class Eraser:
    def __init__(self, mu, proj_left, proj_right, rank):
        self.mu, self.PL, self.PR, self.rank = mu, proj_left, proj_right, rank

    def __call__(self, X):
        X = torch.as_tensor(np.asarray(X), dtype=DT)
        return (X - ((X - self.mu) @ self.PR.T) @ self.PL.T).numpy()

    def logit_shift(self, D, W_cls):
        """D = X − μ (torch, n×d) tính sẵn; W_cls (2×d). Trả về Δlogit (n×2) = −(D P_Rᵀ)(W_cls P_L)ᵀ."""
        return -((D @ self.PR.T) @ (W_cls @ self.PL).T)

    def on(self, device, dtype=torch.float32):
        """Bản tensor trên thiết bị (dùng cho can thiệp trong mô hình, L_ℓ)."""
        return TorchEraser(self.mu.to(device, dtype), self.PL.to(device, dtype), self.PR.to(device, dtype), self.rank)


class TorchEraser:
    def __init__(self, mu, PL, PR, rank):
        self.mu, self.PL, self.PR, self.rank = mu, PL, PR, rank

    def __call__(self, X):
        """X (..., d) cùng thiết bị; tính theo dtype của eraser (tắt autocast để phép chiếu không bị hạ xuống fp16),
        trả về dtype của X."""
        with torch.autocast(device_type=X.device.type, enabled=False):
            Xf = X.to(self.mu.dtype)
            return (Xf - ((Xf - self.mu) @ self.PR.T) @ self.PL.T).to(X.dtype)


def mean_diff_direction(X, z, W):
    """Hướng khái niệm nhị phân trong không gian whitened chung: W (μ_1 − μ_0) / ‖·‖.
    Với Z one-hot 2 lớp, W Σ_xz có hạng 1 và cùng phương với W(μ_1 − μ_0) -> chính là hướng LEACE xoá (SPEC_DULIEU §6.1
    bước 7: cosine giữa các hướng xoá trong không gian whitened; dùng chung một W để các hướng so được với nhau)."""
    X = torch.as_tensor(np.asarray(X), dtype=DT)
    z = np.asarray(z)
    v = W @ (X[z == 1].mean(0) - X[z == 0].mean(0))
    return (v / v.norm()).numpy()


def var_removed(stats, eraser):
    """VarRemoved = 1 − tr(Σ_{h'}) / tr(Σ_h) trên tập fit (dùng covariance mẫu, không co rút)."""
    Xe = stats.Xc - (stats.Xc @ eraser.PR.T) @ eraser.PL.T
    t0 = float((stats.Xc ** 2).sum())
    return 1.0 - float((Xe ** 2).sum()) / t0 if t0 > 0 else float("nan")     # tập fit không có phương sai


def stratified_permutation(z, strata, rng):
    """Hoán vị nhãn khái niệm trong từng ô (giữ nguyên số lượng mỗi ô)."""
    z = np.asarray(z).copy()
    strata = np.asarray(strata)
    out = z.copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        out[idx] = z[idx][rng.permutation(len(idx))]
    return out
