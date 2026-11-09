# %% [markdown]
# # Khoá luận — notebook mẫu Kaggle (A-09)
# Notebook này **chỉ** làm 4 việc: (1) lấy code đúng commit, (2) cài gói thiếu, (3) chạy một TASK bằng script trong repo,
# (4) ghi `run_record.json`. Không viết logic thí nghiệm trong notebook (root §3.7: notebooks chỉ để xem).
#
# Trước khi chạy: Settings (panel phải) → Internet **On**; Accelerator: **None** cho `prepare_data`, **GPU T4 x2** cho `env_bench`.
# Add-ons → Secrets → bật secret `GITHUB_TOKEN` cho notebook này (nếu repo private).
# Chạy chính thức bằng **Save Version → Save & Run All (Commit)**; chạy thử bằng Run All trong phiên tương tác.

# %%
# ===== CẤU HÌNH — chỉ sửa ô này =====
GITHUB_REPO = "TEN_TAI_KHOAN_GITHUB/khoaluan-amnesic"   # <-- sửa
COMMIT = "main"            # chạy chính thức: thay bằng mã commit đầy đủ (40 ký tự), KHÔNG dùng "main"
TASK = "env"               # "env" | "env_bench" | "prepare_data"
KAGGLE_ACCOUNT = "TV-A"    # "TV-A" (Tài) | "TV-B" (Thuing) — để ghi registry
RUN_ID = {"env": "env-check", "env_bench": "env-bench", "prepare_data": "data-prepare-v1"}[TASK]
REPO_DIR = "/kaggle/working/repo"

# %%
import os, subprocess, sys, json, time
t_start = time.time()

class _R:  # kết quả lệnh: .stdout (đã che token), .returncode
    pass

def sh(cmd, check=True):
    """Chạy lệnh shell, in log theo thời gian thực, che token nếu có."""
    mask = lambda t: t.replace(os.environ["GH_TOKEN"], "***") if os.environ.get("GH_TOKEN") else t
    print("$", mask(cmd), flush=True)
    p = subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    for line in p.stdout:
        line = mask(line)
        lines.append(line)
        print(line, end="", flush=True)
    r = _R(); r.returncode = p.wait(); r.stdout = "".join(lines)
    if check and r.returncode != 0:
        raise RuntimeError(f"Lệnh lỗi (mã {r.returncode})")
    return r

# Lấy code: ưu tiên git (có commit hash). Repo private cần secret GITHUB_TOKEN (fine-grained PAT, chỉ đọc).
token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception as e:
    print("Không lấy được secret GITHUB_TOKEN (bỏ qua nếu repo public):", type(e).__name__)
if token:
    os.environ["GH_TOKEN"] = token
    url = f"https://x-access-token:{token}@github.com/{GITHUB_REPO}.git"
else:
    url = f"https://github.com/{GITHUB_REPO}.git"
if not os.path.exists(REPO_DIR):
    sh(f"git clone --quiet {url} {REPO_DIR}")
else:
    sh(f"git -C {REPO_DIR} remote set-url origin {url}")
REF = "origin/main" if COMMIT == "main" else COMMIT
sh(f"git -C {REPO_DIR} fetch --quiet origin && git -C {REPO_DIR} checkout --quiet {REF}")
GIT_COMMIT = sh(f"git -C {REPO_DIR} rev-parse HEAD").stdout.strip()
sh(f"git -C {REPO_DIR} remote set-url origin https://github.com/{GITHUB_REPO}.git")  # xoá token khỏi .git/config
print("GIT_COMMIT =", GIT_COMMIT)

# %%
sh(f"pip install -q -r {REPO_DIR}/requirements.txt")
os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

# %%
if TASK == "env":
    sh("python scripts/env_check.py --out /kaggle/working/env")
elif TASK == "env_bench":
    sh("python scripts/env_check.py --out /kaggle/working/env --bench --parallel")
elif TASK == "prepare_data":
    sh("python -m pytest -q tests/")
    sh("python scripts/prepare_data.py --out /kaggle/working/data --data-seed 1234")
else:
    raise ValueError(TASK)

# %%
from khoaluan.utils.registry import write_run_record, sha256_file
from khoaluan.utils.env import collect_env
env = collect_env(check_net=False)
write_run_record(
    "/kaggle/working",
    run_id=RUN_ID, kind=TASK, git_commit=GIT_COMMIT,
    data_manifest_sha256=sha256_file("/kaggle/working/data/manifest.json"),
    data_seed=1234 if TASK == "prepare_data" else None,
    kaggle_account=KAGGLE_ACCOUNT,
    gpu_name=(env.get("gpus") or [{}])[0].get("name"),
    torch=env["packages"].get("torch"), transformers=env["packages"].get("transformers"),
    start_utc=None, wall_minutes=round((time.time() - t_start) / 60, 1),
    notes="A-09 template run",
)
print(open("/kaggle/working/run_record.json").read())
# Xoá bản clone khỏi output để không đóng gói code vào output version (code đã có trên GitHub theo commit).
os.chdir("/kaggle/working")
sh(f"rm -rf {REPO_DIR}", check=False)
