# %% [markdown]
# # Khoá luận — notebook P2 (fine-tune + tái lập V&dL) — SPEC_REPRODUCE_VDL v1.1
# Notebook **chỉ**: lấy code đúng commit → cài gói thiếu → chạy unit test → gọi `scripts/run_p2.py` cho từng run → in `p2_summary.txt`.
# Không viết logic thí nghiệm ở đây.
#
# Trước khi chạy:
# 1. Settings → **Internet On**; **Accelerator: GPU T4 x2**.
# 2. Add Input → Dataset `khoaluan-data` (đầu ra prepare_data v1.2 của lần A-06 chính thức). Script tự tìm `manifest.json`
#    dưới `/kaggle/input` và **dừng** nếu vân tay dữ liệu khác lần A-06 chính thức.
# 3. Sửa ô CẤU HÌNH. Chạy chính thức bằng **Save Version → Save & Run All (Commit)**. Không để phiên tương tác có GPU mở không dùng
#    (vẫn bị trừ quota).

# %%
# ===== CẤU HÌNH — chỉ sửa ô này =====
GITHUB_REPO = "CalmyTai-IT/khoaluan-amnesic"
COMMIT = "main"            # chạy chính thức: mã commit đầy đủ (40 ký tự) chứa code P2; KHÔNG dùng "main"
ACCOUNT = "TV-B"           # "TV-A" | "TV-B"
STAGES = "train,infer,contrast,lr"
# Mỗi run một GPU riêng (gpu 0 hoặc 1). Pilot: CHỈ một run. Sau khi pilot đạt: tối đa 2 run song song (gpu 0 và 1).
# Run có thể ghi đè "stages" riêng; M-PT bắt buộc: {"data": "pt", "size": 0, "seed": 0, "gpu": 0, "stages": "infer,contrast,lr"}
RUNS = [
    {"data": "mnliB", "size": 16000, "seed": 11, "gpu": 0},
    # {"data": "mix", "size": 16000, "seed": 11, "gpu": 1},
]
RUN_SUFFIX = ""            # "-rep2" cho lần chạy lặp kiểm S0 (TV-A chạy lại M-16K-s11)
DATA_ROOT = "/kaggle/input"
OUT_ROOT = "/kaggle/working/out"
REPO_DIR = "/kaggle/working/repo"

# %%
import os, subprocess, time
os.environ["HF_HOME"] = "/tmp/hf"                 # cache HF trên /tmp (rộng), không làm đầy /kaggle/working
os.environ["TOKENIZERS_PARALLELISM"] = "false"
t_start = time.time()


def sh(cmd, check=True, env=None):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    out = []
    for line in p.stdout:
        out.append(line)
        print(line, end="", flush=True)
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError(f"Lệnh lỗi (mã {rc})")
    return "".join(out)


assert len({r["gpu"] for r in RUNS}) == len(RUNS), "mỗi run phải dùng một GPU khác nhau"
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Interactive":
    print("CẢNH BÁO: phiên tương tác — kết quả KHÔNG vào registry; nhớ tắt phiên sau khi xong để không mất quota GPU.")
if COMMIT == "main":
    print("CẢNH BÁO: COMMIT='main' -> chạy thử, kết quả KHÔNG vào registry.")
sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")

# %%
if not os.path.exists(REPO_DIR):
    sh(f"git clone --quiet https://github.com/{GITHUB_REPO}.git {REPO_DIR}")
REF = "origin/main" if COMMIT == "main" else COMMIT
sh(f"git -C {REPO_DIR} fetch --quiet origin && git -C {REPO_DIR} checkout --quiet {REF}")
GIT_COMMIT = sh(f"git -C {REPO_DIR} rev-parse HEAD").strip()
print("GIT_COMMIT =", GIT_COMMIT)
sh(f"pip install -q -r {REPO_DIR}/requirements.txt")
os.chdir(REPO_DIR)
sh('CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/')   # test chạy CPU; dừng ở đây nếu lỗi — chưa tốn GPU

# %%
# Tải sẵn bert-large-cased một lần (tránh 2 tiến trình cùng tải). Chỉ lấy safetensors + cấu hình + tokenizer.
from huggingface_hub import snapshot_download
snapshot_download("google-bert/bert-large-cased", allow_patterns=["*.json", "*.txt", "model.safetensors"])

# %%
procs = []
for r in RUNS:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(r["gpu"]), GIT_COMMIT=GIT_COMMIT,
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    cmd = (f"python scripts/run_p2.py --data-root {DATA_ROOT} --out-root {OUT_ROOT} --data {r['data']} "
           f"--size {r['size']} --seed {r['seed']} --stages {r.get('stages', STAGES)} --account {ACCOUNT}"
           + (f" --run-suffix {RUN_SUFFIX}" if RUN_SUFFIX else ""))
    os.makedirs(OUT_ROOT, exist_ok=True)
    logf = open(f"{OUT_ROOT}/log_{r['data']}_{r['size']}_s{r['seed']}{RUN_SUFFIX}.txt", "w")
    print("$", cmd, "| GPU", r["gpu"], flush=True)
    procs.append((r, subprocess.Popen(cmd, shell=True, text=True, stdout=logf, stderr=subprocess.STDOUT, env=env), logf))

# Theo dõi: in 5 dòng cuối log mỗi 5 phút cho tới khi mọi run xong
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(300)
    for r, p, lf in procs:
        lf.flush()
        tail = open(lf.name).read().splitlines()[-5:]
        print(f"--- {r['data']}-{r['size']}-s{r['seed']} ({'đang chạy' if p.poll() is None else 'xong'}):")
        print("\n".join(tail), flush=True)
codes = {f"{r['data']}-{r['size']}-s{r['seed']}": p.wait() for r, p, _ in procs}
print("mã thoát:", codes)

# %%
for r, p, lf in procs:
    print(f"===== LOG {lf.name} (60 dòng cuối) =====")
    print("\n".join(open(lf.name).read().splitlines()[-60:]))
for d in sorted(os.listdir(OUT_ROOT)):
    s = os.path.join(OUT_ROOT, d, "p2_summary.txt")
    if os.path.exists(s):
        print(f"\n===== {s} =====\n" + open(s).read())
sh(f"du -sh {OUT_ROOT}/*", check=False)
print(f"tổng thời gian notebook: {(time.time() - t_start) / 60:.1f} phút")
os.chdir("/kaggle/working")
sh(f"rm -rf {REPO_DIR}", check=False)       # không đóng gói code vào output (code đã có trên GitHub theo commit)
if any(codes.values()):
    raise RuntimeError(f"Có run lỗi: {codes} — xem log ở trên")
