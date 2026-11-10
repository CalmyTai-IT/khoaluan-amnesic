# %% [markdown]
# # Khoá luận — pilot erasure tại L_out (LEACE) — `src/khoaluan/erase/pilot.py`
# Notebook chỉ: kiểm input → lấy code đúng commit → cài gói → chạy test → chạy pilot → nén kết quả nhỏ `erase_small.zip`.
#
# Trước khi chạy: Settings → **Internet On**, **Accelerator GPU T4 x2** (chỉ dùng 1 GPU, ~1 phút/mô hình để trích MNLI-POOL;
# phần còn lại chạy CPU). Add Input: **khoaluan-data** và **khoaluan-p2-s1 — chọn Version 1** (chứa M-16K s111/s22/s33/s44).
# Chạy bằng **Save Version → Save & Run All (Commit)**.

# %%
# ===== CẤU HÌNH — chỉ sửa ô này =====
GITHUB_REPO = "CalmyTai-IT/khoaluan-amnesic"
COMMIT = "main"            # mã commit đầy đủ (40 ký tự) chứa bản vá erasure; KHÔNG dùng "main" khi chạy chính thức
RUN_IDS = ["bertL-mnliB-16k-s111", "bertL-mnliB-16k-s22", "bertL-mnliB-16k-s33"]   # pilot: 3 seed (mức MVP)
N_NULL = 1000              # số lần null N3 và N1 cho mỗi khái niệm (≥ 1000 để Holm trên 9 ô có thể đạt ý nghĩa)
OUT = "/kaggle/working/erase"
REPO_DIR = "/kaggle/working/repo"

# %%
# ===== KIỂM INPUT TRƯỚC KHI LÀM BẤT CỨ GÌ (vài giây) =====
import glob, os
data_hits = sorted({os.path.realpath(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/manifest.json", recursive=True)
                    if os.path.exists(os.path.join(os.path.dirname(p), "ft_uids_16000.txt"))})
print("khoaluan-data:", data_hits)
if len(data_hits) != 1:
    raise RuntimeError("Cần đúng 1 bộ khoaluan-data trong Input.")
missing = []
for rid in RUN_IDS:
    hits = glob.glob(f"/kaggle/input/**/{rid}/std_hans_eval_pooler.npy", recursive=True)
    ok = len(hits) == 1 and os.path.exists(os.path.join(os.path.dirname(hits[0]), "model", "model.safetensors"))
    print(f"{rid}: {'OK' if ok else 'THIẾU'} {hits}")
    if not ok:
        missing.append(rid)
if missing:
    raise RuntimeError(f"Thiếu cache/checkpoint cho {missing}. Kiểm Input khoaluan-p2-s1 đã chọn đúng Version 1 chưa.")

# %%
import subprocess, time, zipfile
os.environ["TOKENIZERS_PARALLELISM"] = "false"
t_start = time.time()


def sh(cmd, check=True, env=None):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError(f"Lệnh lỗi (mã {rc})")


if COMMIT == "main":
    print("CẢNH BÁO: COMMIT='main' -> chạy thử, kết quả KHÔNG được coi là chính thức.")
if not os.path.exists(REPO_DIR):
    sh(f"git clone --quiet https://github.com/{GITHUB_REPO}.git {REPO_DIR}")
sh(f"git -C {REPO_DIR} fetch --quiet origin && git -C {REPO_DIR} checkout --quiet {'origin/main' if COMMIT == 'main' else COMMIT}")
sh(f"git -C {REPO_DIR} rev-parse HEAD")
sh(f"pip install -q -r {REPO_DIR}/requirements.txt")
os.chdir(REPO_DIR)
sh('CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/')    # test chạy CPU; dừng ở đây nếu lỗi

# %%
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
sh(f"python scripts/run_erase_pilot.py --data-root /kaggle/input --runs-root /kaggle/input --out {OUT} "
   f"--n-null {N_NULL} --run-ids {' '.join(RUN_IDS)}", env=env)

# %%
zpath = "/kaggle/working/erase_small.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _d, files in os.walk(OUT):
        for fn in files:
            if not fn.endswith(("_pooler.npy", "_logits.npy")):          # bỏ cache lớn, giữ kết quả + null + tập fit
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, "/kaggle/working"))
print(open(os.path.join(OUT, "erase_summary.md")).read())
print(f"\nFile gửi Claude: {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB) | tổng {(time.time() - t_start) / 60:.1f} phút")
os.chdir("/kaggle/working")
sh(f"rm -rf {REPO_DIR}", check=False)
