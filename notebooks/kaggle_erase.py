# %% [markdown]
# # Khoá luận — erasure pilot 2: L_out (sửa sau pilot 1) + [CLS] tầng 20/22/24 — `src/khoaluan/erase/`
# Notebook chỉ: kiểm input → lấy code đúng commit → cài gói → chạy test → chạy 2 tiến trình song song (mỗi GPU một
# tiến trình) → gộp kết quả → nén kết quả nhỏ `erase2_small.zip`.
#
# Trước khi chạy: Settings → **Internet On**, **Accelerator GPU T4 x2**. Add Input: **khoaluan-data** và
# **khoaluan-p2-s1 — chọn Version 1** (chứa M-16K s111/s22/s33/s44). Chạy bằng **Save Version → Save & Run All (Commit)**.

# %%
# ===== CẤU HÌNH — chỉ sửa ô này =====
GITHUB_REPO = "CalmyTai-IT/khoaluan-amnesic"
COMMIT = "main"            # mã commit đầy đủ (40 ký tự) chứa bản vá pilot 2; KHÔNG dùng "main" khi chạy chính thức
# Mỗi GPU một hàng đợi (mô hình, stage). s33 được tách: tầng giữa (GPU nặng) ở GPU 0, L_out (gần như chỉ CPU) ở GPU 1,
# để hai hàng đợi xong gần cùng lúc. Ước lượng ~70–80 phút cho cả notebook [PHỎNG ĐOÁN].
QUEUES = {0: [("bertL-mnliB-16k-s111", "lout,lmid"), ("bertL-mnliB-16k-s33", "lmid")],
          1: [("bertL-mnliB-16k-s22", "lout,lmid"), ("bertL-mnliB-16k-s33", "lout")]}
N_NULL = 1000              # null N3 và N1 tại L_out (giữ như pilot 1)
OUT = "/kaggle/working/erase2"
REPO_DIR = "/kaggle/working/repo"

# %%
# ===== KIỂM INPUT TRƯỚC KHI LÀM BẤT CỨ GÌ (vài giây) =====
import glob, json, os
data_hits = sorted({os.path.realpath(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/manifest.json", recursive=True)
                    if os.path.exists(os.path.join(os.path.dirname(p), "ft_uids_16000.txt"))})
print("khoaluan-data:", data_hits)
if len(data_hits) != 1:
    raise RuntimeError("Cần đúng 1 bộ khoaluan-data trong Input.")
problems = []
for rid in sorted({r for q in QUEUES.values() for r, _ in q}):
    hits = glob.glob(f"/kaggle/input/**/{rid}/std_hans_eval_pooler.npy", recursive=True)
    if len(hits) != 1:
        problems.append(f"{rid}: tìm thấy {len(hits)} thư mục cache")
        continue
    d = os.path.dirname(hits[0])
    need = ["model/model.safetensors", "model/config.json", "std_layers.json", "std_hans_eval_cls.npy",
            "std_hans_train_B_cls.npy", "std_dev_m_cls.npy", "preds_hans_eval.parquet", "preds_dev_m.parquet"]
    miss = [f for f in need if not os.path.exists(os.path.join(d, f))]
    layers = json.load(open(os.path.join(d, "std_layers.json")))["cls_layers"] if "std_layers.json" not in miss else []
    if miss or not {20, 22, 24} <= set(layers):
        problems.append(f"{rid}: thiếu {miss}; tầng CLS trong cache {layers}")
    print(f"{rid}: {'OK' if not miss else 'THIẾU'} {d}")
if problems:
    raise RuntimeError("Input chưa đủ: " + " | ".join(problems) + " — kiểm khoaluan-p2-s1 đã chọn đúng Version 1 chưa.")

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
gpus = subprocess.run("nvidia-smi --query-gpu=index,name --format=csv,noheader", shell=True, capture_output=True,
                      text=True).stdout.strip().splitlines()
print("GPU:", gpus)
if len(gpus) < 2:
    raise RuntimeError(f"Cần accelerator GPU T4 x2, đang thấy: {gpus}")
if not os.path.exists(REPO_DIR):
    sh(f"git clone --quiet https://github.com/{GITHUB_REPO}.git {REPO_DIR}")
sh(f"git -C {REPO_DIR} fetch --quiet origin && git -C {REPO_DIR} checkout --quiet {'origin/main' if COMMIT == 'main' else COMMIT}")
sh(f"git -C {REPO_DIR} rev-parse HEAD")
sh(f"pip install -q -r {REPO_DIR}/requirements.txt")
os.chdir(REPO_DIR)
sh('CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/')    # test chạy CPU (~2–3 phút); dừng ở đây nếu lỗi

# %%
# Hai tiến trình song song, mỗi GPU một tiến trình; mỗi tiến trình 2 luồng CPU (máy Kaggle có 4 lõi)
os.makedirs(OUT, exist_ok=True)
procs = []
for g, jobs in QUEUES.items():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2")
    parts = [f"python scripts/run_erase.py --data-root /kaggle/input --runs-root /kaggle/input --out {OUT}/q{g} "
             f"--stages {st} --n-null {N_NULL} --run-ids {rid}; rc=$((rc | $?))" for rid, st in jobs]
    cmd = "rc=0; " + "; ".join(parts) + "; exit $rc"            # việc sau vẫn chạy nếu việc trước lỗi
    logf = open(f"{OUT}/log_q{g}.txt", "w")
    print("$", cmd, "| GPU", g, flush=True)
    procs.append((g, subprocess.Popen(cmd, shell=True, text=True, stdout=logf, stderr=subprocess.STDOUT, env=env), logf))
last = time.time()
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(20)
    if time.time() - last >= 300:
        last = time.time()
        for g, p, lf in procs:
            lf.flush()
            tail = [l for l in open(lf.name).read().splitlines() if l.startswith("[")][-4:]
            print(f"--- q{g} ({'đang chạy' if p.poll() is None else 'xong'}) | {(time.time() - t_start) / 60:.0f} phút:")
            print("\n".join(tail), flush=True)
codes = {f"q{g}": p.wait() for g, p, _ in procs}
print("mã thoát:", codes)
for g, _, _ in procs:
    print(f"\n===== log q{g} (40 dòng cuối) =====")
    print("\n".join(open(f"{OUT}/log_q{g}.txt").read().splitlines()[-40:]))

# %%
# Gộp hai tiến trình, nén kết quả nhỏ (bỏ cache biểu diễn lớn)
ok = [f"{OUT}/q{g}" for g, _, _ in procs if os.path.exists(f"{OUT}/q{g}/sets/sets_report.json")]
if ok:
    sh(f"python scripts/summarize_erase.py --in {' '.join(ok)} --out {OUT}/all --n-null {N_NULL}", check=False)
zpath = "/kaggle/working/erase2_small.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _d, files in os.walk(OUT):
        for fn in files:
            if not (fn.startswith("pool_") and fn.endswith(".npy")):
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, "/kaggle/working"))
print(f"\nFile gửi Claude: {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB) | tổng {(time.time() - t_start) / 60:.1f} phút")
os.chdir("/kaggle/working")
sh(f"rm -rf {REPO_DIR}", check=False)
if any(codes.values()):
    raise RuntimeError(f"Có tiến trình dừng vì lỗi: {codes} — xem {OUT}/log_q*.txt (đã có trong file zip)")
