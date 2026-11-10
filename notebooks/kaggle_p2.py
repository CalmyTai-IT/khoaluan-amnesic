# %% [markdown]
# # Khoá luận — notebook P2 (fine-tune + tái lập V&dL) — SPEC_REPRODUCE_VDL v1.1, kế hoạch `configs/p2/plan.yaml`
# Notebook **chỉ**: kiểm dữ liệu → lấy code đúng commit → cài gói thiếu → chạy unit test → chạy 2 hàng đợi mô hình
# (mỗi GPU một hàng đợi) → in tóm tắt → nén các file kết quả nhỏ thành `p2_small_<SESSION>.zip`.
#
# Trước khi chạy: Settings → **Internet On**, **Accelerator GPU T4 x2**; Add Input → Dataset **khoaluan-data**.
# Chỉ sửa ô CẤU HÌNH (SESSION = S1, S2, S3, S4 lần lượt). Chạy bằng **Save Version → Save & Run All (Commit)**.

# %%
# ===== CẤU HÌNH — chỉ sửa ô này =====
GITHUB_REPO = "CalmyTai-IT/khoaluan-amnesic"
COMMIT = "main"            # mã commit đầy đủ (40 ký tự) chứa bản vá P2 v3; KHÔNG dùng "main" khi chạy chính thức
SESSION = "S1"             # "S1" | "S2" | "S3" | "S4" — xem configs/p2/plan.yaml
ACCOUNT = "TV-B"           # "TV-A" | "TV-B"
OUT_ROOT = "/kaggle/working/out"
REPO_DIR = "/kaggle/working/repo"

# %%
# ===== KIỂM DỮ LIỆU TRƯỚC KHI LÀM BẤT CỨ GÌ (vài giây) =====
import glob, os
hits = sorted({os.path.realpath(os.path.dirname(p))
               for p in glob.glob("/kaggle/input/**/manifest.json", recursive=True)
               if os.path.exists(os.path.join(os.path.dirname(p), "ft_uids_16000.txt"))})
print("Thư mục dữ liệu tìm thấy:", hits)
if len(hits) != 1:
    os.system("ls -la /kaggle/input; ls -la /kaggle/input/* 2>/dev/null | head -40")
    raise RuntimeError("Cần đúng 1 bộ khoaluan-data. Panel phải → Input → Add Input → khoaluan-data, rồi chạy lại.")
DATA_ROOT = hits[0]

# %%
import subprocess, time, json, zipfile
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


if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Interactive":
    print("CẢNH BÁO: phiên tương tác — kết quả KHÔNG vào registry; nhớ tắt phiên sau khi xong để không mất quota GPU.")
if COMMIT == "main":
    print("CẢNH BÁO: COMMIT='main' -> chạy thử, kết quả KHÔNG vào registry.")
gpus = sh("nvidia-smi --query-gpu=index,name --format=csv,noheader").strip().splitlines()
if len(gpus) < 2:
    raise RuntimeError(f"Cần accelerator GPU T4 x2, đang thấy: {gpus}")

# %%
if not os.path.exists(REPO_DIR):
    sh(f"git clone --quiet https://github.com/{GITHUB_REPO}.git {REPO_DIR}")
REF = "origin/main" if COMMIT == "main" else COMMIT
sh(f"git -C {REPO_DIR} fetch --quiet origin && git -C {REPO_DIR} checkout --quiet {REF}")
GIT_COMMIT = sh(f"git -C {REPO_DIR} rev-parse HEAD").strip()
print("GIT_COMMIT =", GIT_COMMIT)
sh(f"pip install -q -r {REPO_DIR}/requirements.txt")
os.chdir(REPO_DIR)
sh('CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/')   # test chạy CPU; dừng ở đây nếu lỗi — chưa tốn GPU đáng kể
import yaml
PLAN = yaml.safe_load(open("configs/p2/plan.yaml"))[SESSION]
print(f"Phiên {SESSION}: gpu0 = {PLAN['gpu0']} | gpu1 = {PLAN['gpu1']}")

# %%
# Tải sẵn bert-large-cased một lần (tránh 2 tiến trình cùng tải). Chỉ lấy safetensors + cấu hình + tokenizer.
from huggingface_hub import snapshot_download
snapshot_download("google-bert/bert-large-cased", allow_patterns=["*.json", "*.txt", "model.safetensors"])

# %%
os.makedirs(OUT_ROOT, exist_ok=True)
procs = []
for g in (0, 1):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), GIT_COMMIT=GIT_COMMIT, CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")   # 2 hàng đợi chia 4 lõi CPU
    cmd = (f"python scripts/run_queue.py --data-root {DATA_ROOT} --out-root {OUT_ROOT} --account {ACCOUNT} "
           f"--report {OUT_ROOT}/queue_report_{SESSION}_gpu{g}.json --specs {' '.join(PLAN[f'gpu{g}'])}")
    logf = open(f"{OUT_ROOT}/log_{SESSION}_gpu{g}.txt", "w")
    print("$", cmd, "| GPU", g, flush=True)
    procs.append((g, subprocess.Popen(cmd, shell=True, text=True, stdout=logf, stderr=subprocess.STDOUT, env=env), logf))

# Theo dõi: kiểm mỗi 20 giây, in 5 dòng cuối log mỗi 5 phút
last = time.time()
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(20)
    if time.time() - last >= 300:
        last = time.time()
        for g, p, lf in procs:
            lf.flush()
            tail = open(lf.name).read().splitlines()[-5:]
            print(f"--- gpu{g} ({'đang chạy' if p.poll() is None else 'xong'}) | {(time.time() - t_start) / 60:.0f} phút:")
            print("\n".join(tail), flush=True)
codes = {f"gpu{g}": p.wait() for g, p, _ in procs}
print("mã thoát hàng đợi:", codes)

# %%
# Tóm tắt + nén file kết quả nhỏ (gửi Claude file zip này)
SMALL = {"p2_summary.txt", "p2_metrics.json", "run_record.json", "lr_summary.csv", "lr_raw.csv", "ccs_summary.csv",
         "ccs_raw.csv", "dr1_summary.csv", "dr1_raw.csv", "hans_subcase.csv", "cp_fp16_diag.csv", "train_log.json",
         "cp_layers.json", "std_layers.json", "lr_aux_summary.csv", "ccs_aux_summary.csv"}
zpath = f"/kaggle/working/p2_small_{SESSION}.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _dirs, files in os.walk(OUT_ROOT):
        for fn in files:
            if fn in SMALL or fn.startswith(("queue_report_", "log_")):
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, "/kaggle/working"))
for g in (0, 1):
    rp = f"{OUT_ROOT}/queue_report_{SESSION}_gpu{g}.json"
    if os.path.exists(rp):
        rep = json.load(open(rp))
        print(f"\n===== gpu{g}: " + "; ".join(f"{r['run_id']} thoái hoá={r['degenerate']} ({r['minutes']} phút)"
                                            for r in rep["runs"]) + (f" | DỪNG VÌ LỖI: {rep['stopped_on_error']}"
                                                                     if rep["stopped_on_error"] else ""))
for d in sorted(os.listdir(OUT_ROOT)):
    s = os.path.join(OUT_ROOT, d, "p2_summary.txt")
    if os.path.exists(s):
        print(f"\n===== {d} =====\n" + "".join(open(s).readlines()[:12]))
sh(f"du -sh {OUT_ROOT}/* | sort -h | tail -20; du -sh /kaggle/working", check=False)
print(f"File gửi Claude: {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB)")
print(f"Tổng thời gian notebook: {(time.time() - t_start) / 60:.1f} phút")
os.chdir("/kaggle/working")
sh(f"rm -rf {REPO_DIR}", check=False)       # không đóng gói code vào output (code đã có trên GitHub theo commit)
if any(codes.values()):
    raise RuntimeError(f"Có hàng đợi dừng vì lỗi: {codes} — xem log_{SESSION}_gpu*.txt")
