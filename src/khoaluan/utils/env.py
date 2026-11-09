"""Thu thập thông tin môi trường (S0 — SPEC_REPRODUCE_VDL §2.3). Không in bí mật (token, secret)."""
import importlib
import os
import platform
import shutil
import subprocess
import urllib.request

PKGS = ["torch", "transformers", "datasets", "accelerate", "sklearn", "numpy", "pandas", "pyarrow",
        "scipy", "yaml", "concept_erasure"]
URLS = {
    "huggingface": "https://huggingface.co/api/models/google-bert/bert-large-cased",
    "github_raw": "https://raw.githubusercontent.com/tommccoy1/hans/7299f6f657089ce06a0f98e7e81f8d0f5b7741ce/README.md",
    "pypi": "https://pypi.org/simple/concept-erasure/",
}


def _ver(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "installed")
    except Exception:
        return None


def _disk(path):
    try:
        u = shutil.disk_usage(path)
        return {"total_gb": round(u.total / 1e9, 1), "free_gb": round(u.free / 1e9, 1)}
    except Exception:
        return None


def _mem_gb():
    try:
        with open("/proc/meminfo") as f:
            kv = dict(line.split(":", 1) for line in f)
        return round(int(kv["MemTotal"].split()[0]) / 1e6, 1)
    except Exception:
        return None


def _net(url):
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.status
    except Exception as e:
        return f"FAIL: {type(e).__name__}"


def _git_commit(repo_dir="."):
    try:
        return subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def collect_env(check_net=True):
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _mem_gb(),
        "disk_kaggle_working": _disk("/kaggle/working"),
        "disk_tmp": _disk("/tmp"),
        "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),  # "Interactive" | "Batch"
        "git_commit": _git_commit(),
        "packages": {p: _ver(p) for p in PKGS},
    }
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        env["gpus"] = [{"name": torch.cuda.get_device_name(i),
                        "mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
                       for i in range(torch.cuda.device_count())]
    except Exception as e:
        env["torch_error"] = repr(e)
    if check_net:
        env["internet"] = {k: _net(u) for k, u in URLS.items()}
    return env


def pip_freeze():
    try:
        return subprocess.check_output(["pip", "freeze"], text=True)
    except Exception as e:
        return f"# pip freeze lỗi: {e}\n"
