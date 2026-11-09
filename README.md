# khoaluan-amnesic

Code khoá luận: *Kiểm định nhân quả ở mức biểu diễn đối với các heuristic cú pháp trong NLI bằng Amnesic Probing* (HCMUS).
Đặc tả nằm trong knowledge base của Project (root `KHOALUAN_KE_HOACH_THEO_DOI.md`, `SPEC_*.md`) — **code phải theo SPEC, không ngược lại**.

## Cấu trúc

```
configs/finetune/base.yaml      # siêu tham số fine-tune (SPEC_REPRODUCE_VDL §3.2)
src/khoaluan/data/feats.py      # bộ trích LO/SUB/CON (SPEC_DULIEU_KHAINIEM §3)
src/khoaluan/data/prepare_data.py   # dựng dữ liệu + bảng 2x2 (A-06)
src/khoaluan/utils/{env,seed,registry}.py
scripts/env_check.py            # A-04: môi trường + benchmark tốc độ
scripts/prepare_data.py         # A-06
scripts/merge_registry.py       # gộp run_record.json -> results/registry.csv
notebooks/kaggle_template.ipynb # notebook mẫu Kaggle (A-09); bản .py cùng nội dung để review diff
results/registry.csv            # registry kết quả — nguồn duy nhất cho số liệu trong luận văn
tests/                          # python -m pytest -q tests/
# Sẽ viết ở phiên tái lập (P2): src/khoaluan/train/, src/khoaluan/eval/, src/khoaluan/probe/ (LR/CCS)
# Sẽ viết ở P3–P4: src/khoaluan/erase/, src/khoaluan/stats/
```

## Quy tắc

1. Notebook chỉ: clone đúng commit → cài gói thiếu → gọi script → ghi `run_record.json`. Không viết logic thí nghiệm trong notebook.
2. Chạy chính thức: `COMMIT` = mã commit đầy đủ, **Save & Run All (Commit)**. Chạy thử: được dùng `main`, kết quả không vào registry.
3. Mọi run chính thức → 1 dòng `results/registry.csv` (qua `scripts/merge_registry.py`), người còn lại review commit đó.
4. Không commit dữ liệu, checkpoint, representation (`.gitignore`). Đầu ra lớn → Kaggle Dataset có phiên bản.
5. Không sửa `configs/finetune/base.yaml` khi SPEC chưa đổi.

## Chạy cục bộ

```bash
pip install -r requirements.txt pandas pyarrow numpy
python -m pytest -q tests/
python scripts/prepare_data.py --out ./data --skip-mnli      # chỉ phần HANS, không cần HF
```
