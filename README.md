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
src/khoaluan/data/load.py       # P2: đọc đầu ra prepare_data v1.2 + kiểm vân tay lần A-06 chính thức
src/khoaluan/train/finetune.py  # P2/S2: fine-tune 5 epoch cố định (HF Trainer v5; chốt 1 GPU, batch 8, số bước); dừng sớm run thoái hoá
src/khoaluan/eval/{forward,metrics}.py  # P2: inference (logits theo uid, pooler = L_out), contrast pair, chỉ số HANS, Q4
src/khoaluan/probe/lr.py        # P2/S3: LR faithful (DLK) + robust; D-R1; chẩn đoán lưu fp16
src/khoaluan/probe/ccs.py       # P2/S3: CCS theo DLK (10 lần thử huấn luyện đồng thời)
src/khoaluan/p2.py              # một mô hình: train -> infer -> contrast -> lr -> ccs -> dr1
src/khoaluan/p2_queue.py        # hàng đợi một GPU; tự chạy lại seed+100, +200 khi thoái hoá (SPEC §3.3)
src/khoaluan/analysis/p2_report.py  # gộp kết quả + G2 theo quy tắc viết TRƯỚC khi có kết quả; D-R2, D-R3
src/khoaluan/erase/{sets,leace_fast,pilot}.py  # P3-P4 sớm: tập fit khái niệm, LEACE nhanh (khớp concept-erasure), pilot L_out
scripts/run_erase_pilot.py | notebooks/kaggle_erase_pilot.ipynb
configs/p2/plan.yaml            # phiên S1–S4 (21 mô hình = 20 fine-tune + M-PT)
scripts/run_p2.py | run_queue.py | analyze_p2.py
notebooks/kaggle_p2.ipynb       # notebook P2: chọn SESSION, 2 hàng đợi song song (bản .py cùng nội dung)
tests/                          # python -m pytest -q tests/   (test_p2.py: BERT 2 tầng, CPU, offline)
# Chưa viết: P3–P4: src/khoaluan/erase/, src/khoaluan/stats/
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
