"""Unit test cho logic khử trùng lặp / uid của prepare_data.py (v1.2)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from khoaluan.data.prepare_data import dedupe_mnli, pairid_dup_report  # noqa: E402


def _df(rows):
    df = pd.DataFrame(rows, columns=["pairID", "premise", "hypothesis", "y"])
    df["row_idx"] = range(len(df))
    return df


def test_dedupe_conflict_and_exact():
    df = _df([
        ("1e", "A cat sat.", "A cat sat", 0),      # trùng chính xác với dòng 1 (khác dấu câu) -> giữ 1
        ("1e", "A cat sat", "a cat sat.", 0),
        ("2c", "Dogs bark.", "Dogs do not bark.", 1),  # mâu thuẫn nhãn với dòng 3 -> bỏ cả hai
        ("3e", "Dogs bark.", "Dogs do not bark.", 0),
        ("4e", "Birds fly.", "Birds fly.", 0),
    ])
    out, st = dedupe_mnli(df)
    assert st == {"rows_in": 5, "conflict_rows_dropped": 2, "conflict_groups": 1,
                  "exact_dup_rows_dropped": 1, "rows_out": 2}
    assert list(out.row_idx) == [0, 4]


def test_pairid_report():
    df = _df([("1e", "a", "b", 0), ("1e", "a", "b", 0), ("2e", "c", "d", 0), ("2e", "x", "y", 0)])
    r = pairid_dup_report(df)
    assert r["dup_groups"] == 2 and r["groups_same_text"] == 1 and r["groups_diff_text"] == 1
