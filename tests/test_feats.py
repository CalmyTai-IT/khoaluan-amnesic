"""Unit test cho feats.py. Chạy: python -m pytest -q tests/  (từ thư mục gốc repo)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from khoaluan.data.feats import (c4, is_con_anynode, is_con_clause, is_lo, is_sub,  # noqa: E402
                                 labeled_yields, parse_alignment)

IF_PARSE = ("(ROOT (S (SBAR (IN If) (S (NP (DT the) (NNS accountants)) (VP (VBD knew)))) (, ,) "
            "(NP (PRP they)) (VP (MD would) (RB n't) (VP (VB have) (VP (VBN resigned)))) (. .)))")
P_IF = "If the accountants knew , they would n't have resigned ."


def test_hierarchy_hans_style():
    assert c4(P_IF, "The accountants knew .", IF_PARSE) == "CON"
    assert is_con_clause(IF_PARSE, "They would n't have resigned .") is False  # VP, không phải S
    assert c4("The president advised the doctor .", "The doctor advised the president .",
              "(ROOT (S (NP (DT The) (NN president)) (VP (VBD advised) (NP (DT the) (NN doctor))) (. .)))") == "LO_only"


def test_contraction_raw_vs_ptb():
    # MNLI: văn bản thô có co rút, cây PTB tách "would n't"
    parse = ("(ROOT (S (SBAR (IN If) (S (NP (PRP they)) (VP (VBD knew)))) (, ,) (NP (PRP they)) "
             "(VP (MD would) (RB n't) (VP (VB have) (VP (VBN left)))) (. .)))")
    prem = "If they knew, they wouldn't have left."
    assert parse_alignment(prem, parse)
    assert is_con_clause(parse, "they knew")
    assert is_con_anynode(parse, "wouldn't have left")
    assert c4(prem, "They knew.", parse) == "CON"


def test_ptb_escape_brackets():
    parse = ("(ROOT (S (NP (NNP Bob) (PRN (-LRB- -LRB-) (NNP Robert) (-RRB- -RRB-))) "
             "(VP (VBD left)) (. .)))")
    prem = "Bob (Robert) left."
    assert parse_alignment(prem, parse)
    assert is_con_clause(parse, "Bob (Robert) left")


def test_malformed_parse_does_not_crash():
    assert labeled_yields("") == []
    assert labeled_yields("(ROOT (S (NP") == []
    assert c4("A b c", "a b", "(ROOT (S (NP") == "SUB_only"


def test_lo_sub_basic():
    assert is_lo("The cat sat on the mat.", "the mat sat")
    assert not is_sub("The cat sat on the mat.", "the mat sat")
    assert is_sub("The cat sat on the mat.", "sat on the mat")
    assert not is_sub("the actor left", "the act")  # so khớp token, không phải chuỗi ký tự
