"""
feats.py — Bộ trích đặc trưng heuristic (S1c).
Tương thích: SPEC_DULIEU_KHAINIEM.md v1.1, §3.

Định nghĩa (LO/SUB: token-level trên văn bản thô, lowercase, bỏ dấu câu):
  is_lo(p, h)            : mọi token của hypothesis đều có trong premise (lexical overlap, McCoy et al. 2019)
  is_sub(p, h)           : dãy token của hypothesis là dãy con LIÊN TIẾP của premise (subsequence)
  is_con_clause(pp, h)   : hypothesis trùng yield của một nút MỆNH ĐỀ (S/SINV/SQ/SBARQ) trong cây PTB
                           có nhãn của premise (constituent). So khớp sau CHUẨN HOÁ CHUỖI (§3.1 SPEC v1.1)
                           để không lệch giữa tokenization thô ("don't") và PTB ("do n't", "-LRB-").
  c4(...)                : {"NONE","LO_only","SUB_only","CON"} — phân loại lồng nhau CON ⊂ SUB ⊂ LO

Đã kiểm (05/11/2026): HANS-eval + HANS-train (60,000 cặp) khớp 100% trường `heuristic`;
unit test cho ca co rút / ngoặc PTB: tests/test_feats.py.
Changelog:
  v1.1 (05/11/2026): CON so khớp bằng chuỗi chuẩn hoá (sửa lỗi lệch tokenization PTB trên MNLI);
                     thêm parse_alignment(); parse lỗi -> trả False thay vì crash.
  v1.0 (04/11/2026): bản đầu.
"""
import re

_TOK = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*|n't")
_NONALNUM = re.compile(r"[^a-z0-9]")
NEG = re.compile(r"\b(not|no|never|nobody|none|nothing|nowhere|neither|nor|cannot)\b|n't", re.I)
CLAUSE_LABELS = {"S", "SINV", "SQ", "SBARQ"}
PTB_ESCAPES = {"-lrb-", "-rrb-", "-lsb-", "-rsb-", "-lcb-", "-rcb-"}


def toks(s: str):
    """Lowercase; bỏ dấu câu; giữ token có nháy (don't, n't, governor's)."""
    return _TOK.findall(s.lower().replace("’", "'"))


def norm_str(s: str) -> str:
    """Chuẩn hoá chuỗi để so khớp CON: lowercase, chỉ giữ [a-z0-9]."""
    return _NONALNUM.sub("", s.lower().replace("’", "'"))


def _norm_leaves(leaves) -> str:
    return norm_str(" ".join(w for w in leaves if w.lower() not in PTB_ESCAPES))


def is_lo(p: str, h: str) -> bool:
    th = toks(h)
    return len(th) > 0 and set(th) <= set(toks(p))


def _contiguous(tp, th) -> bool:
    n, m = len(tp), len(th)
    if m == 0 or m > n:
        return False
    return any(tp[i:i + m] == th for i in range(n - m + 1))


def is_sub(p: str, h: str) -> bool:
    return _contiguous(toks(p), toks(h))


def _parse_labeled(tokens, i, out):
    """Cây PTB có nhãn: '(' LABEL (con | từ) ')'. Ghi (label, yield_chuẩn_hoá) của mọi nút vào out."""
    if tokens[i] != "(":
        raise ValueError("parse không bắt đầu bằng '('")
    label = tokens[i + 1]
    i += 2
    leaves = []
    while tokens[i] != ")":
        if tokens[i] == "(":
            sub, i = _parse_labeled(tokens, i, out)
            leaves += sub
        else:
            leaves.append(tokens[i])
            i += 1
    out.append((label, _norm_leaves(leaves)))
    return leaves, i + 1


def labeled_yields(parse: str):
    """[(nhãn, yield chuẩn hoá)] cho mọi nút của cây PTB có nhãn (MNLI premise_parse / HANS sentence1_parse).
    KHÔNG dùng *_binary_parse (không có nhãn). Parse lỗi -> [] (không crash)."""
    if not isinstance(parse, str) or not parse.strip():
        return []
    tokens = re.findall(r"\(|\)|[^\s()]+", parse)
    out = []
    try:
        _parse_labeled(tokens, 0, out)
    except (IndexError, ValueError):
        return []
    return out


def parse_alignment(p: str, p_parse: str) -> bool:
    """True nếu yield của nút gốc (chuẩn hoá) trùng premise thô (chuẩn hoá) — kiểm cây parse khớp văn bản."""
    ys = labeled_yields(p_parse)
    return bool(ys) and ys[-1][1] == norm_str(p)


def is_con_clause(p_parse: str, h: str) -> bool:
    nh = norm_str(h)
    return len(nh) > 0 and any(lab in CLAUSE_LABELS and y == nh for lab, y in labeled_yields(p_parse))


def is_con_anynode(p_parse: str, h: str) -> bool:
    """Biến thể McCoy (const_finder.py): yield của BẤT KỲ nút nào. Chỉ để báo cáo so sánh."""
    nh = norm_str(h)
    return len(nh) > 0 and any(y == nh for _, y in labeled_yields(p_parse))


def c4(p: str, h: str, p_parse: str) -> str:
    if not is_lo(p, h):
        return "NONE"
    if not is_sub(p, h):
        return "LO_only"
    return "CON" if is_con_clause(p_parse, h) else "SUB_only"


def covariates(p: str, h: str) -> dict:
    tp, th = toks(p), toks(h)
    neg_p, neg_h = bool(NEG.search(p)), bool(NEG.search(h))
    sp = set(tp)
    return {
        "hyp_len": len(th),
        "prem_len": len(tp),
        "neg_p": int(neg_p),
        "neg_h": int(neg_h),
        "neg_pattern": f"P{int(neg_p)}H{int(neg_h)}",
        "overlap_rate": (sum(t in sp for t in th) / len(th)) if th else 0.0,
    }


def hyp_len_bin(n: int) -> str:
    return "<=5" if n <= 5 else "6-8" if n <= 8 else "9-12" if n <= 12 else ">=13"


def hans_subject_number(h_parse: str):
    """c_CTRL_H: số của danh từ chủ ngữ hypothesis HANS ('NN'/'NNS'); None nếu không xác định."""
    m = re.search(r"\(NP (?:\(DT \w+\) )?(?:\(JJ \w+\) )?\((NNS?) ", h_parse)
    return m.group(1) if m else None
