import numpy as np
from rapidfuzz import fuzz

try:
    from .preprocess import clean_text, get_skeleton_name, normalize_address, extract_numbers, extract_tokens
except ImportError:
    from preprocess import clean_text, get_skeleton_name, normalize_address, extract_numbers, extract_tokens

FEATURE_NAMES = [
    'name_ratio',
    'name_partial',
    'name_token_sort',
    'name_token_set',
    'name_wratio',
    'name_sk_ratio',
    'name_sk_token_sort',
    'name_exact',
    'name_sk_exact',
    'name_len_diff',
    'name_len_ratio',
    'name_token_jaccard',
    'addr_token_set',
    'addr_token_sort',
    'addr_ratio',
    'addr_partial',
    'addr_token_jaccard',
    'addr_num_match',
    'addr_num_jaccard',
    'addr_is_null',
    'cand_is_s2',
    'joint_token_set',
    'name_sub_match'
]

def extract_record_metadata(name, addr):
    """Precompute reusable representations for a record as a lightweight tuple."""
    c_name = clean_text(name)
    sk_name = get_skeleton_name(c_name)
    norm_addr = normalize_address(addr)
    nums = set(extract_numbers(norm_addr))
    tokens = extract_tokens(sk_name) if sk_name else extract_tokens(c_name)
    addr_tokens = extract_tokens(norm_addr)
    return (c_name, sk_name, norm_addr, nums, tokens, addr_tokens)

def extract_pair_features(s1_meta, cand_meta, cand_id):
    """Extract a 23-dimensional feature vector for a candidate pair."""
    if isinstance(s1_meta, tuple):
        s1_name, s1_sk, s1_addr, s1_nums, s1_toks, s1_atok = s1_meta
    else:
        s1_name, s1_sk, s1_addr = s1_meta['name'], s1_meta['sk'], s1_meta['addr']
        s1_nums, s1_toks, s1_atok = s1_meta['nums'], s1_meta['tokens'], s1_meta['addr_tokens']

    if isinstance(cand_meta, tuple):
        cand_name, cand_sk, cand_addr, cand_nums, cand_toks, cand_atok = cand_meta
    else:
        cand_name, cand_sk, cand_addr = cand_meta['name'], cand_meta['sk'], cand_meta['addr']
        cand_nums, cand_toks, cand_atok = cand_meta['nums'], cand_meta['tokens'], cand_meta['addr_tokens']

    # Name similarities
    name_ratio = fuzz.ratio(s1_name, cand_name)
    name_partial = fuzz.partial_ratio(s1_name, cand_name)
    name_token_sort = fuzz.token_sort_ratio(s1_name, cand_name)
    name_token_set = fuzz.token_set_ratio(s1_name, cand_name)
    name_wratio = fuzz.WRatio(s1_name, cand_name)

    name_sk_ratio = fuzz.ratio(s1_sk, cand_sk) if (s1_sk and cand_sk) else name_ratio
    name_sk_token_sort = fuzz.token_sort_ratio(s1_sk, cand_sk) if (s1_sk and cand_sk) else name_token_sort

    name_exact = 1.0 if s1_name and s1_name == cand_name else 0.0
    name_sk_exact = 1.0 if s1_sk and s1_sk == cand_sk else 0.0

    len1, len2 = len(s1_name), len(cand_name)
    name_len_diff = float(abs(len1 - len2))
    name_len_ratio = float(min(len1, len2) / max(len1, len2, 1))

    # Token Jaccard
    union_toks = len(s1_toks | cand_toks)
    name_token_jaccard = float(len(s1_toks & cand_toks) / union_toks) if union_toks > 0 else 0.0

    # Address similarities
    if s1_addr and cand_addr:
        addr_token_set = fuzz.token_set_ratio(s1_addr, cand_addr)
        addr_token_sort = fuzz.token_sort_ratio(s1_addr, cand_addr)
        addr_ratio = fuzz.ratio(s1_addr, cand_addr)
        addr_partial = fuzz.partial_ratio(s1_addr, cand_addr)
        union_atok = len(s1_atok | cand_atok)
        addr_token_jaccard = float(len(s1_atok & cand_atok) / union_atok) if union_atok > 0 else 0.0
        addr_is_null = 0.0
    else:
        addr_token_set = 0.0
        addr_token_sort = 0.0
        addr_ratio = 0.0
        addr_partial = 0.0
        addr_token_jaccard = 0.0
        addr_is_null = 1.0

    # Number overlap in address
    if s1_nums and cand_nums:
        num_inter = len(s1_nums & cand_nums)
        addr_num_match = 1.0 if num_inter > 0 else 0.0
        addr_num_jaccard = float(num_inter / len(s1_nums | cand_nums))
    else:
        addr_num_match = 0.0
        addr_num_jaccard = 0.0

    cand_is_s2 = 1.0 if str(cand_id).startswith('S2-') else 0.0
    
    # Joint string similarity
    joint1 = f"{s1_name} {s1_addr}".strip()
    joint2 = f"{cand_name} {cand_addr}".strip()
    joint_token_set = fuzz.token_set_ratio(joint1, joint2)

    # Substring match
    name_sub_match = 1.0 if (s1_sk and cand_sk and (s1_sk in cand_sk or cand_sk in s1_sk)) else 0.0

    return [
        name_ratio,
        name_partial,
        name_token_sort,
        name_token_set,
        name_wratio,
        name_sk_ratio,
        name_sk_token_sort,
        name_exact,
        name_sk_exact,
        name_len_diff,
        name_len_ratio,
        name_token_jaccard,
        addr_token_set,
        addr_token_sort,
        addr_ratio,
        addr_partial,
        addr_token_jaccard,
        addr_num_match,
        addr_num_jaccard,
        addr_is_null,
        cand_is_s2,
        joint_token_set,
        name_sub_match
    ]
