import re
from collections import defaultdict

try:
    from .preprocess import clean_text, get_skeleton_name, normalize_address, extract_numbers, extract_tokens
except ImportError:
    from preprocess import clean_text, get_skeleton_name, normalize_address, extract_numbers, extract_tokens

def extract_meta_and_keys(name, addr):
    """
    Combined single-pass metadata and blocking keys generation.
    Eliminates redundant regex operations across millions of records.
    """
    c_name = clean_text(name)
    sk = get_skeleton_name(c_name)
    norm_addr = normalize_address(addr)
    addr_nums = extract_numbers(norm_addr)
    nums_set = set(addr_nums)
    tokens = extract_tokens(sk) if sk else extract_tokens(c_name)
    addr_tokens = extract_tokens(norm_addr)
    
    keys = []
    if sk:
        keys.append(f"sk:{sk}")
        compact = sk.replace(' ', '')
        if len(compact) >= 4:
            keys.append(f"cp:{compact}")
        toks = sk.split()
        if len(toks) >= 2:
            st = " ".join(sorted(list(set(toks))))
            keys.append(f"st:{st}")
            keys.append(f"f2:{toks[0]} {toks[1]}")
            if len(toks) >= 3:
                keys.append(f"fl:{toks[0]} {toks[-1]}")
        if toks and len(toks[0]) >= 3:
            keys.append(f"f1:{toks[0]}")
            
    if c_name and c_name != sk:
        keys.append(f"cn:{c_name}")
        c_compact = c_name.replace(' ', '')
        if len(c_compact) >= 4:
            keys.append(f"cp:{c_compact}")

    name_toks = sk.split() if sk else c_name.split()
    if name_toks and addr_nums:
        for nt in name_toks[:2]:
            for an in addr_nums[:2]:
                keys.append(f"n_num:{nt}_{an}")

    addr_toks_list = [t for t in norm_addr.split() if not t.isdigit() and len(t) > 2]
    if addr_nums and addr_toks_list:
        for an in addr_nums[:2]:
            for at in addr_toks_list[:3]:
                keys.append(f"num_ad:{an}_{at}")
                
    if len(addr_toks_list) >= 2:
        for j in range(min(2, len(addr_toks_list) - 1)):
            if len(addr_toks_list[j]) >= 4 and len(addr_toks_list[j+1]) >= 4:
                keys.append(f"ad_p:{addr_toks_list[j]}_{addr_toks_list[j+1]}")
                
    for ln in addr_nums[:2]:
        if len(ln) >= 5:
            keys.append(f"ln:{ln}")

    meta = (c_name, sk, norm_addr, nums_set, tokens, addr_tokens)
    return meta, keys

def generate_blocking_keys(name, addr):
    """Generate multi-pass blocking keys for a record."""
    _, keys = extract_meta_and_keys(name, addr)
    return keys

class MultiPassBlockingEngine:
    """
    High-performance inverted index blocking engine.
    Stores precomputed metadata alongside inverted index in a single pass.
    """
    def __init__(self, max_candidates_per_key=50, max_key_capacity=300):
        self.max_candidates_per_key = max_candidates_per_key
        self.max_key_capacity = max_key_capacity
        self.index = defaultdict(list)
        self.target_ids = []
        self.target_meta = []

    def build_index(self, target_ids, target_names, target_addrs):
        """Index all target records (Source 2 and Source 3) in a single pass."""
        self.target_ids = target_ids
        self.target_meta = []
        self.index = defaultdict(list)
        cap = self.max_key_capacity
        
        for idx in range(len(target_ids)):
            meta, keys = extract_meta_and_keys(target_names[idx], target_addrs[idx])
            self.target_meta.append(meta)
            for k in keys:
                bucket = self.index[k]
                if len(bucket) < cap:
                    bucket.append(idx)

    def get_candidate_indices(self, query_keys):
        """Retrieve candidate integer target indices for given keys."""
        cand_indices = set()
        cap = self.max_candidates_per_key
        
        for k in query_keys:
            if k in self.index:
                bucket = self.index[k]
                if len(bucket) <= cap:
                    cand_indices.update(bucket)
                else:
                    cand_indices.update(bucket[:cap])
                    
        return cand_indices

    def get_candidates(self, query_name, query_addr):
        """Retrieve candidate string entity IDs for a query name and address."""
        _, keys = extract_meta_and_keys(query_name, query_addr)
        cand_indices = self.get_candidate_indices(keys)
        return [self.target_ids[idx] for idx in cand_indices]
