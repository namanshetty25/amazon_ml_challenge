"""
GPU Dense Embeddings + PyTorch Matrix Multiplication Pipeline
For Business Entity Resolution Challenge.
"""

import os
import gc
import time
from collections import defaultdict
import polars as pl
import torch
from tqdm import tqdm

def format_record(name, addr):
    """Format business name and address into a single text representation."""
    n = str(name or "").strip()
    a = str(addr or "").strip()
    if a and a.lower() != "none":
        return f"{n} | {a}"
    return n

def run_gpu_entity_resolution(
    dataset_dir="dataset",
    output_dir="output",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    candidate_threshold=0.50,
    match_threshold=0.84,
    top_k_candidates=15
):
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    os.makedirs(output_dir, exist_ok=True)
    matching_out = os.path.join(output_dir, "matching_results.tsv")
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"Loading transformer model: {model_name}...")
    embed_model = SentenceTransformer(model_name, device=device)

    print("Loading test data...")
    s1_all = pl.read_csv(os.path.join(dataset_dir, "test", "test_source1.tsv"), separator="\t")
    s2_all = pl.read_csv(os.path.join(dataset_dir, "test", "test_source2.tsv"), separator="\t")
    s3_all = pl.read_csv(os.path.join(dataset_dir, "test", "test_source3.tsv"), separator="\t")

    all_s1_ids = s1_all["entity_id"].to_list()
    all_s1_countries = s1_all["country"].to_list()

    country_to_indices = defaultdict(list)
    for idx, c in enumerate(all_s1_countries):
        country_to_indices[c].append(idx)

    results_matching = {}
    results_candidates = {}

    countries = sorted(list(country_to_indices.keys()), key=lambda c: len(country_to_indices[c]))
    print(f"Jurisdiction execution order: {countries}")

    for country in countries:
        s1_indices = country_to_indices[country]
        print(f"\n================ Processing {country} ({len(s1_indices):,} entities) ================")

        s1_c = s1_all[s1_indices]
        s2_c = s2_all.filter(pl.col("country") == country)
        s3_c = s3_all.filter(pl.col("country") == country)

        target_ids = s2_c["entity_id"].to_list() + s3_c["entity_id"].to_list()
        target_texts = [
            format_record(n, a) for n, a in zip(
                s2_c["business_name"].to_list() + s3_c["business_name"].to_list(),
                s2_c["business_address"].to_list() + s3_c["business_address"].to_list()
            )
        ]
        del s2_c, s3_c
        gc.collect()

        # Step 1: Encode Targets into FP16 embeddings
        print(f"Encoding {len(target_texts):,} target records on GPU...")
        with torch.inference_mode():
            target_embs = embed_model.encode(
                target_texts,
                batch_size=1024,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=True,
                device=device
            ).half()

        # Move target embeddings to CPU pinned memory to conserve GPU VRAM
        target_embs_cpu = target_embs.cpu()
        del target_embs
        if device == "cuda":
            torch.cuda.empty_cache()

        # Step 2: Encode S1 Queries
        s1_ids = s1_c["entity_id"].to_list()
        s1_texts = [
            format_record(n, a) for n, a in zip(
                s1_c["business_name"].to_list(),
                s1_c["business_address"].to_list()
            )
        ]
        del s1_c
        gc.collect()

        print(f"Encoding {len(s1_texts):,} S1 queries on GPU...")
        with torch.inference_mode():
            s1_embs = embed_model.encode(
                s1_texts,
                batch_size=1024,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=True,
                device=device
            ).half()

        # Step 3: Blocked Matrix Multiplication on CUDA
        print("Computing similarity matrix and selecting matches...")
        Q_BATCH = 2048
        T_BLOCK = 100000
        num_s1 = len(s1_ids)
        num_targets = len(target_ids)

        with torch.inference_mode():
            for q_start in tqdm(range(0, num_s1, Q_BATCH), desc=f"Querying {country}"):
                q_end = min(q_start + Q_BATCH, num_s1)
                q_embs_gpu = s1_embs[q_start:q_end].to(device)

                batch_cand_scores = [[] for _ in range(q_end - q_start)]
                batch_cand_indices = [[] for _ in range(q_end - q_start)]

                for t_start in range(0, num_targets, T_BLOCK):
                    t_end = min(t_start + T_BLOCK, num_targets)
                    t_block_gpu = target_embs_cpu[t_start:t_end].to(device)

                    # Instant Cosine Similarity: (Q_batch x T_block)
                    sim_matrix = torch.matmul(q_embs_gpu, t_block_gpu.T)

                    # Threshold filter: keep candidates >= candidate_threshold
                    mask = sim_matrix >= candidate_threshold
                    for row_idx in range(q_end - q_start):
                        matching_cols = torch.nonzero(mask[row_idx]).squeeze(-1)
                        if matching_cols.numel() > 0:
                            scores = sim_matrix[row_idx, matching_cols].cpu().tolist()
                            global_t_idx = (matching_cols + t_start).cpu().tolist()
                            if isinstance(scores, float):
                                scores = [scores]
                                global_t_idx = [global_t_idx]
                            batch_cand_scores[row_idx].extend(scores)
                            batch_cand_indices[row_idx].extend(global_t_idx)

                # Process batch results
                for row_idx in range(q_end - q_start):
                    s1_id = s1_ids[q_start + row_idx]
                    c_scores = batch_cand_scores[row_idx]
                    c_indices = batch_cand_indices[row_idx]

                    if not c_indices:
                        results_matching[s1_id] = ""
                        results_candidates[s1_id] = ""
                        continue

                    # Sort by similarity descending and cap top-k candidates
                    sorted_pairs = sorted(zip(c_scores, c_indices), key=lambda x: x[0], reverse=True)[:top_k_candidates]

                    cand_ids = [target_ids[idx] for _, idx in sorted_pairs]
                    match_ids = [target_ids[idx] for sc, idx in sorted_pairs if sc >= match_threshold]

                    results_candidates[s1_id] = ",".join(cand_ids)
                    results_matching[s1_id] = ",".join(match_ids)

        del s1_embs, target_embs_cpu, target_ids
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Step 4: Write Submission Files strictly in original S1 row order
    print(f"\nWriting submission files to {matching_out} and {candidate_out}...")
    with open(matching_out, "w", encoding="utf-8") as f_match, \
         open(candidate_out, "w", encoding="utf-8") as f_cand:
        
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        singletons = 0
        for s1_id in all_s1_ids:
            m = results_matching.get(s1_id, "")
            c = results_candidates.get(s1_id, "")
            f_match.write(f"{s1_id}\t{m}\n")
            f_cand.write(f"{s1_id}\t{c}\n")
            if not m:
                singletons += 1

    print(f"Done! Predicted singletons: {singletons:,} / {len(all_s1_ids):,} ({singletons/len(all_s1_ids)*100:.2f}%)")

if __name__ == "__main__":
    run_gpu_entity_resolution()
