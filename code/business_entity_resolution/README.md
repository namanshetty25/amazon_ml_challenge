# Business Entity Resolution Solution

An end-to-end Machine Learning solution for cross-source Business Entity Resolution, achieving state-of-the-art precision-weighted F_0.5 performance across diverse geographical jurisdictions (US, India, France).

---

## 1. Pipeline Architecture

The solution operates as a modular multi-stage pipeline:


1. **Preprocessing & Standardization (`src/preprocess.py`)**:
   - URL & domain removal (`.com`, `.org`, `.net`, `.in`, `.fr`, etc.)
   - Doing-Business-As (DBA) extraction & alias resolution
   - Legal suffix normalization & stripping (`pvt ltd`, `inc`, `llc`, `corp`, `gmbh`, `sarl`, `sa`, etc.)
   - Address token standardization (road abbreviations, municipal and unit identifiers)
   - Digits and key tokens extraction

2. **High-Recall Multi-Pass Blocking (`src/blocking.py`)**:
   - Strictly isolates candidate generation by country (`US`, `India`, `France`) to guarantee zero cross-country contamination and minimize computational search space.
   - Complementary blocking keys:
     - Exact normalized skeleton name
     - Compact whitespace-free skeleton name
     - Alphabetically sorted token permutation
     - Prefix n-gram (first two tokens)
     - Full cleaned name
     - Hybrid name token + address house number / digits
     - Address house number + first street token
   - Inverted indexing with compact integer arrays for sub-millisecond lookup.

3. **Feature Engineering (`src/features.py`)**:
   - 23 vectorized features capturing orthographic, token-level, and phonetic similarities:
     - Levenshtein ratio, partial ratio, token sort ratio, token set ratio, WRatio via RapidFuzz
     - Skeleton name similarity and exact match flags
     - Token Jaccard overlap and character length ratios
     - Address token set and sort ratios
     - Address numerical / PIN / house number exact match & Jaccard
     - Null/missing address indicators
     - Joint multi-field similarity and substring inclusions

4. **Machine Learning Model & Threshold Calibration (`src/model.py`)**:
   - Gradient Boosted Decision Trees (LightGBM) trained on balanced candidate pairs with negative subsampling.
   - Threshold calibration strictly tuned on a held-out validation set to maximize the macro-averaged F_0.5 metric (precision weighted 2x over recall).
   - Singleton decision policy: Entities with no candidates exceeding confidence threshold are assigned empty matches (`1.0` reward).

5. **Submission Generation & Validation (`src/pipeline.py`)**:
   - Generates compliant `matching_results.tsv` and `candidate_pairs.tsv` in `output/`.

---

## 2. Reproduction Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run End-to-End Pipeline
From the project root (`student_resource/`):
# Option A (Standard CPU ML Pipeline):
python run.py --mode all --dataset-dir dataset --output-dir output
# Or running the module directly:
python code/business_entity_resolution/src/pipeline.py --mode all --dataset-dir dataset --output-dir output

# Option B (GPU Dense Embeddings + PyTorch CUDA Matrix Multiplication):
python gpu_pipeline.py --dataset-dir dataset --output-dir output
```

### Step 3: Validate Outputs
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
