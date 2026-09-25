import os
import joblib
import numpy as np
import lightgbm as lgb

try:
    from .features import FEATURE_NAMES
    from .evaluate import compute_macro_f05
except ImportError:
    from features import FEATURE_NAMES
    from evaluate import compute_macro_f05

class EntityMatchingModel:
    """LightGBM Classifier with calibrated thresholding for macro F_0.5 optimization."""
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.model = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            n_estimators=350,
            learning_rate=0.06,
            num_leaves=63,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.feature_names = FEATURE_NAMES

    def fit(self, X, y):
        """Train LightGBM model on extracted feature matrix X and binary labels y."""
        self.model.fit(X, y)

    def predict_proba(self, X):
        """Predict match probabilities."""
        if len(X) == 0:
            return np.array([])
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(self, validation_pairs, ground_truth_map):
        """
        Tune decision threshold strictly against macro F_0.5 metric on validation set.
        validation_pairs is list of (s1_id, cand_id, score).
        """
        s1_to_cand_scores = {}
        for s1_id, cand_id, score in validation_pairs:
            if s1_id not in s1_to_cand_scores:
                s1_to_cand_scores[s1_id] = []
            s1_to_cand_scores[s1_id].append((cand_id, score))

        best_th = self.threshold
        best_f05 = -1.0

        for th in np.arange(0.50, 0.92, 0.02):
            preds = {}
            for s1_id in ground_truth_map:
                cand_list = s1_to_cand_scores.get(s1_id, [])
                matched = {cid for cid, sc in cand_list if sc >= th}
                preds[s1_id] = matched
            score = compute_macro_f05(ground_truth_map, preds)
            if score > best_f05:
                best_f05 = score
                best_th = float(th)

        self.threshold = best_th
        return best_th, best_f05

    def save(self, filepath):
        """Save model and threshold to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({'model': self.model, 'threshold': self.threshold}, filepath)

    def load(self, filepath):
        """Load model and threshold from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.threshold = data.get('threshold', 0.75)
