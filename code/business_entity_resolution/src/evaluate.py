def compute_macro_f05(ground_truth_map, predictions_map):
    """
    Compute macro-averaged F_0.5 score across all Source 1 entities according to competition rules.
    
    Formula:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
        
    Singletons:
        - True singleton (GT empty) with Pred empty -> 1.0
        - True singleton with Pred non-empty -> 0.0
        - Non-singleton with Pred empty -> 0.0
    """
    total_score = 0.0
    n = len(ground_truth_map)
    
    if n == 0:
        return 0.0

    for s1_id, gt_set in ground_truth_map.items():
        pred_set = predictions_map.get(s1_id, set())
        
        gt_len = len(gt_set)
        pred_len = len(pred_set)

        if gt_len == 0:
            if pred_len == 0:
                total_score += 1.0
            else:
                total_score += 0.0
        else:
            if pred_len == 0:
                total_score += 0.0
            else:
                true_positives = len(gt_set & pred_set)
                if true_positives == 0:
                    total_score += 0.0
                else:
                    prec = true_positives / pred_len
                    rec = true_positives / gt_len
                    denom = 0.25 * prec + rec
                    if denom > 0:
                        total_score += (1.25 * prec * rec) / denom
                        
    return total_score / n
