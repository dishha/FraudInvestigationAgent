def counterfactual_explanation(model, row, top_shap_features, target_threshold=0.5):
    results = []
    for feat in top_shap_features:
        perturbed = row.copy()
        original_val = row[feat]
        # Try reducing by 25%, 50%, 75%
        for reduction in [0.25, 0.5, 0.75]:
            perturbed[feat] = original_val * (1 - reduction)
            new_score = model.predict(perturbed)[0]
            if new_score < target_threshold:
                results.append({
                    "feature": feat,
                    "original": original_val,
                    "changed_to": perturbed[feat],
                    "new_score": new_score,
                    "reduction_pct": reduction * 100
                })
                break
    return results  # "If X were 50% lower, score drops from 0.87 → 0.43"