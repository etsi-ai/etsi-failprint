def suggest_counterfactual(failure_row, X_all, target_col):
    features = failure_row.drop(labels=[target_col], errors='ignore')
    candidate_pool = X_all.drop_duplicates()

    min_diff = float('inf')
    best_match = None

    for _, row in candidate_pool.iterrows():
        if row[features.index.tolist()].equals(features):
            continue


        diff = sum(row[feat] != val for feat, val in features.items())

        if diff < min_diff:
            min_diff = diff
            best_match = row

    if best_match is not None:
        changes = {feat: best_match[feat] for feat in features.index if best_match[feat] != features[feat]}
        print("Changes computed:", changes)  
        return changes

    print("No suitable match found.") 
    return None
