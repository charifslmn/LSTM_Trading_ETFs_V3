import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from itertools import combinations
from collections import Counter

from Equations_Run_Combo_V_2 import run_combo_V_4


#### notice the eq below is also in the Equations_Run_Combo_V_2 file
def evaluate_binary_0_1_selective_ensemble(predicted_array_flat, actual_array_flat,do_print : bool):

    predicted_array_correction = []
    actual_array_correction = []
    actual_array_all = []
        
    for idx, (pred,act) in enumerate(zip(predicted_array_flat,actual_array_flat)):
        if not isinstance(pred, str) and not None :
            predicted_array_correction.append(pred)
            actual_array_correction.append(act)
        
        actual_array_all.append(act)

    # print(predicted_array_correction)
    # # print(predicted_array_correction)
    # # print(actual_array_correction)

    if not predicted_array_correction:      # if predicted_array_correction == [] or None:

        return {
        'accuracy': 'No Agreed Predictions',
        'precision_up': 'No Agreed Predictions',
        'recall_up': 'No Agreed Predictions',
        'precision_down': 'No Agreed Predictions',
        'recall_down': 'No Agreed Predictions',
    }
    
    else:
        # predicted_array_correction = [i for i in predicted_array_correction]
        # actual_array_correction = [i for i in actual_array_correction]

        # actual_array_all = [i for i in actual_array_all] ### FIX RECALL

        predicted_array_correction = np.array(predicted_array_correction)
        actual_array_correction = np.array(actual_array_correction)
        actual_array_all = np.array(actual_array_all) ### FIX RECALL

        pred_direction = (predicted_array_correction > 0.5).astype(int)
        actual_direction = (actual_array_correction > 0.5).astype(int)
        actual_all_direction = (actual_array_all > 0.5).astype(int) ### FIX RECALL


        correct = (pred_direction == actual_direction).astype(int)
        
        accuracy = correct.sum() / len(correct) * 100
        actual_ups = (actual_direction == 1)

        actual_all_ups = (actual_all_direction == 1) ### FIX RECALL

        predicted_ups = (pred_direction == 1)
        true_positives_up = (predicted_ups & actual_ups).sum()
        precision_up = true_positives_up / predicted_ups.sum() * 100 if predicted_ups.sum() > 0 else float('nan')
        recall_up = true_positives_up / actual_all_ups.sum() * 100 if actual_all_ups.sum() > 0 else float('nan')
        actual_downs = (actual_direction == 0)

        actual_all_downs = (actual_all_direction == 0) ### FIX RECALL

        predicted_downs = (pred_direction == 0)
        true_positives_down = (predicted_downs & actual_downs).sum()
        precision_down = true_positives_down / predicted_downs.sum() * 100 if predicted_downs.sum() > 0 else float('nan')
        recall_down = true_positives_down / actual_all_downs.sum() * 100 if actual_all_downs.sum() > 0 else float('nan')

        if actual_ups.sum() == 0 and predicted_ups.sum() == 0:
            precision_up = None
            recall_up = None

        if actual_ups.sum() == 0 and predicted_ups.sum() > 0:
            precision_up = 0
            recall_up = None      

        if actual_ups.sum() > 0 and predicted_ups.sum() == 0:
            precision_up = None
            recall_up = 0

            ####################################

        if actual_downs.sum() == 0 and predicted_downs.sum() == 0:
            precision_down = None
            recall_down = None

        if actual_downs.sum() == 0 and predicted_downs.sum() > 0:
            precision_down = 0
            recall_down = None
        
        if actual_downs.sum() > 0 and predicted_downs.sum() == 0:
            precision_down = None
            recall_down = 0


        # if do_print:
        #     print(f"Directional Accuracy: {accuracy:.2f}%")
        #     print(f'Up Precision: {precision_up:.2f}%')
        #     print(f'Up Recall:    {recall_up:.2f}%')
        #     print(f'Down Precision: {precision_down:.2f}%')
        #     print(f'Down Recall:    {recall_down:.2f}%')
        return {
            'accuracy': accuracy,
            'precision_up': precision_up,
            'recall_up': recall_up,
            'precision_down': precision_down,
            'recall_down': recall_down,
        }




def selective_ensemble_all_agree_thresh(existing_data: list, combo_list, num_cv_sets, total_offset, INDEX, combo_numbers, 
                                 use_existing_data: bool): 

    if not use_existing_data:
        with parallel_backend("loky", n_jobs=1):
            all_results = Parallel()(
                delayed(run_combo_V_4)(i, combo, 0, use_print_acc_vs_pred=False)
                for i, combo in enumerate(combo_list)
            )
        results, weights = zip(*all_results)

    if use_existing_data:
        results = existing_data

    all_model_preds = [res["all_preds"] for res in results]  # shape: (num_models, num_folds)

    transposed_preds = list(zip(*all_model_preds))  
    
    # Handle string values in predictions (like 'below_threshold')
    stacked_bool_int = []
    for fold_preds in transposed_preds:
        fold_array = []
        for model_preds in fold_preds:
            # Convert each prediction, handling string values
            converted_preds = []
            for pred in model_preds:
                if isinstance(pred, str):
                    # For string values like 'below_threshold', treat as 0 (down prediction)
                    converted_preds.append(0)
                else:
                    converted_preds.append(1 if pred > 0.5 else 0)
            fold_array.append(converted_preds)
        stacked_bool_int.append(np.array(fold_array))
    
    predictions_folds = []

    for fold in stacked_bool_int:
        agreement = (fold == fold[0]).all(axis=0)  
        agreed_vals = fold[0]  
    
        result = [
            agreed_vals[i] if agreement[i] else "no agreement"
            for i in range(fold.shape[1])
        ]
        
        predictions_folds.append(result)

    actuals_folds = results[0]["all_actuals"]
    raw_actuals_folds = results[0]["raw_actuals"]

    cv_data = {}  

    for set_idx, (pred_fold, act_fold) in enumerate(zip(predictions_folds, actuals_folds)):
        metrics = evaluate_binary_0_1_selective_ensemble(pred_fold, act_fold, do_print=False)
        cv_data[f"set_{set_idx + 1}"] = metrics       

    metrics_keys = cv_data[f"set_{set_idx + 1}"].keys()

    overall_avg = {}
    for k in metrics_keys:
        values = [cv_data[f"set_{i + 1}"][k] for i in range(num_cv_sets)]
        numeric_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)) and isinstance(v, (int, float))]
        if len(numeric_values) > 0:
            overall_avg[k] = np.mean(numeric_values)
        else:
            overall_avg[k] = None

    cv_data["avg_across_all_sets"] = overall_avg

    # Overall metrics across ALL folds
    flat_preds = [p for fold in predictions_folds for p in fold]
    flat_actuals = [a for fold in actuals_folds for a in fold]
    cv_data["overall_metrics"] = evaluate_binary_0_1_selective_ensemble(flat_preds, flat_actuals, do_print=False)

    result_entry = {
        "combo_number": total_offset + INDEX + 1,
        "parameters": combo_list,
        "cv_sets": cv_data,
        "all_preds": predictions_folds,
        "all_actuals": actuals_folds,
        "raw_actuals": raw_actuals_folds, 
        "combo_numbers": combo_numbers
    }

    return result_entry



def add_threshold_metrics(data: List[Dict] ,threshold: float) -> None:
    """Add threshold metrics to result data"""
    for res in data:
        all_preds = res["all_preds"]
        actuals_per_fold = res["all_actuals"]
        actuals_per_fold_flat = [i for fold in res["all_actuals"] for i in fold]


        pred_threshold_sigmoid01_up = threshold
        all_actuals_threshold_per_fold = []  ; all_preds_threshold_per_fold = []


        for p_fold, a_fold in zip(all_preds, actuals_per_fold):
            new_p_fold = [] ; new_a_fold = [] 

            for p, a in zip(p_fold, a_fold):
                if p > 0.5 and p > pred_threshold_sigmoid01_up:
                    new_p_fold.append(p) ; new_a_fold.append(a)


                elif p > 0.5 and p <= pred_threshold_sigmoid01_up:
                    new_p_fold.append('below_threshold') ; new_a_fold.append('below_threshold')

                else:
                    new_p_fold.append(p) ; new_a_fold.append(a)


            all_preds_threshold_per_fold.append(new_p_fold) ; all_actuals_threshold_per_fold.append(new_a_fold)


        all_actuals_threshold_per_fold_flattened = [j for parts in all_actuals_threshold_per_fold for j in parts] 
        all_preds_threshold_per_fold_flattened = [j for parts in all_preds_threshold_per_fold for j in parts]


        res["all_preds_threshold"] = all_preds_threshold_per_fold
        res["all_actuals_threshold"] = all_actuals_threshold_per_fold


        threshold_metrics = evaluate_binary_0_1_selective_ensemble(
            all_preds_threshold_per_fold_flattened, 
            actuals_per_fold_flat, 
            do_print=False
        )
        
        threshold_metrics_renamed = {}
        for key, value in threshold_metrics.items():
            threshold_metrics_renamed[f"{key}_thresh"] = value
        
        res["overall_metrics_thresh"] = threshold_metrics_renamed


def flatten_results(result_list: List[Dict]) -> pd.DataFrame:
    """Flatten one list of results into a DataFrame without modifying the original."""
    flattened = []
    for entry in result_list:
        flat_entry = {k: v for k, v in entry.items() if k != "cv_sets"}
        flat_entry.update(entry["cv_sets"])
        flattened.append(flat_entry)
    return pd.DataFrame(flattened)

def add_up_prediction_counts(dfs: List[pd.DataFrame]) -> None:
    """Add number of up predictions for normal and threshold versions"""
    for df in dfs:
        no_up_list = []  
        no_up_thresh_list = []
        for row in df["all_preds"]:
            flat_preds = [p for fold in row for p in fold]
            no_up_list.append(sum(p > 0.5 for p in flat_preds))
        for row in df["all_preds_threshold"]:
            flat_preds = [p for fold in row for p in fold]
            no_up_thresh_list.append(sum(isinstance(p, (int, float)) and p > 0.5 for p in flat_preds))
        df["no_up_preds"] = no_up_list
        df["no_up_preds_thresh"] = no_up_thresh_list





def add_false_correct_up_stats(dfs: List[pd.DataFrame]) -> None:
    """Add false and correct up prediction statistics"""
    for df in dfs:
        false_up_preds_col_list_actual = []
        false_up_preds_col_probabs_list_actual = []
        false_up_preds_col_list_actual_thresh = []
        false_up_preds_col_probabs_list_actual_thresh = []
        correct_up_preds_col_list_actual = []
        correct_up_preds_col_probabs_list = []
        correct_up_preds_col_list_actual_thresh = []
        correct_up_preds_col_probabs_list_thresh = []


        for row_raw_actuals, row_01_actuals, row_preds, row_preds_thresh in zip(
            df["raw_actuals"], df["all_actuals"], df["all_preds"], df["all_preds_threshold"]
        ):


            false_up_preds_row_actual = []
            false_up_preds_row_probabs = []
            false_up_preds_row_actual_thresh = []
            false_up_preds_row_probabs_thresh = []
            correct_up_preds_row_actual = []
            correct_up_preds_row_probabs = []
            correct_up_preds_row_actual_thresh = []
            correct_up_preds_row_probabs_thresh = []


            row_raw_actuals_flattened = [p for fold in row_raw_actuals for p in fold]
            row_01_actuals_flattened = [p for fold in row_01_actuals for p in fold]
            row_preds_flattened = [p for fold in row_preds for p in fold]
            row_preds_thresh_flattened = [p for fold in row_preds_thresh for p in fold]


            # Normal version
            for entry_raw_actual, entry_01_actual, entry_pred in zip(row_raw_actuals_flattened, row_01_actuals_flattened, row_preds_flattened):
                if entry_pred > 0.5 and entry_01_actual < 0.5:
                    false_up_preds_row_actual.append(round(entry_raw_actual, 4)) 
                    false_up_preds_row_probabs.append(round(entry_pred,4))


            for entry_raw_actual, entry_01_actual, entry_pred in zip(row_raw_actuals_flattened, row_01_actuals_flattened, row_preds_flattened):
                if entry_pred > 0.5 and entry_01_actual > 0.5:
                    correct_up_preds_row_actual.append(round(entry_raw_actual, 4)) 
                    correct_up_preds_row_probabs.append(round(entry_pred,4))


            # Threshold version
            for entry_raw_actual, entry_01_actual, entry_pred in zip(row_raw_actuals_flattened, row_01_actuals_flattened, row_preds_thresh_flattened):
                if ( not isinstance(entry_pred, str)) and entry_pred > 0.5 and entry_01_actual < 0.5:
                    false_up_preds_row_actual_thresh.append(round(entry_raw_actual, 4))
                    false_up_preds_row_probabs_thresh.append(round(entry_pred,4))


            for entry_raw_actual, entry_01_actual, entry_pred in zip(row_raw_actuals_flattened, row_01_actuals_flattened, row_preds_thresh_flattened):
                
                if ( not isinstance(entry_pred, str) ) and entry_pred > 0.5 and entry_01_actual > 0.5:
                    correct_up_preds_row_actual_thresh.append(round(entry_raw_actual, 4))
                    correct_up_preds_row_probabs_thresh.append(round(entry_pred, 4))


            false_up_preds_col_list_actual.append(false_up_preds_row_actual)
            false_up_preds_col_probabs_list_actual.append(false_up_preds_row_probabs)
            false_up_preds_col_list_actual_thresh.append(false_up_preds_row_actual_thresh)
            false_up_preds_col_probabs_list_actual_thresh.append(false_up_preds_row_probabs_thresh)
            
            correct_up_preds_col_list_actual.append(correct_up_preds_row_actual)
            correct_up_preds_col_probabs_list.append(correct_up_preds_row_probabs)
            correct_up_preds_col_list_actual_thresh.append(correct_up_preds_row_actual_thresh)
            correct_up_preds_col_probabs_list_thresh.append(correct_up_preds_row_probabs_thresh)



        df["actuals_false_up"] = false_up_preds_col_list_actual
        df["false_up_preds"] = false_up_preds_col_probabs_list_actual
        df["actuals_false_up_thresh"] = false_up_preds_col_list_actual_thresh
        df["false_up_preds_thresh"] = false_up_preds_col_probabs_list_actual_thresh
        df["actuals_correct_up"] = correct_up_preds_col_list_actual
        df["correct_up_preds"] = correct_up_preds_col_probabs_list
        df["actuals_correct_up_thresh"] = correct_up_preds_col_list_actual_thresh
        df["correct_up_preds_thresh"] = correct_up_preds_col_probabs_list_thresh


################################################################## TESTING

def flatten_metrics_columns(dfs: List[pd.DataFrame]) -> None:
    """Flatten overall metrics columns"""
    for df in dfs:
        df.rename(columns={"overall_metrics": "overall"}, inplace=True)
    
    for df in dfs:
        if "overall" in df.columns:
            params_df = pd.json_normalize(df["overall"])
            params_df.columns = [f"OA_{col}" for col in params_df.columns]
            df[params_df.columns] = params_df
        
        if "overall_metrics_thresh" in df.columns:
            params_df = pd.json_normalize(df["overall_metrics_thresh"])
            params_df.columns = [f"OA_thresh_{col}" for col in params_df.columns]
            df[params_df.columns] = params_df



def process_parameters_and_merge(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Process parameters and create master DataFrame"""

    if len(dfs) == 2:
        concat_df_val = dfs[0]
        concat_df_test = dfs[1]
    if len(dfs) == 4:
        concat_df_val = pd.concat([dfs[0], dfs[1]], ignore_index=True)
        concat_df_test = pd.concat([dfs[2], dfs[3]], ignore_index=True)
    
    temp_dfs = [concat_df_val, concat_df_test]
    
    for df in temp_dfs:
        params_chosen = []
        for param_dict in df["parameters"]:
            param_dict_fix = {}
            for param_key in param_dict.keys():
                if param_key not in ['val_start_month', 'val_end_month']:
                    param_dict_fix[str(param_key)] = param_dict[param_key]
            params_chosen.append(str(param_dict_fix))
        df["params_fix"] = params_chosen

    dict_fix_params = {}
    for idx, p in enumerate(concat_df_test["params_fix"]):
        p_str = str(p).replace(' ', '').replace("'", '').replace('  ', '')
        dict_fix_params[p_str] = idx

    for df in temp_dfs:
        param_values = []
        for p in df["params_fix"]:
            p_str = str(p).replace(' ', '').replace("'", '')
            param_values.append(dict_fix_params.get(p_str, -1))
        df["param_int_value"] = param_values

    concat_df_val.rename(columns=lambda x: f"{x}_mac_val" if x != "params_fix" else x, inplace=True)
    concat_df_test.rename(columns=lambda x: f"{x}_mac_test" if x != "params_fix" else x, inplace=True)

    master_df = concat_df_val.merge(concat_df_test, on="params_fix", how="outer")
    
    master_df.columns = (
        master_df.columns
        .str.replace('precision', 'prec')
        .str.replace('recall', 'rec')
        .str.replace('down', '0')
        .str.replace('up', '1')
        .str.replace('accuracy', 'acc')
        .str.replace('test', 'T')
        .str.replace('val', 'V')
    )
    
    return master_df



############## NEW NEW NEW NEW 

def get_model_groups_in_corr_range_diff_params(master_df, machine, set_type, corr_range, group_size=2, 
                                 use_spearman_bool=False, num_diff_params=None):
    """
    Find groups of models (size=group_size) whose pairwise correlations
    of predictions fall within [low, high), with optional parameter difference filtering.

    Args:
        master_df (pd.DataFrame): your merged DF.
        machine (str): 'mac' or 'gc'.
        set_type (str): 'V' or 'T'.
        corr_range (tuple): (low, high) inclusive/exclusive as [low, high).
        group_size (int): number of models per group (>=2).
        use_spearman_bool (bool): if True, use Spearman correlation; else use Pearson.
        num_diff_params (int or None): Minimum number of different parameters required 
                                     between models in the group. If None, no filtering.

    Returns:
        list[tuple]: list of tuples of model IDs (param_int_Vue_...) meeting the criteria.
    """
    assert group_size >= 2, "group_size must be >= 2"

    preds_col = f"all_preds_{machine}_{set_type}"
    id_col = f"param_int_Vue_{machine}_{set_type}"
    params_col = f"parameters_{machine}_{set_type}"

    # Keep only rows with predictions, ID, and parameters
    block = master_df.loc[master_df[preds_col].notna() & 
                         master_df[id_col].notna() & 
                         master_df[params_col].notna(), 
                         [id_col, preds_col, params_col]]

    # Flatten predictions per model id and store parameters
    preds_by_id = {}
    params_by_id = {}
    
    for _, row in block.iterrows():
        mid = int(row[id_col])
        flat = [p for fold in row[preds_col] for p in fold]
        preds_by_id[mid] = np.asarray(flat, dtype=float)
        params_by_id[mid] = row[params_col]  # Store parameters

    if not preds_by_id:
        return []

    model_ids = sorted(preds_by_id.keys())

    # Build data matrix (n_models, n_samples)
    data = np.vstack([preds_by_id[mid] for mid in model_ids])

    # Correlation matrix
    if use_spearman_bool:
        from scipy.stats import spearmanr
        corr = spearmanr(data, axis=1).correlation
    else:
        corr = np.corrcoef(data)

    low, high = corr_range

    groups = []
    idx_map = {i: mid for i, mid in enumerate(model_ids)}

    # Fast path for pairs
    if group_size == 2:
        for i in range(len(model_ids)):
            for j in range(i+1, len(model_ids)):
                c = corr[i, j]
                if not np.isnan(c) and (low <= c < high):
                    groups.append((idx_map[i], idx_map[j]))
    else:
        # For k >= 3: check all combinations are within range
        for combo in combinations(range(len(model_ids)), group_size):
            ok = True
            for a, b in combinations(combo, 2):
                c = corr[a, b]
                if np.isnan(c) or not (low <= c < high):
                    ok = False
                    break
            if ok:
                groups.append(tuple(idx_map[i] for i in combo))

    # Filter by parameter differences if requested
    if num_diff_params is not None and groups:
        filtered_groups = []
        
        for group in groups:
            # Get parameters for all models in this group
            group_params = [params_by_id[mid] for mid in group]
            
            # Check if all models have the same parameter structure
            if not all(isinstance(params, dict) for params in group_params):
                continue
                
            # Count different parameters across all pairs in the group
            min_diff_params = float('inf')
            
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    params1 = group_params[i]
                    params2 = group_params[j]
                    
                    # Count different parameters between this pair
                    diff_count = 0
                    all_keys = set(params1.keys()) | set(params2.keys())
                    
                    for key in all_keys:
                        val1 = params1.get(key)
                        val2 = params2.get(key)
                        
                        # Consider parameters different if:
                        # 1. One has the parameter and the other doesn't
                        # 2. Both have the parameter but with different values
                        if (key not in params1 or key not in params2) or (val1 != val2):
                            diff_count += 1
                    
                    min_diff_params = min(min_diff_params, diff_count)
            
            # Keep group if minimum difference meets the threshold
            if min_diff_params >= num_diff_params:
                filtered_groups.append(group)
        
        return filtered_groups

    return groups






############## NEW NEW NEW NEW 





def get_model_groups_in_corr_range(master_df, machine, set_type, corr_range, group_size=2, use_spearman_bool=False):
    """
    Find groups of models (size=group_size) whose pairwise correlations
    of predictions fall within [low, high).

    Args:
        master_df (pd.DataFrame): your merged DF.
        machine (str): 'mac' or 'gc'.
        set_type (str): 'V' or 'T'.
        corr_range (tuple): (low, high) inclusive/exclusive as [low, high).
        group_size (int): number of models per group (>=2).
        use_spearman_bool (bool): if True, use Spearman correlation; else use Pearson.

    Returns:
        list[tuple]: list of tuples of model IDs (param_int_Vue_...) meeting the criterion.
    """
    # assert group_size >= 2, "group_size must be >= 2"

    preds_col = f"all_preds_{machine}_{set_type}"
    id_col    = f"param_int_Vue_{machine}_{set_type}"

    # Keep only rows with predictions and an ID
    block = master_df.loc[master_df[preds_col].notna() & master_df[id_col].notna(), [id_col, preds_col]]

    # Flatten predictions per model id
    preds_by_id = {}
    for _, row in block.iterrows():
        mid = int(row[id_col])
        flat = [p for fold in row[preds_col] for p in fold]
        preds_by_id[mid] = np.asarray(flat, dtype=float)

    if not preds_by_id:
        return []

    # Handle potential unequal lengths robustly:
    # lengths = [len(v) for v in preds_by_id.values()]
    # modal_len = Counter(lengths).most_common(1)[0][0]
    # Keep only models with the modal length to ensure comparable correlations
    # preds_by_id = {k: v for k, v in preds_by_id.items() if len(v) == modal_len}

    model_ids = sorted(preds_by_id.keys())


    # Build data matrix (n_models, n_samples)
    data = np.vstack([preds_by_id[mid] for mid in model_ids])

    # Correlation matrix - CHANGED: Added if statement for correlation type
    if use_spearman_bool:
        from scipy.stats import spearmanr
        corr = spearmanr(data, axis=1).correlation
    else:
        corr = np.corrcoef(data)

    low, high = corr_range

    groups = []
    idx_map = {i: mid for i, mid in enumerate(model_ids)}

    # Fast path for pairs
    if group_size == 2:
        for i in range(len(model_ids)):
            for j in range(i+1, len(model_ids)):
                c = corr[i, j]
                if not np.isnan(c) and (low <= c < high):
                    groups.append((idx_map[i], idx_map[j]))
        return groups

    # For k >= 3: check all combinations are within range
    for combo in combinations(range(len(model_ids)), group_size):
        ok = True
        for a, b in combinations(combo, 2):
            c = corr[a, b]
            if np.isnan(c) or not (low <= c < high):
                ok = False
                break
        if ok:
            groups.append(tuple(idx_map[i] for i in combo))

    return groups


##### NEW NEW NEW 


def get_model_groups_in_corr_range_diff_params_and_same_up_preds(master_df, machine, set_type, corr_range, group_size=2, 
                                 use_spearman_bool=False, num_diff_params=None, 
                                 min_same_up_preds=None, max_same_up_preds=None):
    """
    Find groups of models (size=group_size) whose pairwise correlations
    of predictions fall within [low, high), with optional parameter difference filtering
    and same "up" predictions filtering.

    Args:
        master_df (pd.DataFrame): your merged DF.
        machine (str): 'mac' or 'gc'.
        set_type (str): 'V' or 'T'.
        corr_range (tuple): (low, high) inclusive/exclusive as [low, high).
        group_size (int): number of models per group (>=2).
        use_spearman_bool (bool): if True, use Spearman correlation; else use Pearson.
        num_diff_params (int or None): Minimum number of different parameters required 
                                     between models in the group. If None, no filtering.
        min_same_up_preds (int or None): Minimum number of times both models predict "up" (>=0.5)
                                        for the same entry. If None, no filtering.
        max_same_up_preds (int or None): Maximum number of times both models predict "up" (>=0.5)
                                        for the same entry. If None, no filtering.

    Returns:
        list[tuple]: list of tuples of model IDs (param_int_Vue_...) meeting the criteria.
    """
    assert group_size >= 2, "group_size must be >= 2"

    preds_col = f"all_preds_{machine}_{set_type}"
    id_col = f"param_int_Vue_{machine}_{set_type}"
    params_col = f"parameters_{machine}_{set_type}"

    # Keep only rows with predictions, ID, and parameters
    block = master_df.loc[master_df[preds_col].notna() & 
                         master_df[id_col].notna() & 
                         master_df[params_col].notna(), 
                         [id_col, preds_col, params_col]]

    # Flatten predictions per model id and store parameters
    preds_by_id = {}
    params_by_id = {}
    
    for _, row in block.iterrows():
        mid = int(row[id_col])
        flat = [p for fold in row[preds_col] for p in fold]
        preds_by_id[mid] = np.asarray(flat, dtype=float)
        params_by_id[mid] = row[params_col]  # Store parameters

    if not preds_by_id:
        return []

    model_ids = sorted(preds_by_id.keys())

    # Build data matrix (n_models, n_samples)
    data = np.vstack([preds_by_id[mid] for mid in model_ids])

    # Correlation matrix
    if use_spearman_bool:
        from scipy.stats import spearmanr
        corr = spearmanr(data, axis=1).correlation
    else:
        corr = np.corrcoef(data)

    low, high = corr_range

    groups = []
    idx_map = {i: mid for i, mid in enumerate(model_ids)}

    # Fast path for pairs
    if group_size == 2:
        for i in range(len(model_ids)):
            for j in range(i+1, len(model_ids)):
                c = corr[i, j]
                if not np.isnan(c) and (low <= c < high):
                    groups.append((idx_map[i], idx_map[j]))
    else:
        # For k >= 3: check all combinations are within range
        for combo in combinations(range(len(model_ids)), group_size):
            ok = True
            for a, b in combinations(combo, 2):
                c = corr[a, b]
                if np.isnan(c) or not (low <= c < high):
                    ok = False
                    break
            if ok:
                groups.append(tuple(idx_map[i] for i in combo))

    # Filter by parameter differences if requested
    if num_diff_params is not None and groups:
        filtered_groups = []
        
        for group in groups:
            # Get parameters for all models in this group
            group_params = [params_by_id[mid] for mid in group]
            
            # Check if all models have the same parameter structure
            if not all(isinstance(params, dict) for params in group_params):
                continue
                
            # Count different parameters across all pairs in the group
            min_diff_params = float('inf')
            
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    params1 = group_params[i]
                    params2 = group_params[j]
                    
                    # Count different parameters between this pair
                    diff_count = 0
                    all_keys = set(params1.keys()) | set(params2.keys())
                    
                    for key in all_keys:
                        val1 = params1.get(key)
                        val2 = params2.get(key)
                        
                        # Consider parameters different if:
                        # 1. One has the parameter and the other doesn't
                        # 2. Both have the parameter but with different values
                        if (key not in params1 or key not in params2) or (val1 != val2):
                            diff_count += 1
                    
                    min_diff_params = min(min_diff_params, diff_count)
            
            # Keep group if minimum difference meets the threshold
            if min_diff_params >= num_diff_params:
                filtered_groups.append(group)
        
        groups = filtered_groups

    # Filter by same "up" predictions if requested (both min and max)
    if (min_same_up_preds is not None or max_same_up_preds is not None) and groups:
        filtered_groups = []
        
        for group in groups:
            # For groups of size 2, check the pair directly
            if group_size == 2:
                mid1, mid2 = group
                preds1 = preds_by_id[mid1]
                preds2 = preds_by_id[mid2]
                
                # Count number of times both predict "up" (>=0.5)
                both_up_count = np.sum((preds1 >= 0.5) & (preds2 >= 0.5))
                
                # Check both min and max constraints
                min_ok = (min_same_up_preds is None) or (both_up_count >= min_same_up_preds)
                max_ok = (max_same_up_preds is None) or (both_up_count <= max_same_up_preds)
                
                if min_ok and max_ok:
                    filtered_groups.append(group)
            
            # For groups larger than 2, check all pairs
            else:
                keep_group = True
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        mid1, mid2 = group[i], group[j]
                        preds1 = preds_by_id[mid1]
                        preds2 = preds_by_id[mid2]
                        
                        # Count number of times both predict "up" (>=0.5)
                        both_up_count = np.sum((preds1 >= 0.5) & (preds2 >= 0.5))
                        
                        # Check both min and max constraints for this pair
                        min_ok = (min_same_up_preds is None) or (both_up_count >= min_same_up_preds)
                        max_ok = (max_same_up_preds is None) or (both_up_count <= max_same_up_preds)
                        
                        if not (min_ok and max_ok):
                            keep_group = False
                            break
                    
                    if not keep_group:
                        break
                
                if keep_group:
                    filtered_groups.append(group)
        
        groups = filtered_groups

    return groups

def get_model_groups_in_corr_range_diff_params_and_same_up_preds_and_fps( ### notice, the same up preds and the same up fps filters can be used together or seperately
    master_df, machine, set_type, corr_range, group_size=2, 
    use_spearman_bool=False, num_diff_params=None, 
    min_same_up_preds=None, max_same_up_preds=None,
    use_same_up_fps: bool = False, max_same_up_fps: int | None = None
):
    """
    Find groups of models (size=group_size) whose pairwise correlations
    of predictions fall within [low, high), with optional parameter difference filtering,
    same "up" predictions filtering, and optional co–false-positive filtering.

    Args:
        ...
        min_same_up_preds (int or None): Minimum number of times both models predict "up" (>=0.5).
        max_same_up_preds (int or None): Maximum number of times both models predict "up" (>=0.5).
        use_same_up_fps (bool): If True, also filter by times both predict "up" on negatives (FP co-occurrences).
        max_same_up_fps (int or None): Maximum allowed count of co–false-positives (both >=0.5 when actual==0).
    """
    assert group_size >= 2, "group_size must be >= 2"

    preds_col  = f"all_preds_{machine}_{set_type}"
    actuals_col = f"all_actuals_{machine}_{set_type}"  # used only if use_same_up_fps is True
    id_col     = f"param_int_Vue_{machine}_{set_type}"
    params_col = f"parameters_{machine}_{set_type}"

    # Keep only rows with needed columns
    needed_cols = [id_col, preds_col, params_col]
    if use_same_up_fps:
        needed_cols.append(actuals_col)

    block = master_df.loc[
        master_df[preds_col].notna() &
        master_df[id_col].notna() &
        master_df[params_col].notna() &
        (master_df[actuals_col].notna() if use_same_up_fps else True),
        needed_cols
    ]

    # Flatten per model
    preds_by_id = {}
    actuals_by_id = {}
    params_by_id = {}

    for _, row in block.iterrows():
        mid = int(row[id_col])
        flat_preds = [p for fold in row[preds_col] for p in fold]
        preds_by_id[mid] = np.asarray(flat_preds, dtype=float)
        params_by_id[mid] = row[params_col]
        if use_same_up_fps:
            flat_actuals = [a for fold in row[actuals_col] for a in fold]
            actuals_by_id[mid] = np.asarray(flat_actuals, dtype=float)

    if not preds_by_id:
        return []

    model_ids = sorted(preds_by_id.keys())
    data = np.vstack([preds_by_id[mid] for mid in model_ids])

    # Correlation matrix
    if use_spearman_bool:
        from scipy.stats import spearmanr
        corr = spearmanr(data, axis=1).correlation
    else:
        corr = np.corrcoef(data)

    low, high = corr_range
    groups = []
    idx_map = {i: mid for i, mid in enumerate(model_ids)}

    # Build groups by correlation range
    if group_size == 2:
        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                c = corr[i, j]
                if not np.isnan(c) and (low <= c < high):
                    groups.append((idx_map[i], idx_map[j]))
    else:
        from itertools import combinations
        for combo in combinations(range(len(model_ids)), group_size):
            ok = True
            for a, b in combinations(combo, 2):
                c = corr[a, b]
                if np.isnan(c) or not (low <= c < high):
                    ok = False
                    break
            if ok:
                groups.append(tuple(idx_map[i] for i in combo))

    # Filter by parameter differences (optional)
    if num_diff_params is not None and groups:
        filtered_groups = []
        for group in groups:
            group_params = [params_by_id[mid] for mid in group]
            if not all(isinstance(p, dict) for p in group_params):
                continue
            min_diff_params = float('inf')
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    p1, p2 = group_params[i], group_params[j]
                    diff = 0
                    all_keys = set(p1.keys()) | set(p2.keys())
                    for k in all_keys:
                        v1, v2 = p1.get(k), p2.get(k)
                        if (k not in p1 or k not in p2) or (v1 != v2):
                            diff += 1
                    min_diff_params = min(min_diff_params, diff)
            if min_diff_params >= num_diff_params:
                filtered_groups.append(group)
        groups = filtered_groups

    # Filter by same "up" predictions (optional)
    if (min_same_up_preds is not None or max_same_up_preds is not None) and groups:
        filtered_groups = []
        for group in groups:
            if group_size == 2:
                m1, m2 = group
                both_up = np.sum((preds_by_id[m1] >= 0.5) & (preds_by_id[m2] >= 0.5))
                min_ok = (min_same_up_preds is None) or (both_up >= min_same_up_preds)
                max_ok = (max_same_up_preds is None) or (both_up <= max_same_up_preds)
                if min_ok and max_ok:
                    filtered_groups.append(group)
            else:
                keep = True
                from itertools import combinations
                for i, j in combinations(range(len(group)), 2):
                    m1, m2 = group[i], group[j]
                    both_up = np.sum((preds_by_id[m1] >= 0.5) & (preds_by_id[m2] >= 0.5))
                    min_ok = (min_same_up_preds is None) or (both_up >= min_same_up_preds)
                    max_ok = (max_same_up_preds is None) or (both_up <= max_same_up_preds)
                    if not (min_ok and max_ok):
                        keep = False
                        break
                if keep:
                    filtered_groups.append(group)
        groups = filtered_groups

    # NEW: Filter by co–false-positives (optional)
    if use_same_up_fps and (max_same_up_fps is not None) and groups:
        filtered_groups = []
        if group_size == 2:
            for (m1, m2) in groups:
                # Use actuals vector for one model (they should be identical per sample)
                actuals = actuals_by_id[m1]
                both_up_fp = np.sum((preds_by_id[m1] >= 0.5) & (preds_by_id[m2] >= 0.5) & (actuals < 0.5))
                if both_up_fp <= max_same_up_fps:
                    filtered_groups.append((m1, m2))
        else:
            from itertools import combinations
            for group in groups:
                keep = True
                for i, j in combinations(range(len(group)), 2):
                    m1, m2 = group[i], group[j]
                    actuals = actuals_by_id[m1]
                    both_up_fp = np.sum((preds_by_id[m1] >= 0.5) & (preds_by_id[m2] >= 0.5) & (actuals < 0.5))
                    if both_up_fp > max_same_up_fps:
                        keep = False
                        break
                if keep:
                    filtered_groups.append(group)
        groups = filtered_groups

    return groups


##. NEW NEW NEW 

def create_pair_params_map(random_groups: list, master_df: pd.DataFrame, num_models: int, data_type: str = "V") -> dict:
    """
    Create parameter maps for model groups
    
    Args:
        random_groups: List of model ID tuples
        master_df: Master DataFrame containing model data
        num_models: Number of models in each group (2, 3, or 4)
        data_type: 'V' for validation or 'T' for test data
    
    Returns:
        Dictionary with model groups as keys and their parameters/predictions as values
    """
    pair_params_map = {}
    
    for pair in random_groups:
        pair_params_map[pair] = {
            "parameters": [
                master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, f"parameters_mac_{data_type}"].iloc[0]
                for model_id in pair[:num_models]  # Take first num_models from the tuple
            ],
            "predictions": [
                master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, f"all_preds_mac_{data_type}"].iloc[0]
                for model_id in pair[:num_models]
            ],
            "actuals": [
                master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, f"all_actuals_mac_{data_type}"].iloc[0]
                for model_id in pair[:num_models]
            ],
            "raw_actuals": [
                master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, f"raw_actuals_mac_{data_type}"].iloc[0]
                for model_id in pair[:num_models]
            ]
        }
    
    return pair_params_map






def process_ensemble_groups(ensemble_config: list, master_df: pd.DataFrame, data_type: str = "T", 
                           use_threshold_data: bool = False) :
    """
    Process multiple ensemble groups with flexible configuration for both regular and threshold data
    
    Args:
        ensemble_config: List of tuples with (group_name, pair_map, num_models)
        master_df: Master DataFrame containing model data
        data_type: 'V' for validation or 'T' for test data
        use_threshold_data: True for threshold data, False for regular data
    
    Returns:
        Dictionary with ensemble results for all groups
    """
    ensemble_results = {}
    
    for group_name, pair_map, num_models in ensemble_config:
        ensemble_results[group_name] = []
        
        for pair, data in pair_map.items():
            existing_data = []
            individual_model_data = {}
            
            for i, model_id in enumerate(pair[:num_models]):
                # Determine which prediction column to use based on threshold flag
                if data_type == "T":
                    preds_column = "all_preds_threshold_mac_T" if use_threshold_data else "all_preds_mac_T"

                if data_type == "V": ##### ERROR this was missing before sept 14 !!!! so it was always jsut T set
                    preds_column = "all_preds_threshold_mac_V" if use_threshold_data else "all_preds_mac_V"

                # Get the data
                model_preds = master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, preds_column].iloc[0]
                model_actuals = master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, f"all_actuals_mac_{data_type}"].iloc[0]
                model_raw_actuals = master_df.loc[master_df[f"param_int_Vue_mac_{data_type}"] == model_id, f"raw_actuals_mac_{data_type}"].iloc[0]
                
                existing_data.append({
                    "all_preds": model_preds,
                    "all_actuals": model_actuals,
                    "raw_actuals": model_raw_actuals
                })
                
                # Store individual model data for the results
                individual_model_data[f"model_{model_id}_preds"] = model_preds

            # print("process_ensemble_groups: - existing_data:", existing_data)


                ##### ERROR this was missing before sept 14 !!!! so it was always jsut T ser 
            if data_type == "V":
                num_cv_sets = 8 
            if data_type == "T":
                num_cv_sets = 4   
                ##### ERROR this was missing before sept 14 !!!! so it was always jsut T ser 

            
            result = selective_ensemble_all_agree_thresh(
                existing_data=existing_data,
                combo_list=None,
                num_cv_sets=num_cv_sets,
                total_offset=0,
                INDEX=len(ensemble_results[group_name]),
                combo_numbers=pair,
                use_existing_data=True,
            )

            # print("process_ensemble_groups: - result:", result)
            
            # Add individual model data to the result
            result.update(individual_model_data)
            ensemble_results[group_name].append(result)
    
    return ensemble_results






import random



import pickle




def process_func_PLUS_return_analytics(master_df: pd.DataFrame, 
                                      groups_config: list, 
                                      data_type_corr_groups_creation: str = "V",
                                      data_type_ensemble: str = "T",
                                      use_threshold_data: bool = False, 
                                      seed = None, 
                                      num_maps_per_group: int = 10 , 
                                    filter_outliers: bool = False) -> dict:
    """
    Complete function that creates groups, processes ensembles, and returns analytics
    
    Args:
        master_df: Master DataFrame containing model data
        groups_config: List of tuples with (machine, data_type, corr_range, group_size)
        data_type_corr_groups_creation: Data type for correlation group creation ('V' or 'T')
        data_type_ensemble: Data type for ensemble processing ('V' or 'T')
        use_threshold_data: True for threshold data, False for regular data
        seed: Random seed for reproducibility
        num_maps_per_group: Number of pairs/triplets to sample per group
    
    Returns:
        Dictionary with detailed analytics and summary
    """


    # Set random seed
    if seed is not None:
        random.seed(seed)
    
    # Create groups and random samples
    all_pair_maps = {}
    
    for machine, corr_range, group_size in groups_config:
        # Create group name based on parameters
        group_name = f"pair_{group_size}_{corr_range[0]}_{corr_range[1]}".replace('.', '').replace('-', 'neg')
        
        # Get model groups
        groups = get_model_groups_in_corr_range(master_df, machine,data_type_corr_groups_creation , corr_range, group_size=group_size)
        
        if groups:

            # Randomly select from each group
            random_groups = random.sample(groups, min(num_maps_per_group, len(groups))) if groups else []
            
            # Create parameter map for both V and T data
            pair_params_map_V = create_pair_params_map(random_groups, master_df, num_models=group_size, data_type="V")
            pair_params_map_T = create_pair_params_map(random_groups, master_df, num_models=group_size, data_type="T")
            
            # Store both versions
            all_pair_maps[group_name] = {
                "V": pair_params_map_V,
                "T": pair_params_map_T,
                "random_groups": random_groups
            }

        else:
            print(f"No groups found for {group_name} with machine {machine} and data type {data_type_corr_groups_creation}.")

    # Create ensemble configuration using the specified data type
    ensemble_config = []
    for group_name, data in all_pair_maps.items():
        group_size = len(next(iter(data[data_type_ensemble].values()))["parameters"])  # Get group size from first item
        # group_size = len([ v for v in iter(data[data_type_ensemble].keys()) ][0])  # NOTE this is the same as the line above
        ensemble_config.append((group_name, data[data_type_ensemble], group_size))
    
    # Process the data
    ensemble_results = process_ensemble_groups(
        ensemble_config=ensemble_config,
        master_df=master_df,
        data_type=data_type_ensemble,
        use_threshold_data=use_threshold_data
    )

    # Initialize counters
    # total_ups = 0
    # total_correct_ups = 0
    # sum_actuals_ups = 0
    sum_actuals_ups_list_ALL = []


    group_results = {}
    detailed_results = {}




    # Process each group
    for ensemble_group_name, ensemble_group in ensemble_results.items():
        group_output = []
        group_ups = 0
        group_correct_ups = 0
        group_sum_actuals = 0
        group_details = []
        
        # Iterate through each ensemble result in the group
        for ensemble_result in ensemble_group:
            flatten_preds = [p for part in ensemble_result["all_preds"] for p in part]
            flatten_raw_actuals = [a for part in ensemble_result["raw_actuals"] for a in part]
            flatten_actuals = [a for part in ensemble_result["all_actuals"] for a in part]

            p_ups = sum(1 for i in flatten_preds if not isinstance(i, str) and i > 0.5)
            group_ups += p_ups
            # total_ups += p_ups

            up_vals_predicted_raw_val = []

            correct_ups = 0

            for p, a, actual_binary in zip(flatten_preds, flatten_raw_actuals, flatten_actuals):
                if not isinstance(p, str) and p > 0.5:
                    up_vals_predicted_raw_val.append(a)


                    group_sum_actuals += a
                    sum_actuals_ups_list_ALL.append(a)

                    # sum_actuals_ups += a
                    # Check if up prediction was correct (actual_bin > 0.5)
                    if actual_binary > 0.5:
                        correct_ups += 1
            
            group_correct_ups += correct_ups
            # total_correct_ups += correct_ups

            # Create output line
            output_line = (
                f"{ensemble_result['cv_sets']['overall_metrics']['precision_up']} - "
                f"{ensemble_result['cv_sets']['overall_metrics']['recall_up']} "
                f"{ensemble_result['combo_numbers']} - {p_ups} - Correct: {correct_ups} - {up_vals_predicted_raw_val} "
            )
            group_output.append(output_line)
            
            # Store detailed results
            group_details.append({
                "combo_numbers": ensemble_result["combo_numbers"],
                "precision_up": ensemble_result["cv_sets"]["overall_metrics"]["precision_up"],
                "recall_up": ensemble_result["cv_sets"]["overall_metrics"]["recall_up"],
                "up_predictions": p_ups,
                "correct_ups": correct_ups,
                "actual_returns": up_vals_predicted_raw_val,

            })
        
        group_results[ensemble_group_name] = group_output
        detailed_results[ensemble_group_name] = group_details
        
        # Add group summary
        group_results[f"{ensemble_group_name}_summary"] = [
            f"Total Up Predictions: {group_ups}",
            f"Total Correct Up Predictions: {group_correct_ups}",
            f"Sum of Actual Returns for Up Predictions: {group_sum_actuals:.3f}",
            f"Prec Up: {group_correct_ups/group_ups if group_ups > 0 else 0:.3f}"
        ]
        
    #     ## set outlier values past < -0.5 to -0.06
    # if filter_outliers:
    #     for i in range(len(sum_actuals_ups_list_ALL)):
    #         if sum_actuals_ups_list_ALL[i] < -0.5:
    #             sum_actuals_ups_list_ALL[i] = -0.1

    print("before set:", sum_actuals_ups_list_ALL)
    print("after set:", set(sum_actuals_ups_list_ALL))
    total_ups = len(set(sum_actuals_ups_list_ALL))
    total_correct_ups = len(set([val for val in sum_actuals_ups_list_ALL if val > 0.1]))  

    sum_actuals_ups = sum(set(sum_actuals_ups_list_ALL))


    # Create overall summary
    summary = {
        "Total Up Predictions": total_ups,
        "Total Correct Up Predictions": total_correct_ups,
        "Sum of Actual Returns for Up Predictions": sum_actuals_ups,

        "Prec Up": total_correct_ups/total_ups if total_ups > 0 else 0
    }

    # Return structured results
    return {
        "group_results": group_results,
        "detailed_results": detailed_results,
        "summary": summary,
        "total_metrics": {
            "total_ups": total_ups,
            "total_correct_ups": total_correct_ups,
            "sum_actuals_ups": sum_actuals_ups,

            "precision": total_correct_ups / total_ups if total_ups > 0 else 0
        },
        "config_info": {
            "groups_config": groups_config,
            "data_type_corr_groups_creation": data_type_corr_groups_creation,
            "data_type_ensemble": data_type_ensemble,
            "use_threshold_data": use_threshold_data,
            "seed": seed,
            "num_maps_per_group": num_maps_per_group
        }
    }




import numpy as np
from itertools import combinations
from collections import Counter
import random





# names_all = {'res_mac_L_val' : res_mac_L_val, 'res_mac_H_val' : res_mac_H_val, 
#              'res_mac_L_test' : res_mac_L_test, 'res_mac_H_test' : res_mac_H_test}

# names_test = {'res_mac_L_test' : res_mac_L_test, 'res_mac_H_test' : res_mac_H_test}
# names_val = {'res_mac_L_val' : res_mac_L_val, 'res_mac_H_val' : res_mac_H_val}



def process_func_PLUS_return_analytics_THRESH_var_included( #master_df: pd.DataFrame, 
    
    
                                    ## new params
                                    threshold : float,
                                    names_all: dict,

                                    #new arams 
                                
                                    groups_config: list, 
                                    data_type_corr_groups_creation: str = "V",
                                    data_type_ensemble: str = "T",
                                    use_threshold_data: bool = False, 
                                    seed = None, 
                                    num_maps_per_group: int = 10 , 
                                    filter_outliers: bool = False,
                                    use_spearman_corr: bool = False,

                                    use_corr_with_diff_params: bool =True, 
                                    min_diff_params : int = 0,
                                    
                                    use_corr_with_same_up_preds_and_fps: bool = False,
                                    min_same_up_preds: int = None,
                                    max_same_up_preds: int = None,
                                    max_same_up_fps: int = None,
                                    use_same_up_fps_for_corr: bool = False,

                                    ) -> dict:
    



    for data in names_all.values():
        add_threshold_metrics(data, threshold = threshold)

    if len(names_all) == 4 :
 
        df_mac_L_val = flatten_results(names_all["res_mac_L_val"])
        df_mac_H_val = flatten_results(names_all["res_mac_H_val"])
        df_mac_L_test = flatten_results(names_all["res_mac_L_test"])
        df_mac_H_test = flatten_results(names_all["res_mac_H_test"])

                # Add H/L labels
        df_mac_L_val['H_L'] = 'L'
        df_mac_H_val['H_L'] = 'H'
        df_mac_L_test['H_L'] = 'L'
        df_mac_H_test['H_L'] = 'H'

        dfs = [df_mac_L_val, df_mac_H_val, df_mac_L_test, df_mac_H_test]


    if len(names_all) == 2 : ### use the names below for the top combos version 

        df_mac_H_val = flatten_results(names_all["res_mac_H_val"])
        df_mac_H_test = flatten_results(names_all["res_mac_H_test"])
        
                # Add H/L labels
        df_mac_H_val['H_L'] = 'H'
        df_mac_H_test['H_L'] = 'H'

        dfs = [ df_mac_H_val, df_mac_H_test]



    # Add various metrics
    add_up_prediction_counts(dfs)
    add_false_correct_up_stats(dfs)
    flatten_metrics_columns(dfs)

    master_df = process_parameters_and_merge(dfs)

    # Set random seed
    if seed is not None:
        random.seed(seed)
    

    # Create groups and random samples
    all_pair_maps = {}
    groups_data_for_output = {} #### NEW NEW NEW
    
    for machine, corr_range, group_size in groups_config:
        # Create group name based on parameters
        group_name = f"pair_{group_size}_{corr_range[0]}_{corr_range[1]}".replace('.', '').replace('-', 'neg')
        
        # Get model groups
        if use_corr_with_diff_params:
            groups = get_model_groups_in_corr_range_diff_params(master_df, machine, data_type_corr_groups_creation, corr_range, group_size=group_size, use_spearman_bool=use_spearman_corr, num_diff_params=min_diff_params)

        elif use_corr_with_same_up_preds_and_fps:
            groups = get_model_groups_in_corr_range_diff_params_and_same_up_preds_and_fps(master_df, machine, data_type_corr_groups_creation, corr_range, group_size=group_size, use_spearman_bool=use_spearman_corr , num_diff_params=None, min_same_up_preds=min_same_up_preds, max_same_up_preds=max_same_up_preds , use_same_up_fps=use_same_up_fps_for_corr, max_same_up_fps= max_same_up_fps)
        
        else:
            groups = get_model_groups_in_corr_range(master_df, machine, data_type_corr_groups_creation, corr_range, group_size=group_size, use_spearman_bool=use_spearman_corr , num_diff_params=None)
    

        groups_data_for_output[group_name] = {}  #### NEW NEW NEW
        groups_data_for_output[group_name]['all_groups'] = groups  #### NEW NEW NEW
        groups_data_for_output[group_name]['num_groups'] = len(groups)  #### NEW NEW
        
        # Randomly select from each group
        random_groups = random.sample(groups, min(num_maps_per_group, len(groups))) if groups else []
        
        # Create parameter map for both V and T data
        pair_params_map_V = create_pair_params_map(random_groups, master_df, num_models=group_size, data_type="V")
        pair_params_map_T = create_pair_params_map(random_groups, master_df, num_models=group_size, data_type="T")
        
        # Store both versions
        all_pair_maps[group_name] = {
            "V": pair_params_map_V,
            "T": pair_params_map_T,
            "random_groups": random_groups
        }
    
    # Create ensemble configuration using the specified data type
# Create ensemble configuration using the specified data type
    ensemble_config = []
    for group_name, data in all_pair_maps.items():
        group = data.get(data_type_ensemble, {})          # safe: may be missing/empty

        first = next(iter(group.values()), None)          # safe: returns None if empty

        if not first or "parameters" not in first:
            print(f"[skip] {group_name}: no items or missing 'parameters' for '{data_type_ensemble}'")
            continue

        group_size = len(first["parameters"])
        print(f"[use] {group_name}: {len(group)} ensembles of size {group_size} for '{data_type_ensemble}'")
        
        ensemble_config.append((group_name, group, group_size))

    # Process the data
    ensemble_results = process_ensemble_groups(
        ensemble_config=ensemble_config,
        master_df=master_df,
        data_type=data_type_ensemble,
        use_threshold_data=use_threshold_data
    )

    # Initialize counters
    total_ups = 0
    total_correct_ups = 0
    # sum_actuals_ups = 0
    sum_actuals_ups_list_ALL = []


    group_results = {}
    detailed_results = {}



    # Process each group
    for ensemble_group_name, ensemble_group in ensemble_results.items():
        group_output = []
        group_ups = 0
        group_correct_ups = 0
        group_sum_actuals = 0
        group_details = []
        
        # Iterate through each ensemble result in the group
        for ensemble_result in ensemble_group:
            flatten_preds = [p for part in ensemble_result["all_preds"] for p in part]
            flatten_raw_actuals = [a for part in ensemble_result["raw_actuals"] for a in part]
            flatten_actuals = [a for part in ensemble_result["all_actuals"] for a in part]

            p_ups = sum(1 for i in flatten_preds if not isinstance(i, str) and i > 0.5)
            group_ups += p_ups
            total_ups += p_ups

            up_vals_predicted_raw_val = []

            correct_ups = 0

            for p, a, actual_binary in zip(flatten_preds, flatten_raw_actuals, flatten_actuals):
                if not isinstance(p, str) and p > 0.5:
                    up_vals_predicted_raw_val.append(a)


                    group_sum_actuals += a
                    sum_actuals_ups_list_ALL.append(a)

                    # sum_actuals_ups += a
                    # Check if up prediction was correct (actual_bin > 0.5)
                    if actual_binary > 0.5:
                        correct_ups += 1
            
            group_correct_ups += correct_ups
            total_correct_ups += correct_ups

            # Create output line
            output_line = (
                f"{ensemble_result['cv_sets']['overall_metrics']['precision_up']} - "
                f"{ensemble_result['cv_sets']['overall_metrics']['recall_up']} "
                f"{ensemble_result['combo_numbers']} - {p_ups} - Correct: {correct_ups} - {up_vals_predicted_raw_val} "
            )
            group_output.append(output_line)
            
            # Store detailed results
            group_details.append({
                "combo_numbers": ensemble_result["combo_numbers"],
                "precision_up": ensemble_result["cv_sets"]["overall_metrics"]["precision_up"],
                "recall_up": ensemble_result["cv_sets"]["overall_metrics"]["recall_up"],
                "up_predictions": p_ups,
                "correct_ups": correct_ups,
                "actual_returns": up_vals_predicted_raw_val,

            })
        
        group_results[ensemble_group_name] = group_output
        detailed_results[ensemble_group_name] = group_details
        
        # Add group summary
        group_results[f"{ensemble_group_name}_summary"] = [
            f"Total Up Predictions: {group_ups}",
            f"Total Correct Up Predictions: {group_correct_ups}",
            f"Sum of Actual Returns for Up Predictions: {group_sum_actuals:.3f}",
            f"Prec Up: {group_correct_ups/group_ups if group_ups > 0 else 0:.3f}"
        ]

    # ## set outlier values past < -0.5 to -0.06
    # if filter_outliers:
    #     for i in range(len(sum_actuals_ups_list_ALL)):
    #         if sum_actuals_ups_list_ALL[i] < -0.5:
    #             sum_actuals_ups_list_ALL[i] = -0.1

    total_ups = len(set(sum_actuals_ups_list_ALL))
    total_correct_ups = len(set([val for val in sum_actuals_ups_list_ALL if val > 0.1]))  

    sum_actuals_ups = sum(set(sum_actuals_ups_list_ALL))



    # Create overall summary
    summary = {
        "Total Up Predictions": total_ups,
        "Total Correct Up Predictions": total_correct_ups,
        "Sum of Actual Returns for Up Predictions": sum_actuals_ups,
        "Unique Actual Returns for Up Predictions": set(sum_actuals_ups_list_ALL),
        "Prec Up": total_correct_ups/total_ups if total_ups > 0 else 0
    }

    # Return structured results
    return {
        "group_results": group_results,
        "detailed_results": detailed_results,
        "summary": summary,
        "total_metrics": {
            "total_ups": total_ups,
            "total_correct_ups": total_correct_ups,
            "sum_actuals_ups": sum_actuals_ups,
            "unique_actuals_ups_list": set(sum_actuals_ups_list_ALL),

            "precision": total_correct_ups / total_ups if total_ups > 0 else 0
        },


        "config_info": {
            "groups_config": groups_config,
            "data_type_corr_groups_creation": data_type_corr_groups_creation,
            "data_type_ensemble": data_type_ensemble,
            "use_threshold_data": use_threshold_data,
            "seed": seed,
            "num_maps_per_group": num_maps_per_group
        },


       "groups_data": groups_data_for_output 

    }



### USED In the (2) and (3) file to run prediction Algo
# the defaults parameters are used for both ETFS in all yaers 



def collect_V_T_set_FULLraw_data(models_selected, results_dist_disc, results_dist_disc_Tset_same_seeds_organized):
    T_set_data = [] 
    V_set_data = []

    # select 5 models at random from the selected models
    models_RAND = random.sample(models_selected, min(4, len(models_selected)))

    for model_entry in models_RAND:
        model_seeds = []
        ff_combo_idx = model_entry['combo_index']
        for seed_entry in model_entry['selected_seeds']:
            seed_num = seed_entry['seed_num']
            model_seeds.append(seed_num)

        ##### Find randomly chosen seeds in the V and T raw data 
        for raw_data_entry_V in results_dist_disc:
            V_combo_idx = raw_data_entry_V["combo_index"]
            if ff_combo_idx == V_combo_idx:
                rand_chosen_seeds = random.sample(model_seeds, min(1, len(model_seeds)))
                for seed_chosen in rand_chosen_seeds:
                    for seed_raw in raw_data_entry_V['per_seed_all_results']:
                        if seed_raw['seed'] == seed_chosen:
                            V_set_data.append(seed_raw['result_entry'])

        for raw_data_entry_T in results_dist_disc_Tset_same_seeds_organized:
            T_combo_idx = raw_data_entry_T["combo_index"]
            if ff_combo_idx == T_combo_idx:
                for seed_chosen in rand_chosen_seeds:
                    for seed_raw in raw_data_entry_T['per_seed_all_results']:
                        if seed_raw['seed'] == seed_chosen:
                            T_set_data.append(seed_raw['result_entry'])

    return {"V_set_data": V_set_data, "T_set_data": T_set_data}


                               
def process_and_RETURN_analytics_2_3_Model_Performance(V_set , T_set , 
                                                       
                                    num_realizations = 1 ,
                                 
                                 threshold = 0.9, 
                                 min_diff_params = 2,

                                 min_same_up_preds = None ,
                                 max_same_up_preds = None ,

                                 num_maps_per_group = 1 ,

                                max_same_up_fps = 0,
                                 use_same_up_fps_for_corr = True,

                                 do_print = True):
    
    if V_set == [] or T_set == []:
        output = {
                            "all_realizations_unique_actuals_ups_regular_UNIQUE": 0,
                            "all_realizations_unique_actuals_ups_regular_UNIQUE_SUM": 0,
                            "LEN_all_realizations_unique_actuals_ups_regular_UNIQUE": 0,

                            "all_realizations_unique_actuals_ups_threshold_UNIQUE": 0,
                            "all_realizations_unique_actuals_ups_threshold_UNIQUE_SUM": 0,
                            "LEN_all_realizations_unique_actuals_ups_threshold_UNIQUE": 0
                        }
        return output





    names_all = { 'res_mac_H_val': V_set, 
                'res_mac_H_test': T_set}

    # Define your groups configuration
    groups_config = [
        ("mac",  (-1, 1), 2),
        # ("mac",  (0.1, 0.2), 2),
        # ("mac",  (0.3, 0.5), 2),
        # ("mac",  (-1, 0.3), 3),
        # ("mac",  (0.1, 0.3), 3),
        # ("mac",  (0.3, 0.5), 3)
    ]

    number_realizations = num_realizations

    all_results_regular = []
    all_results_threshold = []

    # random_seeds_seed = 4
    # random.seed(random_seeds_seed)

    sample_size = number_realizations
    seeds = random.sample(range(1, 100000), sample_size)

    for i, seed in zip(range(number_realizations), seeds):

        results = process_func_PLUS_return_analytics_THRESH_var_included (
                                            ## new params
                                        threshold = threshold,
                                        names_all = names_all,

                                        #new arams

                                        groups_config = groups_config,
                                        data_type_corr_groups_creation = "V",
                                        data_type_ensemble = "T",
                                        use_threshold_data = False,
                                        seed = None,
                                        num_maps_per_group = num_maps_per_group,
                                        filter_outliers = False ,#### NOTICE NOTICE
                                        use_corr_with_diff_params = False,

                                        min_diff_params = min_diff_params , 
                                        use_spearman_corr = False,
                                        
                                        use_corr_with_same_up_preds_and_fps = True,
                                        min_same_up_preds = min_same_up_preds,
                                        max_same_up_preds = max_same_up_preds ,

                                        use_same_up_fps_for_corr= use_same_up_fps_for_corr,
                                        max_same_up_fps = max_same_up_fps

                                        )
        all_results_regular.append(results)


    for i , seed in zip(range(number_realizations), seeds):

        results = process_func_PLUS_return_analytics_THRESH_var_included (
                                            ## new params
                                        threshold = threshold,
                                        names_all = names_all,

                                        #new arams

                                        groups_config = groups_config,
                                        data_type_corr_groups_creation = "V",
                                        data_type_ensemble = "T",
                                        use_threshold_data = True,
                                        seed = None,
                                        num_maps_per_group = num_maps_per_group,
                                        filter_outliers = False, #### NOTICE NOTICE
                                        use_corr_with_diff_params = False,

                                        min_diff_params = min_diff_params, 
                                        use_spearman_corr = False,
                                        
                                        use_corr_with_same_up_preds_and_fps = True,
                                        min_same_up_preds = min_same_up_preds,
                                        max_same_up_preds = max_same_up_preds,


                                        use_same_up_fps_for_corr= use_same_up_fps_for_corr,
                                        max_same_up_fps = max_same_up_fps

                                        )

        all_results_threshold.append(results)

    all_groups_prec_up_regular = []
    all_groups_total_return_regular = []
    no_up_preds_per_group_regular = []
    all_realizations_unique_actuals_ups_regular = []



            # "Sum of HOD Actual Returns for Up Predictions": HOD_sum_actuals_ups,
            # "Sum of UCO Actual Returns for Up Predictions": UCO_sum_actuals_ups,
            # "Sum of HUC Actual Returns for Up Predictions": HUC_sum_actuals_ups,

    for seed_res in all_results_regular:
        #first set of plots 
        all_groups_total_return_regular.append(seed_res["summary"]["Sum of Actual Returns for Up Predictions"])
        all_groups_prec_up_regular.append(seed_res["summary"]['Prec Up'])
        no_up_preds_per_group_regular.append(seed_res["summary"]['Total Up Predictions'])
        all_realizations_unique_actuals_ups_regular.append(seed_res["summary"]['Unique Actual Returns for Up Predictions'])


    all_realizations_unique_actuals_ups_regular_UNIQUE = set([item for sublist in all_realizations_unique_actuals_ups_regular for item in sublist])
    all_realizations_unique_actuals_ups_regular_UNIQUE_SUM = sum(all_realizations_unique_actuals_ups_regular_UNIQUE)



    all_groups_prec_up_threshold = []
    all_groups_total_return_threshold = []
    no_up_preds_per_group_threshold = []
    all_realizations_unique_actuals_ups_threshold = []



    for seed_res in all_results_threshold:
        all_groups_total_return_threshold.append(seed_res["summary"]["Sum of Actual Returns for Up Predictions"])
        all_groups_prec_up_threshold.append(seed_res["summary"]['Prec Up'])
        no_up_preds_per_group_threshold.append(seed_res["summary"]['Total Up Predictions'])
        all_realizations_unique_actuals_ups_threshold.append(seed_res["summary"]['Unique Actual Returns for Up Predictions'])


    all_realizations_unique_actuals_ups_threshold_UNIQUE = set([item for sublist in all_realizations_unique_actuals_ups_threshold for item in sublist])
    all_realizations_unique_actuals_ups_threshold_UNIQUE_SUM = sum(all_realizations_unique_actuals_ups_threshold_UNIQUE)


    import matplotlib.pyplot as plt

    # List of variable names and titles
    metrics = [
        
        (all_groups_total_return_regular, 'Sum of Actual Returns (Regular)', 'blue'),
        (all_groups_prec_up_regular, 'Precision Up (Regular)', 'blue'),
        (no_up_preds_per_group_regular, 'No Up Predictions (Regular)', 'blue'),

        (all_groups_total_return_threshold, 'Sum of Actual Returns (Threshold)', 'red'),
        (all_groups_prec_up_threshold, 'Precision Up (Threshold)', 'red'),
        (no_up_preds_per_group_threshold, 'No Up Predictions (Threshold)', 'red'),

    ]

    if do_print:
        plt.figure(figsize=(20, 19))

        for i, (var_name, title, color) in enumerate(metrics, 1):

            plt.subplot(6, 3, i)
            data = var_name  # Get the variable by name
            plt.hist(data, bins=15, alpha=0.7, label=title, color=color)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {title}')
            plt.legend()

        plt.tight_layout()
        plt.show()


    # print("ALL Realizations Regular Unique Actuals Ups:", all_realizations_unique_actuals_ups_regular_UNIQUE)
    # print("ALL Realizations Regular Unique Actuals Ups Sum:", all_realizations_unique_actuals_ups_regular_UNIQUE_SUM)
    # print("Total Unique Up Preds Regular:", len(all_realizations_unique_actuals_ups_regular_UNIQUE))

    # print("ALL Realizations Threshold Unique Actuals Ups:", all_realizations_unique_actuals_ups_threshold_UNIQUE)
    # print("ALL Realizations Threshold Unique Actuals Ups Sum:", all_realizations_unique_actuals_ups_threshold_UNIQUE_SUM)
    # print("Total Unique Up Preds Threshold:", len(all_realizations_unique_actuals_ups_threshold_UNIQUE))

    output = {
    "all_realizations_unique_actuals_ups_regular_UNIQUE": all_realizations_unique_actuals_ups_regular_UNIQUE,
    "all_realizations_unique_actuals_ups_regular_UNIQUE_SUM": all_realizations_unique_actuals_ups_regular_UNIQUE_SUM,
    "LEN_all_realizations_unique_actuals_ups_regular_UNIQUE": len(all_realizations_unique_actuals_ups_regular_UNIQUE),

    "all_realizations_unique_actuals_ups_threshold_UNIQUE": all_realizations_unique_actuals_ups_threshold_UNIQUE,
    "all_realizations_unique_actuals_ups_threshold_UNIQUE_SUM": all_realizations_unique_actuals_ups_threshold_UNIQUE_SUM,
    "LEN_all_realizations_unique_actuals_ups_threshold_UNIQUE": len(all_realizations_unique_actuals_ups_threshold_UNIQUE)
}

    return output




#####################################################################################################################################
### versions used in the (1) file for testing --- they call all be consolidated with teh main functon ssince the changes are minimal 
#####################################################################################################################################

## NOTE the only difference is that thsi vcerison incldues the number of chosen mdoels and seeds as an input 
def collect_V_T_set_FULLraw_data_TESTING(models_selected, results_dist_disc, results_dist_disc_Tset_same_seeds_organized , num_models = 90, num_seeds_per_model = 1):
    T_set_data = [] 
    V_set_data = []

    # select 5 models at random from the selected models
    models_RAND = random.sample(models_selected, min(num_models, len(models_selected)))

    for model_entry in models_RAND:
        model_seeds = []
        ff_combo_idx = model_entry['combo_index']
        for seed_entry in model_entry['selected_seeds']:
            seed_num = seed_entry['seed_num']
            model_seeds.append(seed_num)

        ##### Find randomly chosen seeds in the V and T raw data 
        for raw_data_entry_V in results_dist_disc:
            V_combo_idx = raw_data_entry_V["combo_index"]
            if ff_combo_idx == V_combo_idx:
                rand_chosen_seeds = random.sample(model_seeds, min(num_seeds_per_model, len(model_seeds)))
                for seed_chosen in rand_chosen_seeds:
                    for seed_raw in raw_data_entry_V['per_seed_all_results']:
                        if seed_raw['seed'] == seed_chosen:
                            V_set_data.append(seed_raw['result_entry'])

        for raw_data_entry_T in results_dist_disc_Tset_same_seeds_organized:
            T_combo_idx = raw_data_entry_T["combo_index"]
            if ff_combo_idx == T_combo_idx:
                for seed_chosen in rand_chosen_seeds:
                    for seed_raw in raw_data_entry_T['per_seed_all_results']:
                        if seed_raw['seed'] == seed_chosen:
                            T_set_data.append(seed_raw['result_entry'])

    return {"V_set_data": V_set_data, "T_set_data": T_set_data}




## NOTE NOTE this version is used in the (1) file for testing, the only difference is that this includes the master df in the output, the fucntions could simply be consolidated as they are the same otherwise
def process_func_PLUS_return_analytics_THRESH_var_included_TESTING( #master_df: pd.DataFrame, 
    
    
                                    ## new params
                                    threshold : float,
                                    names_all: dict,

                                    #new arams 
                                
                                    groups_config: list, 
                                    data_type_corr_groups_creation: str = "V",
                                    data_type_ensemble: str = "T",
                                    use_threshold_data: bool = False, 
                                    seed = None, 
                                    num_maps_per_group: int = 10 , 
                                    filter_outliers: bool = False,
                                    use_spearman_corr: bool = False,

                                    use_corr_with_diff_params: bool =True, 
                                    min_diff_params : int = 0,
                                    
                                    use_corr_with_same_up_preds_and_fps: bool = False,
                                    min_same_up_preds: int = None,
                                    max_same_up_preds: int = None,
                                    max_same_up_fps: int = None,
                                    use_same_up_fps_for_corr: bool = False,

                                    ) -> dict:
    



    for data in names_all.values():
        add_threshold_metrics(data, threshold = threshold)

    if len(names_all) == 4 :
 
        df_mac_L_val = flatten_results(names_all["res_mac_L_val"])
        df_mac_H_val = flatten_results(names_all["res_mac_H_val"])
        df_mac_L_test = flatten_results(names_all["res_mac_L_test"])
        df_mac_H_test = flatten_results(names_all["res_mac_H_test"])

                # Add H/L labels
        df_mac_L_val['H_L'] = 'L'
        df_mac_H_val['H_L'] = 'H'
        df_mac_L_test['H_L'] = 'L'
        df_mac_H_test['H_L'] = 'H'

        dfs = [df_mac_L_val, df_mac_H_val, df_mac_L_test, df_mac_H_test]


    if len(names_all) == 2 : ### use the names below for the top combos version 

        df_mac_H_val = flatten_results(names_all["res_mac_H_val"])
        df_mac_H_test = flatten_results(names_all["res_mac_H_test"])
        
                # Add H/L labels
        df_mac_H_val['H_L'] = 'H'
        df_mac_H_test['H_L'] = 'H'

        dfs = [ df_mac_H_val, df_mac_H_test]



    # Add various metrics
    add_up_prediction_counts(dfs)
    add_false_correct_up_stats(dfs)
    flatten_metrics_columns(dfs)

    master_df = process_parameters_and_merge(dfs)

    # Set random seed
    if seed is not None:
        random.seed(seed)
    

    # Create groups and random samples
    all_pair_maps = {}
    groups_data_for_output = {} #### NEW NEW NEW
    
    for machine, corr_range, group_size in groups_config:
        # Create group name based on parameters
        group_name = f"pair_{group_size}_{corr_range[0]}_{corr_range[1]}".replace('.', '').replace('-', 'neg')
        
        # Get model groups
        if use_corr_with_diff_params:
            groups = get_model_groups_in_corr_range_diff_params(master_df, machine, data_type_corr_groups_creation, corr_range, group_size=group_size, use_spearman_bool=use_spearman_corr, num_diff_params=min_diff_params)

        elif use_corr_with_same_up_preds_and_fps:
            groups = get_model_groups_in_corr_range_diff_params_and_same_up_preds_and_fps(master_df, machine, data_type_corr_groups_creation, corr_range, group_size=group_size, use_spearman_bool=use_spearman_corr , num_diff_params=None, min_same_up_preds=min_same_up_preds, max_same_up_preds=max_same_up_preds , use_same_up_fps=use_same_up_fps_for_corr, max_same_up_fps= max_same_up_fps)
        
        else:
            groups = get_model_groups_in_corr_range(master_df, machine, data_type_corr_groups_creation, corr_range, group_size=group_size, use_spearman_bool=use_spearman_corr , num_diff_params=None)
    

        groups_data_for_output[group_name] = {}  #### NEW NEW NEW
        groups_data_for_output[group_name]['all_groups'] = groups  #### NEW NEW NEW
        groups_data_for_output[group_name]['num_groups'] = len(groups)  #### NEW NEW
        
        # Randomly select from each group
        random_groups = random.sample(groups, min(num_maps_per_group, len(groups))) if groups else []
        
        # Create parameter map for both V and T data
        pair_params_map_V = create_pair_params_map(random_groups, master_df, num_models=group_size, data_type="V")
        pair_params_map_T = create_pair_params_map(random_groups, master_df, num_models=group_size, data_type="T")
        
        # Store both versions
        all_pair_maps[group_name] = {
            "V": pair_params_map_V,
            "T": pair_params_map_T,
            "random_groups": random_groups
        }
    
    # Create ensemble configuration using the specified data type
# Create ensemble configuration using the specified data type
    ensemble_config = []
    for group_name, data in all_pair_maps.items():
        group = data.get(data_type_ensemble, {})          # safe: may be missing/empty

        first = next(iter(group.values()), None)          # safe: returns None if empty

        if not first or "parameters" not in first:
            print(f"[skip] {group_name}: no items or missing 'parameters' for '{data_type_ensemble}'")
            continue

        group_size = len(first["parameters"])
        print(f"[use] {group_name}: {len(group)} ensembles of size {group_size} for '{data_type_ensemble}'")
        
        ensemble_config.append((group_name, group, group_size))

    # Process the data
    ensemble_results = process_ensemble_groups(
        ensemble_config=ensemble_config,
        master_df=master_df,
        data_type=data_type_ensemble,
        use_threshold_data=use_threshold_data
    )

    # Initialize counters
    total_ups = 0
    total_correct_ups = 0
    # sum_actuals_ups = 0
    sum_actuals_ups_list_ALL = []


    group_results = {}
    detailed_results = {}



    # Process each group
    for ensemble_group_name, ensemble_group in ensemble_results.items():
        group_output = []
        group_ups = 0
        group_correct_ups = 0
        group_sum_actuals = 0
        group_details = []
        
        # Iterate through each ensemble result in the group
        for ensemble_result in ensemble_group:
            flatten_preds = [p for part in ensemble_result["all_preds"] for p in part]
            flatten_raw_actuals = [a for part in ensemble_result["raw_actuals"] for a in part]
            flatten_actuals = [a for part in ensemble_result["all_actuals"] for a in part]

            p_ups = sum(1 for i in flatten_preds if not isinstance(i, str) and i > 0.5)
            group_ups += p_ups
            total_ups += p_ups

            up_vals_predicted_raw_val = []

            correct_ups = 0

            for p, a, actual_binary in zip(flatten_preds, flatten_raw_actuals, flatten_actuals):
                if not isinstance(p, str) and p > 0.5:
                    up_vals_predicted_raw_val.append(a)


                    group_sum_actuals += a
                    sum_actuals_ups_list_ALL.append(a)

                    # sum_actuals_ups += a
                    # Check if up prediction was correct (actual_bin > 0.5)
                    if actual_binary > 0.5:
                        correct_ups += 1
            
            group_correct_ups += correct_ups
            total_correct_ups += correct_ups

            # Create output line
            output_line = (
                f"{ensemble_result['cv_sets']['overall_metrics']['precision_up']} - "
                f"{ensemble_result['cv_sets']['overall_metrics']['recall_up']} "
                f"{ensemble_result['combo_numbers']} - {p_ups} - Correct: {correct_ups} - {up_vals_predicted_raw_val} "
            )
            group_output.append(output_line)
            
            # Store detailed results
            group_details.append({
                "combo_numbers": ensemble_result["combo_numbers"],
                "precision_up": ensemble_result["cv_sets"]["overall_metrics"]["precision_up"],
                "recall_up": ensemble_result["cv_sets"]["overall_metrics"]["recall_up"],
                "up_predictions": p_ups,
                "correct_ups": correct_ups,
                "actual_returns": up_vals_predicted_raw_val,

            })
        
        group_results[ensemble_group_name] = group_output
        detailed_results[ensemble_group_name] = group_details
        
        # Add group summary
        group_results[f"{ensemble_group_name}_summary"] = [
            f"Total Up Predictions: {group_ups}",
            f"Total Correct Up Predictions: {group_correct_ups}",
            f"Sum of Actual Returns for Up Predictions: {group_sum_actuals:.3f}",
            f"Prec Up: {group_correct_ups/group_ups if group_ups > 0 else 0:.3f}"
        ]

    # ## set outlier values past < -0.5 to -0.06
    # if filter_outliers:
    #     for i in range(len(sum_actuals_ups_list_ALL)):
    #         if sum_actuals_ups_list_ALL[i] < -0.5:
    #             sum_actuals_ups_list_ALL[i] = -0.1

    total_ups = len(set(sum_actuals_ups_list_ALL))
    total_correct_ups = len(set([val for val in sum_actuals_ups_list_ALL if val > 0.1]))  

    sum_actuals_ups = sum(set(sum_actuals_ups_list_ALL))



    # Create overall summary
    summary = {
        "Total Up Predictions": total_ups,
        "Total Correct Up Predictions": total_correct_ups,
        "Sum of Actual Returns for Up Predictions": sum_actuals_ups,
        "Unique Actual Returns for Up Predictions": set(sum_actuals_ups_list_ALL),
        "Prec Up": total_correct_ups/total_ups if total_ups > 0 else 0
    }

    # Return structured results
    return {
        "group_results": group_results,
        "detailed_results": detailed_results,
        "summary": summary,
        "total_metrics": {
            "total_ups": total_ups,
            "total_correct_ups": total_correct_ups,
            "sum_actuals_ups": sum_actuals_ups,
            "unique_actuals_ups_list": set(sum_actuals_ups_list_ALL),

            "precision": total_correct_ups / total_ups if total_ups > 0 else 0
        },


        "config_info": {
            "groups_config": groups_config,
            "data_type_corr_groups_creation": data_type_corr_groups_creation,
            "data_type_ensemble": data_type_ensemble,
            "use_threshold_data": use_threshold_data,
            "seed": seed,
            "num_maps_per_group": num_maps_per_group
        },


       "groups_data": groups_data_for_output ,
       "master_df" : master_df

    }




### NOTE NOTE this version inlcudes additional outputs, thsi is the only difference between this adn the non TESTING version
def process_and_RETURN_analytics_2_3_Model_Performance_TESTING(V_set , T_set , 

                                num_realizations = 100,
                                 
                                 threshold = 0.9, 
                                 min_diff_params = 2,

                                 min_same_up_preds = None ,
                                 max_same_up_preds = None ,

                                 num_maps_per_group = 1,

                                max_same_up_fps = 0,
                                 use_same_up_fps_for_corr = True,

                                 do_print = False):
    
    if V_set == [] or T_set == []:
        output = {
                            "all_realizations_unique_actuals_ups_regular_UNIQUE": 0,
                            "all_realizations_unique_actuals_ups_regular_UNIQUE_SUM": 0,
                            "LEN_all_realizations_unique_actuals_ups_regular_UNIQUE": 0,

                            "all_realizations_unique_actuals_ups_threshold_UNIQUE": 0,
                            "all_realizations_unique_actuals_ups_threshold_UNIQUE_SUM": 0,
                            "LEN_all_realizations_unique_actuals_ups_threshold_UNIQUE": 0
                        }
        return output

    names_all = { 'res_mac_H_val': V_set, 
                'res_mac_H_test': T_set}

    # Define your groups configuration
    groups_config = [
        ("mac",  (-1, 1), 2),
        # ("mac",  (0.1, 0.2), 2),
        # ("mac",  (0.3, 0.5), 2),
        # ("mac",  (-1, 0.3), 3),
        # ("mac",  (0.1, 0.3), 3),
        # ("mac",  (0.3, 0.5), 3)
    ]

    number_realizations = num_realizations

    all_results_regular = []
    all_results_threshold = []

    # random_seeds_seed = 4
    # random.seed(random_seeds_seed)

    sample_size = number_realizations
    seeds = random.sample(range(1, 100000), sample_size)

    for i, seed in zip(range(number_realizations), seeds):

        results = process_func_PLUS_return_analytics_THRESH_var_included_TESTING (
                                            ## new params
                                        threshold = threshold,
                                        names_all = names_all,

                                        #new arams

                                        groups_config = groups_config,
                                        data_type_corr_groups_creation = "V",
                                        data_type_ensemble = "T",
                                        use_threshold_data = False,
                                        seed = None,
                                        num_maps_per_group = num_maps_per_group,
                                        filter_outliers = False ,#### NOTICE NOTICE
                                        use_corr_with_diff_params = False,

                                        min_diff_params = min_diff_params , 
                                        use_spearman_corr = False,
                                        
                                        use_corr_with_same_up_preds_and_fps = True,
                                        min_same_up_preds = min_same_up_preds,
                                        max_same_up_preds = max_same_up_preds ,

                                        use_same_up_fps_for_corr= use_same_up_fps_for_corr,
                                        max_same_up_fps = max_same_up_fps

                                        )
        all_results_regular.append(results)


    for i , seed in zip(range(number_realizations), seeds):

        results_THR = process_func_PLUS_return_analytics_THRESH_var_included_TESTING (
                                            ## new params
                                        threshold = threshold,
                                        names_all = names_all,

                                        #new arams

                                        groups_config = groups_config,
                                        data_type_corr_groups_creation = "V",
                                        data_type_ensemble = "T",
                                        use_threshold_data = True,
                                        seed = None,
                                        num_maps_per_group = num_maps_per_group,
                                        filter_outliers = False, #### NOTICE NOTICE
                                        use_corr_with_diff_params = False,

                                        min_diff_params = min_diff_params, 
                                        use_spearman_corr = False,
                                        
                                        use_corr_with_same_up_preds_and_fps = True,
                                        min_same_up_preds = min_same_up_preds,
                                        max_same_up_preds = max_same_up_preds,


                                        use_same_up_fps_for_corr= use_same_up_fps_for_corr,
                                        max_same_up_fps = max_same_up_fps

                                        )

        all_results_threshold.append(results_THR)

    all_groups_prec_up_regular = []
    all_groups_total_return_regular = []
    no_up_preds_per_group_regular = []
    all_realizations_unique_actuals_ups_regular = []



            # "Sum of HOD Actual Returns for Up Predictions": HOD_sum_actuals_ups,
            # "Sum of UCO Actual Returns for Up Predictions": UCO_sum_actuals_ups,
            # "Sum of HUC Actual Returns for Up Predictions": HUC_sum_actuals_ups,

    for seed_res in all_results_regular:
        #first set of plots 
        all_groups_total_return_regular.append(seed_res["summary"]["Sum of Actual Returns for Up Predictions"])
        all_groups_prec_up_regular.append(seed_res["summary"]['Prec Up'])
        no_up_preds_per_group_regular.append(seed_res["summary"]['Total Up Predictions'])
        all_realizations_unique_actuals_ups_regular.append(seed_res["summary"]['Unique Actual Returns for Up Predictions'])


    all_realizations_unique_actuals_ups_regular_UNIQUE = set([item for sublist in all_realizations_unique_actuals_ups_regular for item in sublist])
    all_realizations_unique_actuals_ups_regular_UNIQUE_SUM = sum(all_realizations_unique_actuals_ups_regular_UNIQUE)

    all_groups_prec_up_threshold = []
    all_groups_total_return_threshold = []
    no_up_preds_per_group_threshold = []
    all_realizations_unique_actuals_ups_threshold = []



    for seed_res in all_results_threshold:
        all_groups_total_return_threshold.append(seed_res["summary"]["Sum of Actual Returns for Up Predictions"])
        all_groups_prec_up_threshold.append(seed_res["summary"]['Prec Up'])
        no_up_preds_per_group_threshold.append(seed_res["summary"]['Total Up Predictions'])
        all_realizations_unique_actuals_ups_threshold.append(seed_res["summary"]['Unique Actual Returns for Up Predictions'])


    all_realizations_unique_actuals_ups_threshold_UNIQUE = set([item for sublist in all_realizations_unique_actuals_ups_threshold for item in sublist])
    all_realizations_unique_actuals_ups_threshold_UNIQUE_SUM = sum(all_realizations_unique_actuals_ups_threshold_UNIQUE)


    import matplotlib.pyplot as plt

    # List of variable names and titles
    metrics = [
        
        (all_groups_total_return_regular, 'Sum of Actual Returns (Regular)', 'blue'),
        (all_groups_prec_up_regular, 'Precision Up (Regular)', 'blue'),
        (no_up_preds_per_group_regular, 'No Up Predictions (Regular)', 'blue'),

        (all_groups_total_return_threshold, 'Sum of Actual Returns (Threshold)', 'red'),
        (all_groups_prec_up_threshold, 'Precision Up (Threshold)', 'red'),
        (no_up_preds_per_group_threshold, 'No Up Predictions (Threshold)', 'red'),

    ]

    if do_print:
        plt.figure(figsize=(20, 19))

        for i, (var_name, title, color) in enumerate(metrics, 1):

            plt.subplot(6, 3, i)
            data = var_name  # Get the variable by name
            plt.hist(data, bins=15, alpha=0.7, label=title, color=color)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {title}')
            plt.legend()

        plt.tight_layout()
        plt.show()


        print(f"\n=== SUMMARY STATISTICS REGULAR ===")
        print(f"Precision Up - Mean: {np.mean(all_groups_prec_up_regular):.2f}%, Std: {np.std(all_groups_prec_up_regular):.2f}%")
        print(f"Sum Returns - Mean: {np.mean(all_groups_total_return_regular):.4f}, Std: {np.std(all_groups_total_return_regular):.4f}")
        print(f"Total Up Preds - Mean: {np.mean(no_up_preds_per_group_regular):.1f}, Std: {np.std(no_up_preds_per_group_regular):.1f}")
        print(f"Unique Returns Across All Realizations: {len(all_realizations_unique_actuals_ups_regular_UNIQUE)}")
        print(f"Sum of Unique Returns: {all_realizations_unique_actuals_ups_regular_UNIQUE_SUM:.4f}")

        print(f"\n=== SUMMARY STATISTICS THRESHOLD ===")
        print(f"Precision Up - Mean: {np.mean(all_groups_prec_up_threshold):.2f}%, Std: {np.std(all_groups_prec_up_threshold):.2f}%")
        print(f"Sum Returns - Mean: {np.mean(all_groups_total_return_threshold):.4f}, Std: {np.std(all_groups_total_return_threshold):.4f}")
        print(f"Total Up Preds - Mean: {np.mean(no_up_preds_per_group_threshold):.1f}, Std: {np.std(no_up_preds_per_group_threshold):.1f}")
        print(f"Unique Returns Across All Realizations: {len(all_realizations_unique_actuals_ups_threshold_UNIQUE)}")
        print(f"Sum of Unique Returns: {all_realizations_unique_actuals_ups_threshold_UNIQUE_SUM:.4f}")



    # print("ALL Realizations Regular Unique Actuals Ups:", all_realizations_unique_actuals_ups_regular_UNIQUE)
    # print("ALL Realizations Regular Unique Actuals Ups Sum:", all_realizations_unique_actuals_ups_regular_UNIQUE_SUM)
    # print("Total Unique Up Preds Regular:", len(all_realizations_unique_actuals_ups_regular_UNIQUE))

    # print("ALL Realizations Threshold Unique Actuals Ups:", all_realizations_unique_actuals_ups_threshold_UNIQUE)
    # print("ALL Realizations Threshold Unique Actuals Ups Sum:", all_realizations_unique_actuals_ups_threshold_UNIQUE_SUM)
    # print("Total Unique Up Preds Threshold:", len(all_realizations_unique_actuals_ups_threshold_UNIQUE))

    output = {
    "all_realizations_unique_actuals_ups_regular_UNIQUE": all_realizations_unique_actuals_ups_regular_UNIQUE,
    "all_realizations_unique_actuals_ups_regular_UNIQUE_SUM": all_realizations_unique_actuals_ups_regular_UNIQUE_SUM,
    "LEN_all_realizations_unique_actuals_ups_regular_UNIQUE": len(all_realizations_unique_actuals_ups_regular_UNIQUE),

    "all_realizations_unique_actuals_ups_threshold_UNIQUE": all_realizations_unique_actuals_ups_threshold_UNIQUE,
    "all_realizations_unique_actuals_ups_threshold_UNIQUE_SUM": all_realizations_unique_actuals_ups_threshold_UNIQUE_SUM,
    "LEN_all_realizations_unique_actuals_ups_threshold_UNIQUE": len(all_realizations_unique_actuals_ups_threshold_UNIQUE)
}

    return output ,all_results_regular, all_results_threshold





def analyze_random_models_all_ups(master_df, num_models=2, realization_seed=None):
    """
    Pick N random models from master_df and take ALL unique up predictions from all.
    No ensemble logic, no thresholds - just combine all up preds using set().
    
    Args:
        master_df: DataFrame with model results
        num_models: Number of models to randomly select (default: 2)
        realization_seed: Random seed for reproducibility
    
    Returns:
        dict with precision, total returns, and prediction counts
    """
    if realization_seed is not None:
        random.seed(realization_seed)
    
    # Get all available model IDs from T-set (test data)
    available_models = master_df[master_df['param_int_Vue_mac_T'].notna()]
    
    if len(available_models) < num_models:
        return None
        
    # Randomly select N models
    selected_models = available_models.sample(num_models)
    
    # Collect ALL up predictions with their data
    all_up_predictions = []
    
    # Loop through all selected models
    for model_idx in range(num_models):
        model_preds = selected_models.iloc[model_idx]['all_preds_mac_T']
        model_actuals = selected_models.iloc[model_idx]['all_actuals_mac_T'] 
        model_raw_actuals = selected_models.iloc[model_idx]['raw_actuals_mac_T']
        
        # Flatten predictions and actuals for this model
        flat_preds = [p for fold in model_preds for p in fold]
        flat_actuals = [a for fold in model_actuals for a in fold]
        flat_raw_actuals = [a for fold in model_raw_actuals for a in fold]
        
        # Find indices where prediction > 0.5 and collect data
        for i, pred in enumerate(flat_preds):
            if pred > 0.5:
                all_up_predictions.append({
                    'index': i,
                    'actual_return': flat_raw_actuals[i],
                    'binary_actual': flat_actuals[i],
                    'model': f'model{model_idx+1}'
                })
    
    # Remove duplicates based on actual return values (since same indices might have same returns)
    unique_returns_seen = set()
    unique_predictions = []
    
    for pred_data in all_up_predictions:
        return_val = pred_data['actual_return']
        if return_val not in unique_returns_seen:
            unique_returns_seen.add(return_val)
            unique_predictions.append(pred_data)
    
    # Calculate metrics
    total_up_count = len(unique_predictions)
    correct_up_count = sum(1 for pred in unique_predictions if pred['binary_actual'] > 0.5)
    
    precision_up = (correct_up_count / total_up_count * 100) if total_up_count > 0 else 0
    sum_actual_returns = sum(pred['actual_return'] for pred in unique_predictions)
    unique_actual_returns = set(pred['actual_return'] for pred in unique_predictions)
    
    return {
        "precision_up": precision_up,
        "sum_actual_returns": sum_actual_returns, 
        "total_up_predictions": total_up_count,
        "correct_up_predictions": correct_up_count,
        "unique_actual_returns": unique_actual_returns
    }

def run_random_model_analysis(master_df, num_models=2, number_realizations=20):
    """
    Run multiple realizations of random model analysis.
    
    Args:
        master_df: DataFrame with model results
        num_models: Number of models to randomly select per realization
        number_realizations: Number of realizations to run
    
    Returns:
        dict with all results and statistics
    """
    all_precision_ups = []
    all_sum_returns = [] 
    all_total_up_preds = []
    all_unique_returns_per_realization = []

    for realization in range(number_realizations):
        result = analyze_random_models_all_ups(master_df, num_models=num_models, realization_seed=realization+1)
        
        if result is not None:
            all_precision_ups.append(result["precision_up"])
            all_sum_returns.append(result["sum_actual_returns"])
            all_total_up_preds.append(result["total_up_predictions"])
            all_unique_returns_per_realization.append(result["unique_actual_returns"])

    # Combine all unique returns across ALL realizations
    all_realizations_unique_returns = set()
    for realization_returns in all_unique_returns_per_realization:
        all_realizations_unique_returns.update(realization_returns)

    all_realizations_unique_returns_sum = sum(all_realizations_unique_returns)

    print(f"\nCompleted {len(all_precision_ups)} realizations with {num_models} models each")
    print(f"Total unique returns across all realizations: {len(all_realizations_unique_returns)}")


    metrics = [
        (all_sum_returns, f'Sum of Actual Returns (Random {num_models}-Model All Ups)', 'green'),
        (all_precision_ups, f'Precision Up (Random {num_models}-Model All Ups)', 'green'), 
        (all_total_up_preds, f'Total Up Predictions (Random {num_models}-Model All Ups)', 'green'),
    ]

    plt.figure(figsize=(20, 6))

    for i, (data, title, color) in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        plt.hist(data, bins=15, alpha=0.7, label=title, color=color)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {title}')
        plt.legend()

    plt.tight_layout()
    plt.show()


    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Precision Up - Mean: {np.mean(all_precision_ups):.2f}%, Std: {np.std(all_precision_ups):.2f}%")
    print(f"Sum Returns - Mean: {np.mean(all_sum_returns):.4f}, Std: {np.std(all_sum_returns):.4f}")
    print(f"Total Up Preds - Mean: {np.mean(all_total_up_preds):.1f}, Std: {np.std(all_total_up_preds):.1f}")
    print(f"Unique Returns Across All Realizations: {len(all_realizations_unique_returns)}")
    print(f"Sum of Unique Returns: {all_realizations_unique_returns_sum:.4f}")
    
    return {
        'precision_ups': all_precision_ups,
        'sum_returns': all_sum_returns,
        'total_up_preds': all_total_up_preds,
        'unique_returns_per_realization': all_unique_returns_per_realization,
        'all_realizations_unique_returns': all_realizations_unique_returns,
        'all_realizations_unique_returns_sum': all_realizations_unique_returns_sum
    }