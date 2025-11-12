# ===================================================================================================================
#                                 Functions for (1) Initial Distribution Analytics HOD
# ===================================================================================================================
# Moved from notebook to keep code organized and reusable

import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import defaultdict
import json



def organize_dist_disc_dict(file_name):

    with open(file_name, "r") as f:
        results_dist_disc_Vset_NOTsorted_same_seeds = json.load(f)

    results_dist_disc_Vset_NOTsorted_same_seeds_copy = copy.deepcopy(results_dist_disc_Vset_NOTsorted_same_seeds)
    results_dist_disc = []
    ignore_find_combo = ['seed_num']

    combo_idxs = np.arange(len(results_dist_disc_Vset_NOTsorted_same_seeds_copy) + 1).tolist()
    combo_idxs_counter = 0

    for idx ,entry in enumerate(results_dist_disc_Vset_NOTsorted_same_seeds_copy):
        organized_results_entry = {} ## entry for results_dist_disc
        params_included = [e['combo'] for e in results_dist_disc] ## for each entry, there is a combo , these are unique
        params_included_keys = [str({k: v for k, v in c.items() if k not in ignore_find_combo}).replace(' ', '').replace("'", '').replace('  ', '').replace(',', '') for c in params_included]
        current_param_keys = str({k: v for k, v in entry['parameters'].items() if k not in ignore_find_combo}).replace(' ', '').replace("'", '').replace('  ', '').replace(',', '')
        if current_param_keys not in params_included_keys:
            combo_idxs_counter += 1
            organized_results_entry["combo_index"] = combo_idxs[combo_idxs_counter]
            organized_results_entry["combo"] = entry["parameters"]
            organized_results_entry["per_seed_all_results"] = []
            for params_for_combo in results_dist_disc_Vset_NOTsorted_same_seeds_copy:
                seed_results_entry = {}
                if str({k: v for k, v in params_for_combo['parameters'].items() if k not in ignore_find_combo}).replace(' ', '').replace("'", '').replace('  ', '').replace(',', '') == current_param_keys:
                    seed_results_entry["seed"] = params_for_combo["parameters"]["seed_num"]
                    seed_results_entry["result_entry"] = params_for_combo
                    organized_results_entry["per_seed_all_results"].append(seed_results_entry)
            results_dist_disc.append(organized_results_entry)

    return results_dist_disc


def organize_dist_disc_Tset_dict(file_name_Tset , results_dist_disc):

    with open(file_name_Tset) as f:
        results_dist_disc_Tset_NOTsorted_same_seeds = json.load(f)

    ##### Organizing the results to have same seeds and combos together --- since params were extracted themselves for GC run simplification 
    results_dist_disc_Tset_same_seeds = copy.deepcopy(results_dist_disc_Tset_NOTsorted_same_seeds)

    ignore_find_combo = ['val_start_month', 'val_end_month'  ]
    ### first collect combo_idxs and combo outer info
    results_dist_disc_Tset_same_seeds_organized = []
    for true in results_dist_disc: ### make sure the combo idx lines up, parallel_backend runs could vary since its the idx when func is called in parallel 
        t_cb_idx = true["combo_index"]

        # same_seed_and_combo = []
        dict_entry = {}
        tt = str({k: v for k, v in true['combo'].items() if k not in ignore_find_combo}).replace(' ', '').replace("'", '').replace('  ', '').replace(',', '')

        for find in results_dist_disc_Tset_same_seeds:

            ff = str({k: v for k, v in find['parameters'].items() if k not in ignore_find_combo}).replace(' ', '').replace("'", '').replace('  ', '').replace(',', '')
            if tt == ff:
                dict_entry["combo_index"] = t_cb_idx
                dict_entry["combo"] = find["parameters"]
                results_dist_disc_Tset_same_seeds_organized.append(dict_entry)
                break
            

    ignore_find_seeds = ['val_start_month', 'val_end_month'  , 'seed_num'   ]

    ### then collect per seed results for each combo , combos are
    for entry in results_dist_disc_Tset_same_seeds_organized:
        entry["per_seed_all_results"] = []

        ee = str({k: v for k, v in entry['combo'].items() if k not in ignore_find_seeds}).replace(' ', '').replace("'", '').replace('  ', '').replace(',', '')

        for find in results_dist_disc_Tset_same_seeds:
            ff = str({k: v for k, v in find['parameters'].items() if k not in ignore_find_seeds}).replace(' ', '').replace("'", '').replace('  ', '').replace(',', '')

            if ee == ff:
                seed_entry = {}
                # print(find["parameters"]["seed_num"])
                seed_entry["seed"] = find["parameters"]["seed_num"]
                seed_entry["result_entry"] = find
                entry["per_seed_all_results"].append(seed_entry)

    return results_dist_disc_Tset_same_seeds_organized


def create_model_metrics_dist_plots_inital_exploration(selected_models):
    # mean_precision
        
    plt.figure(figsize=(8, 2))
    mean_precision_list = [model['mean_precision'] for model in selected_models]
    zero_count = sum(1 for mp in mean_precision_list if mp == 0)
    print(f'Number of models with mean precision of 0: {zero_count}')
    Mp_1q = np.percentile(mean_precision_list, 35)
    Mp_2q = np.percentile(mean_precision_list, 85)
    plt.hist(mean_precision_list, bins=30)

    ax = plt.gca()              
    x_min, x_max = ax.get_xlim() ; y_min, y_max = ax.get_ylim()

    plt.text(Mp_1q + 1, y_max * 0.7, f'{Mp_1q:.2f}', rotation=90)
    plt.text(Mp_2q + 1, y_max * 0.7, f'{Mp_2q:.2f}', rotation=90)
    plt.axvline(x=Mp_1q, color='k')
    plt.axvline(x=Mp_2q, color='k')
    plt.title('Model Mean Precision Distribution')
    plt.show()

    # mean_recall_up
    plt.figure(figsize=(8, 2))
    mean_recall_up_list = [model['mean_recall_up'] for model in selected_models]
    zero_count = sum(1 for mr in mean_recall_up_list if mr == 0)
    print(f'Number of models with mean recall up of 0: {zero_count}')
    Mr_1q = np.percentile(mean_recall_up_list, 35)
    Mr_2q = np.percentile(mean_recall_up_list, 85)
    plt.hist(mean_recall_up_list, bins=30)

    ax = plt.gca()
    x_min, x_max = ax.get_xlim() ; y_min, y_max = ax.get_ylim()
    plt.text(Mr_1q + 1, y_max * 0.7, f'{Mr_1q:.2f}', rotation=90)
    plt.text(Mr_2q + 1, y_max * 0.7, f'{Mr_2q:.2f}', rotation=90)
    plt.axvline(x=Mr_1q, color='k')
    plt.axvline(x=Mr_2q, color='k')
    plt.title('Model Mean Recall Up Distribution')
    plt.show()

    # ratio_difference
    plt.figure(figsize=(8, 2))
    ratio_difference_list = [model['ratio_difference'] for model in selected_models if model['ratio_difference'] is not None]
    zero_count = sum(1 for rd in ratio_difference_list if rd == 0)
    print(f'Number of models with ratio difference of 0: {zero_count}')
    Rd_1q = np.percentile(ratio_difference_list, 35)
    Rd_2q = np.percentile(ratio_difference_list, 85)
    plt.hist(ratio_difference_list, bins=30)
    ax = plt.gca()              
    x_min, x_max = ax.get_xlim() ; y_min, y_max = ax.get_ylim()
    plt.text(Rd_1q + 0.01, y_max * 0.7, f'{Rd_1q:.2f}', rotation=90)
    plt.text(Rd_2q + 0.01, y_max * 0.7, f'{Rd_2q:.2f}', rotation=90)
    plt.axvline(x=Rd_1q, color='k')
    plt.axvline(x=Rd_2q, color='k')
    plt.title('Model Ratio Difference Distribution')
    plt.show()

    # fp_severe_ratio_fps
    plt.figure(figsize=(8, 2))
    fp_severe_ratio_fps_list = [model['fp_severe_ratio_fps'] for model in selected_models if model['fp_severe_ratio_fps'] is not None]
    zero_count = sum(1 for fpr in fp_severe_ratio_fps_list if fpr == 0)
    print(f'Number of models with FP Severe Ratio FPs of 0: {zero_count}')
    Fpr_Fps_1q = np.percentile(fp_severe_ratio_fps_list, 35)
    Fpr_Fps_2q = np.percentile(fp_severe_ratio_fps_list, 85)
    plt.hist(fp_severe_ratio_fps_list, bins=30)
    ax = plt.gca()              
    x_min, x_max = ax.get_xlim() ; y_min, y_max = ax.get_ylim()
    plt.text(Fpr_Fps_1q + 0.01, 0.7 * y_max, f'{Fpr_Fps_1q:.2f}', rotation=90)
    plt.text(Fpr_Fps_2q + 0.01, 0.7 * y_max, f'{Fpr_Fps_2q:.2f}', rotation=90)
    plt.axvline(x=Fpr_Fps_1q, color='k')
    plt.axvline(x=Fpr_Fps_2q, color='k')
    plt.title('Model FP Severe Ratio FPs Distribution')
    plt.show()

    # fp_severe_ratio_fps_tps
    plt.figure(figsize=(8, 2))
    fp_severe_ratio_fps_tps_list = [model['fp_severe_ratio_fps_tps'] for model in selected_models if model['fp_severe_ratio_fps_tps'] is not None]
    zero_count = sum(1 for fpr in fp_severe_ratio_fps_tps_list if fpr == 0)
    print(f'Number of models with FP Severe Ratio FPs and TPs of 0: {zero_count}')
    Fpr_Fps_Tps_1q = np.percentile(fp_severe_ratio_fps_tps_list, 35)
    Fpr_Fps_Tps_2q = np.percentile(fp_severe_ratio_fps_tps_list, 85)
    plt.hist(fp_severe_ratio_fps_tps_list, bins=30)
    ax = plt.gca()              
    x_min, x_max = ax.get_xlim() ; y_min, y_max = ax.get_ylim()
    plt.text(Fpr_Fps_Tps_1q + 0.01, 0.7 * y_max, f'{Fpr_Fps_Tps_1q:.2f}', rotation=90)
    plt.text(Fpr_Fps_Tps_2q + 0.01, 0.7 * y_max, f'{Fpr_Fps_Tps_2q:.2f}', rotation=90)
    plt.axvline(x=Fpr_Fps_Tps_1q, color='k')
    plt.axvline(x=Fpr_Fps_Tps_2q, color='k')
    plt.title('Model FP Severe Ratio FPs and TPs Distribution')
    plt.show()

    ###########                 SEED METRICS DISTS

    # seed_precision
    plt.figure(figsize=(8, 2))
    all_seed_precisions = []
    for model in selected_models:
        for seed_result in model['selected_seeds']:
            all_seed_precisions.append(seed_result['precision'])
    sp_1q = np.percentile(all_seed_precisions, 35)
    sp_2q = np.percentile(all_seed_precisions, 85)
    zero_count = sum(1 for sp in all_seed_precisions if sp == 0)
    print(f'Number of seeds with precision of 0: {zero_count}')

    plt.hist(all_seed_precisions, bins=30)
    ax = plt.gca()              
    x_min, x_max = ax.get_xlim() ; y_min, y_max = ax.get_ylim()
    plt.text(sp_1q + 1, 0.7 * y_max, f'{sp_1q:.2f}', rotation=90)
    plt.text(sp_2q + 1, 0.7 * y_max, f'{sp_2q:.2f}', rotation=90)
    plt.axvline(x=sp_1q, color='k')
    plt.axvline(x=sp_2q, color='k')
    plt.title('Seed Precision Distribution')
    plt.show()

    # seed_recall
    all_seed_recalls = []
    for model in selected_models:
        for seed_result in model['selected_seeds']:
            all_seed_recalls.append(seed_result['recall'])
    zero_count = sum(1 for sr in all_seed_recalls if sr == 0)
    print(f'Number of seeds with recall of 0: {zero_count}')
    sr_1q = np.percentile(all_seed_recalls, 35)
    sr_2q = np.percentile(all_seed_recalls, 85)

    plt.figure(figsize=(8, 2))
    plt.hist(all_seed_recalls, bins=30)
    ax = plt.gca()              
    x_min, x_max = ax.get_xlim() ; y_min, y_max = ax.get_ylim()
    plt.text(sr_1q + 1, 0.7 * y_max, f'{sr_1q:.2f}', rotation=90)
    plt.text(sr_2q + 1, 0.7 * y_max, f'{sr_2q:.2f}', rotation=90)
    plt.axvline(x=sr_1q, color='k')
    plt.axvline(x=sr_2q, color='k')
    plt.title('Seed Recall Distribution')
    plt.show()


    # seed_min_TPminusFP_greaterEqual

    all_seed_TPminusFP=[]
    for model in selected_models:
        for seed_result in model['selected_seeds']:

            bracket_high = seed_result["seed_bracket_counts"].get("0.7-1.0", {"TP": 0, "FP": 0})
            bracket_low = seed_result["seed_bracket_counts"].get("0.5-0.7", {"TP": 0, "FP": 0})

            net_high = bracket_high["TP"] - bracket_high["FP"]
            net_low = bracket_low["TP"] - bracket_low["FP"]
            TPminusFP = net_high - net_low
            all_seed_TPminusFP.append(TPminusFP)

    zero_count = sum(1 for st in all_seed_TPminusFP if st == 0)
    print(f'Number of seeds with TP minus FP of 0: {zero_count}')
    s_TPminusFP_1q = np.percentile(all_seed_TPminusFP, 35)
    s_TPminusFP_2q = np.percentile(all_seed_TPminusFP, 85)

    plt.figure(figsize=(8, 2))
    plt.hist(all_seed_TPminusFP, bins=30)

    ax = plt.gca()              
    x_min, x_max = ax.get_xlim() ; y_min, y_max = ax.get_ylim()
    plt.text(s_TPminusFP_1q + 1, .5 * y_max, f'{s_TPminusFP_1q:.2f}', rotation=90)
    plt.text(s_TPminusFP_2q + 1, .5 * y_max, f'{s_TPminusFP_2q:.2f}', rotation=90)
    plt.axvline(x=s_TPminusFP_1q, color='k')
    plt.axvline(x=s_TPminusFP_2q, color='k')
    plt.title('Seed TP minus FP Distribution')
    plt.show()


    # max_seed_severe_FPs_high_bracket
    all_seed_severe_FPs_high_bracket = []
    for model in selected_models:
        for seed_result in model['selected_seeds']:
            all_seed_severe_FPs_high_bracket.append(seed_result['seed_severe_fp_high_bracket'])
    zero_count = sum(1 for sr in all_seed_severe_FPs_high_bracket if sr == 0)
    print(f'Number of seeds with severe FP high bracket of 0: {zero_count}')

    s_severeFPs_H_1q = np.percentile(all_seed_severe_FPs_high_bracket, 35)
    s_severeFPs_H_2q = np.percentile(all_seed_severe_FPs_high_bracket, 85)

    plt.figure(figsize=(8, 2))
    plt.hist(all_seed_severe_FPs_high_bracket, bins=30)

    ax = plt.gca()              
    x_min, x_max = ax.get_xlim() ; y_min, y_max = ax.get_ylim()

    plt.text(s_severeFPs_H_1q + 1, .5 * y_max, f'{s_severeFPs_H_1q:.2f}', rotation=90)
    plt.text(s_severeFPs_H_2q + 1, .5 * y_max, f'{s_severeFPs_H_2q:.2f}', rotation=90)
    plt.axvline(x=s_severeFPs_H_1q, color='k')
    plt.axvline(x=s_severeFPs_H_2q, color='k')
    plt.title('Seed Severe FPs in High Bracket Distribution')
    plt.show()

    # min_seed_TPs_high_bracket

    all_seed_TPs_high_bracket = []

    for model in selected_models:
        for seed_result in model['selected_seeds']:
            if "0.7-1.0" in seed_result['seed_bracket_counts']:

                all_seed_TPs_high_bracket.append(seed_result['seed_bracket_counts']["0.7-1.0"]["TP"])

    zero_count = sum(1 for st in all_seed_TPs_high_bracket if st == 0)
    print(f'Number of seeds with TP high bracket of 0: {zero_count}')
    s_TPs_H_1q = np.percentile(all_seed_TPs_high_bracket, 35)
    s_TPs_H_2q = np.percentile(all_seed_TPs_high_bracket, 85)

    plt.figure(figsize=(8, 2))
    plt.hist(all_seed_TPs_high_bracket, bins=30)
    ax = plt.gca()              
    x_min, x_max = ax.get_xlim() ; y_min, y_max = ax.get_ylim()
    plt.text(s_TPs_H_1q + 1, .5 * y_max, f'{s_TPs_H_1q:.2f}', rotation=90)
    plt.text(s_TPs_H_2q + 1, .5 * y_max, f'{s_TPs_H_2q:.2f}', rotation=90)
    plt.axvline(x=s_TPs_H_1q, color='k')
    plt.axvline(x=s_TPs_H_2q, color='k')
    plt.title('Seed TPs in High Bracket Distribution')
    plt.show()




# ===================================================================================================================
#                               PLOTTING FUNCTIONS
# ===================================================================================================================

def create_three_panel_plots(V_set_data, T_set_data, ranges, range_labels, colors, 
                           plot_title_A, plot_title_B, plot_title_C=None):
    """
    Creates three-panel analysis plots matching the user's specific pattern:
    - Figure A (or 1): V-set (V_set) distributions per range  
    - Figure B (or 2): T-set (T_set) distributions per V_set range
    - Figure C (or 3): Box plot comparison
    
    Parameters:
    -----------
    V_set_data : np.array
        The reference dataset for binning (e.g., V-set means)
    T_set_data : np.array  
        The comparison dataset (e.g., T-set means)
    ranges : list of tuples
        Range boundaries [(L1, H1), (L2, H2), ...]
    range_labels : list of str
        Labels for ranges ['0.0 - 30.0', '30.0 - 60.0', ...]
    colors : list of str
        Colors for each range ['blue', 'green', 'red']
    plot_title_A : str
        Title for first figure (V-set/V_set data distributions)
    plot_title_B : str
        Title for second figure (T-set/T_set data distributions)  
    plot_title_C : str, optional
        Title for boxplot (auto-generated if None)
    """
    
    def range_mask(arr, i, L, H):
        return ((arr >= L) & (arr < H)) if i < len(ranges) - 1 else ((arr >= L) & (arr <= H))
    
    # Clean data
    valid = np.isfinite(V_set_data) & np.isfinite(T_set_data) 
    V_set = V_set_data[valid]
    T_set = T_set_data[valid]
    
    # ---------- Figure A: V-set (V_set) distributions per range ----------
    plt.figure(figsize=(15, 5))
    for i, ((L, H), label, color) in enumerate(zip(ranges, range_labels, colors)):
        mask = range_mask(V_set, i, L, H)
        V_set_r = V_set[mask]

        ax = plt.subplot(1, 3, i + 1)
        if V_set_r.size > 0:
            plt.hist(V_set_r, bins=15, alpha=0.8, edgecolor='black')
            mean_V_set = float(np.mean(V_set_r))
            med_V_set = float(np.median(V_set_r))
            plt.axvline(mean_V_set, linestyle='--', linewidth=1, label=f'Mean: {mean_V_set:.3f}')
            plt.axvline(med_V_set, linestyle=':', linewidth=1, label=f'Median: {med_V_set:.3f}')
            plt.title(f'V-set means: {label}\n(n={V_set_r.size})')
            plt.xlabel('Mean Precision (V-set)')
            plt.ylabel('Frequency')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, f'No data in range\n{label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'V-set means: {label}')
    
    plt.suptitle(plot_title_A, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # ---------- Figure B: T-set (T_set) distributions per V_set range ----------
    plt.figure(figsize=(15, 5))
    for i, ((L, H), label, color) in enumerate(zip(ranges, range_labels, colors)):
        mask = range_mask(V_set, i, L, H)  # bin by V_SET data
        T_set_r = T_set[mask]

        ax = plt.subplot(1, 3, i + 1)
        if T_set_r.size > 0:
            plt.hist(T_set_r, bins=15, alpha=0.7, color=color, edgecolor='black')
            mean_T_set = float(np.mean(T_set_r))
            med_T_set = float(np.median(T_set_r))
            plt.axvline(mean_T_set, color='black', linestyle='--', label=f'Mean: {mean_T_set:.3f}')
            plt.axvline(med_T_set, color='orange', linestyle='--', label=f'Median: {med_T_set:.3f}')
            plt.title(f'V-set means: {label}\n(n={T_set_r.size})')
            plt.xlabel('Mean Value (T-set)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data in range\n{label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'V-set means: {label}')
    
    plt.suptitle(plot_title_B, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # ---------- Figure C: Box plot ----------  
    plt.figure(figsize=(10, 3))
    all_T_set_means, all_labels = [], []
    for i, ((L, H), label) in enumerate(zip(ranges, range_labels)):
        mask = range_mask(V_set, i, L, H)
        T_set_r = T_set[mask]
        if T_set_r.size > 0:
            all_T_set_means.append(T_set_r)
            all_labels.append(f'{label}\n(n={T_set_r.size})')
    
    if all_T_set_means:
        plt.boxplot(all_T_set_means, labels=all_labels)
        if plot_title_C:
            plt.title(plot_title_C)
        else:
            plt.title('Box Plot: T-set Mean Precision by V-set Mean Ranges')
        plt.ylabel('T_set Mean Value (T-set)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No non-empty ranges for boxplot.")
    
    # ---------- Statistics printout ----------
    print("Statistics by (V-set) Mean Range:")
    print("=" * 50)
    for i, ((L, H), label) in enumerate(zip(ranges, range_labels)):
        mask = range_mask(V_set, i, L, H)
        T_set_r = T_set[mask]
        V_set_r = V_set[mask]
        if T_set_r.size > 0:
            print(f"\nRange: {label}")
            print(f"  Count: {T_set_r.size}")
            print(f"  T-set mean  - Mean:   {np.mean(T_set_r):.4f}")
            print(f"  T-set mean  - Median: {np.median(T_set_r):.4f}")
            print(f"  T-set mean  - Std:    {np.std(T_set_r, ddof=0):.4f}")
            print(f"  T-set mean  - Min:    {np.min(T_set_r):.4f}")
            print(f"  T-set mean - Max:    {np.max(T_set_r):.4f}")
            print(f"  V-set mean - Mean:           {np.mean(V_set_r):.4f}")
        else:
            print(f"\nRange: {label} - No data")


def create_simple_three_panel_plots(V_set_data, T_set_data, ranges, range_labels, colors):
    """
    Creates the simple three-panel plots for the "Sanity check" pattern:
    - Figure 1: T-set (T_set) distributions per V_set range
    - Figure 2: V-set (V_set) distributions per V_set range  
    - Figure 3: Box plot comparison
    
    Matches the user's first example with "Extract & sanitize inputs" pattern.
    """
    
    def range_mask(arr, i, L, H):
        return ((arr >= L) & (arr < H)) if i < len(ranges) - 1 else ((arr >= L) & (arr <= H))
    
    # Clean data
    valid = np.isfinite(V_set_data) & np.isfinite(T_set_data)
    V_set = V_set_data[valid]
    T_set = T_set_data[valid]
    
    # ---------- Figure 1: T-set (T_set) distributions per V_SET-mean range ----------
    plt.figure(figsize=(15, 5))
    for i, (range_val, label, color) in enumerate(zip(ranges, range_labels, colors)):
        L, H = range_val
        mask = range_mask(V_set, i, L, H)
        T_set_r = T_set[mask]

        plt.subplot(1, 3, i + 1)
        if T_set_r.size > 0:
            plt.hist(T_set_r, bins=15, alpha=0.7, color=color, edgecolor='black')
            mean_v = float(np.mean(T_set_r))
            med_v = float(np.median(T_set_r))
            plt.axvline(mean_v, color='black', linestyle='--', label=f'Mean: {mean_v:.3f}')
            plt.axvline(med_v, color='orange', linestyle='--', label=f'Median: {med_v:.3f}')
            plt.title(f'V_set means: {label}\n(n={T_set_r.size})')
            plt.xlabel('T_set Mean Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, f'No data in range\n{label}',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'V_set means: {label}')

    plt.suptitle('Distribution of T_set Mean Values by V_set Mean Value Ranges', fontsize=16)
    plt.tight_layout()
    plt.show()

    # ---------- Figure 2: V-set (V_set) distributions per V_SET-mean range ----------
    plt.figure(figsize=(15, 5))
    for i, (range_val, label, color) in enumerate(zip(ranges, range_labels, colors)):
        L, H = range_val
        mask = range_mask(V_set, i, L, H)
        V_set_r = V_set[mask]

        plt.subplot(1, 3, i + 1)
        if V_set_r.size > 0:
            plt.hist(V_set_r, bins=15, alpha=0.8, edgecolor='black')
            mean_V_set = float(np.mean(V_set_r))
            med_V_set = float(np.median(V_set_r))
            plt.axvline(mean_V_set, linestyle='--', linewidth=1, label=f'Mean: {mean_V_set:.3f}')
            plt.axvline(med_V_set, linestyle=':', linewidth=1, label=f'Median: {med_V_set:.3f}')
            plt.title(f'V-set (V_set) means: {label}\n(n={V_set_r.size})')
            plt.xlabel('Mean Value (V-set)')
            plt.ylabel('Frequency')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3, axis='y')
        else:
            plt.text(0.5, 0.5, f'No data in range\n{label}',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'V-set (V_set) means: {label}')

    plt.suptitle('Distribution of V-set Mean Values by V_set Mean Value Ranges', fontsize=16)
    plt.tight_layout()
    plt.show()

    # ---------- Figure 3: Box plot of T_SET (T-set) by V_SET-mean ranges ----------
    plt.figure(figsize=(10, 3))
    all_T_set_means = []
    all_labels = []

    for i, (range_val, label) in enumerate(zip(ranges, range_labels)):
        L, H = range_val
        mask = range_mask(V_set, i, L, H)
        T_set_r = T_set[mask]
        if T_set_r.size > 0:
            all_T_set_means.append(T_set_r)
            all_labels.append(f'{label}\n(n={T_set_r.size})')

    if all_T_set_means:
        plt.boxplot(all_T_set_means, labels=all_labels)
        plt.title('Box Plot Comparison of T_set Mean Values by V_set Mean Ranges')
        plt.ylabel('T_set Mean Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No non-empty ranges for boxplot.")

# Import existing functions from other modules
from Functions_Model_Processing import (
    add_threshold_metrics,
    flatten_results,
    add_up_prediction_counts,
    add_false_correct_up_stats,
    flatten_metrics_columns,
    process_parameters_and_merge,
)

# -------------------------------------------------------------------------------------------------
# Core pairwise metrics extraction
# -------------------------------------------------------------------------------------------------
def get_pairwise_metrics(master_df, set_type="V"):
    """Extract pairwise metrics for both-up and both-up-FP analysis"""
    preds_col = f"all_preds_mac_{set_type}"
    actuals_col = f"all_actuals_mac_{set_type}"
    id_col = f"param_int_Vue_mac_{set_type}"

    block = master_df[[preds_col, actuals_col, id_col]].reset_index(drop=True)
    if block.empty:
        return {}

    # Flatten per-model
    all_preds_list, all_actuals_list, model_ids = [], [], []
    for _, row in block.iterrows():
        flat_preds = [p for fold in row[preds_col] for p in fold]
        flat_acts = [a for fold in row[actuals_col] for a in fold]
        all_preds_list.append(flat_preds)
        all_actuals_list.append(flat_acts)
        model_ids.append(int(row[id_col]))

    n_models = len(all_preds_list)
    
    # Calculate metrics
    both_up_counts = np.zeros((n_models, n_models), dtype=int)
    both_up_fp_counts = np.zeros((n_models, n_models), dtype=int)

    for i in range(n_models):
        for j in range(n_models):
            if i <= j:
                # Binary "up" predictions
                pi = [1 if p >= 0.5 else 0 for p in all_preds_list[i]]
                pj = [1 if p >= 0.5 else 0 for p in all_preds_list[j]]
                acts_i = all_actuals_list[i]

                both_up = sum((a == 1) and (b == 1) for a, b in zip(pi, pj))
                both_up_fp = sum((a == 1) and (b == 1) and (act < 0.5) 
                               for a, b, act in zip(pi, pj, acts_i))

                both_up_counts[i, j] = both_up_counts[j, i] = both_up
                both_up_fp_counts[i, j] = both_up_fp_counts[j, i] = both_up_fp

    # Return pair metrics
    return_data = {}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model_pair = (model_ids[i], model_ids[j])
            return_data[model_pair] = (both_up_counts[i, j], both_up_fp_counts[i, j])
    return return_data

# -------------------------------------------------------------------------------------------------
# Single realization evaluation
# -------------------------------------------------------------------------------------------------
def evaluate_realization(RR, results_dist_disc, results_dist_disc_Tset_same_seeds_organized, 
                        select_models_by_criteria, metric_type="both_up"):
    """
    Evaluate a single realization with given model selection criteria
    
    Parameters:
    - RR: model selection criteria dictionary
    - results_dist_disc: validation set results data
    - results_dist_disc_Tset_same_seeds_organized: test set results data
    - select_models_by_criteria: function to select models based on criteria
    - metric_type: "both_up" or "both_up_fp"
    """
    
    # Build master_df using existing functions
    models = select_models_by_criteria(results_dist_disc, **RR)
    if not models:
        return np.nan

    res = collect_V_T_set_FULLraw_data(models, results_dist_disc, results_dist_disc_Tset_same_seeds_organized)
    V, T = res["V_set_data"], res["T_set_data"]
    if not V or not T:
        return np.nan

    # Process data using existing functions
    for data in (V, T):
        add_threshold_metrics(data, threshold=0.7)

    df_V = flatten_results(V)
    df_T = flatten_results(T)
    dfs = [df_V, df_T]
    add_up_prediction_counts(dfs)
    add_false_correct_up_stats(dfs)
    flatten_metrics_columns(dfs)
    master_df = process_parameters_and_merge(dfs)

    # Get pairwise metrics
    corr_res_V = get_pairwise_metrics(master_df, set_type="V")
    corr_res_T = get_pairwise_metrics(master_df, set_type="T")

    # Calculate mean absolute difference
    pairs = set(corr_res_V.keys()) & set(corr_res_T.keys())
    if not pairs:
        return np.nan

    diffs = []
    idx = 0 if metric_type == "both_up" else 1  # Index for both_up or both_up_fp
    
    for pair in pairs:
        v_val = corr_res_V[pair][idx] or 0
        t_val = corr_res_T[pair][idx] or 0
        diffs.append(abs(int(t_val) - int(v_val)))
    
    return float(np.mean(diffs))

# -------------------------------------------------------------------------------------------------
# Multi-run experiment with plotting
# -------------------------------------------------------------------------------------------------
def run_experiment(model_selection_map, results_dist_disc, results_dist_disc_Tset_same_seeds_organized, 
                  select_models_by_criteria, n_runs=20, seed_start=0, metric_type="both_up"):
    """
    Run repeated experiments and plot distribution
    
    Parameters:
    - model_selection_map: criteria for model selection
    - results_dist_disc: validation set results data
    - results_dist_disc_Tset_same_seeds_organized: test set results data
    - select_models_by_criteria: function to select models based on criteria
    - n_runs: number of runs to execute
    - seed_start: starting seed value
    - metric_type: "both_up" or "both_up_fp"
    """
    diffs = []
    for k in range(n_runs):
        RR = copy.deepcopy(model_selection_map)
        RR["random_seed"] = (seed_start + k)
        diff = evaluate_realization(RR, results_dist_disc, results_dist_disc_Tset_same_seeds_organized, 
                                  select_models_by_criteria, metric_type=metric_type)
        diffs.append(diff)

    diffs = np.asarray(diffs, dtype=float)
    valid = diffs[~np.isnan(diffs)]

    # Plot results
    if valid.size:
        plt.figure(figsize=(6, 2))
        plt.hist(valid, bins=10, alpha=0.9, edgecolor="black")
        
        metric_name = "Both Up" if metric_type == "both_up" else "Both Up FP"
        plt.title(f'Distribution of Mean |Δ "{metric_name}"| per Pair over {valid.size} runs')
        plt.xlabel('Mean absolute change per pair (T - V)')
        plt.ylabel('Frequency')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"Runs: {n_runs} | Valid: {valid.size} | NaNs: {np.isnan(diffs).sum()}")
        print(f"Mean={valid.mean():.3f} | Median={np.median(valid):.3f} | Std={valid.std(ddof=0):.3f} | "
              f"Min={valid.min():.3f} | Max={valid.max():.3f}")
    else:
        print("No valid runs to plot.")

    return diffs

# -------------------------------------------------------------------------------------------------
# Data collection helper
# -------------------------------------------------------------------------------------------------
def collect_V_T_set_FULLraw_data(models_selected, results_dist_disc, results_dist_disc_Tset_same_seeds_organized):
    """
    Collect V and T set data from selected models
    
    Parameters:
    - models_selected: list of selected model entries
    - results_dist_disc: validation set results data
    - results_dist_disc_Tset_same_seeds_organized: test set results data
    """
    T_set_data = []
    V_set_data = []

    # select up to 15 models at random from the selected models
    models_RAND = random.sample(models_selected, min(15, len(models_selected)))

    for model_entry in models_RAND:
        model_seeds = []
        ff_combo_idx = model_entry['combo_index']
        for seed_entry in model_entry['selected_seeds']:
            seed_num = seed_entry['seed_num']
            model_seeds.append(seed_num)

        # Find randomly chosen seeds in the V and T raw data
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

# -------------------------------------------------------------------------------------------------
# Master DataFrame builder
# -------------------------------------------------------------------------------------------------
def build_master_df_from_model_dict(RR, results_dist_disc, results_dist_disc_Tset_same_seeds_organized, 
                                   select_models_by_criteria, threshold=0.7):
    """
    Build master DataFrame from model selection criteria
    
    Parameters:
    - RR: model selection criteria dictionary
    - results_dist_disc: validation set results data
    - results_dist_disc_Tset_same_seeds_organized: test set results data
    - select_models_by_criteria: function to select models based on criteria
    - threshold: threshold for metrics
    """
    
    models = select_models_by_criteria(results_dist_disc, **RR)
    if not models:
        return None
    res = collect_V_T_set_FULLraw_data(models, results_dist_disc, results_dist_disc_Tset_same_seeds_organized)
    V, T = res["V_set_data"], res["T_set_data"]
    if not V or not T:
        return None

    for data in (V, T):
        add_threshold_metrics(data, threshold=threshold)

    df_V = flatten_results(V)
    df_T = flatten_results(T)
    dfs = [df_V, df_T]
    add_up_prediction_counts(dfs)
    add_false_correct_up_stats(dfs)
    flatten_metrics_columns(dfs)
    master_df = process_parameters_and_merge(dfs)
    return master_df

# -------------------------------------------------------------------------------------------------
# Severity classification function
# -------------------------------------------------------------------------------------------------
def classify_severity_10_percent_pos(actual_value):
    """Classify actual values into severity categories"""
    if 0 < actual_value <= 0.05:
        return "Pos_low (0-0.05)"
    elif 0.05 < actual_value <= 0.1:
        return "Pos_high (0.05-0.1)"
    elif 0 > actual_value >= -0.05:
        return "Neg_low (-0.05) - 0"
    elif -0.05 > actual_value > -0.12:
        return "Neg_mid (-0.12) - (-0.05)"
    elif actual_value <= -0.12:
        return "Severe (<-0.12)"
    return "Other"

# -------------------------------------------------------------------------------------------------
# Enhanced pairwise extraction function with severe FP tracking
# -------------------------------------------------------------------------------------------------
def get_pairwise_metrics_with_severity(master_df, set_type="V", pred_threshold=0.5):
    """Extract pairwise metrics including severe FP tracking"""
    preds_col   = f"all_preds_mac_{set_type}"
    actuals_col = f"all_actuals_mac_{set_type}"
    raw_actuals_col = f"raw_actuals_mac_{set_type}"  # Assuming this exists
    id_col      = f"param_int_Vue_mac_{set_type}"

    block = master_df[[preds_col, actuals_col, raw_actuals_col, id_col]].reset_index(drop=True)
    if block.empty:
        raise ValueError(f"No data found for set '{set_type}'.")

    # Flatten per-model
    all_preds_list, all_actuals_list, all_raw_actuals_list, model_ids = [], [], [], []
    for _, row in block.iterrows():
        preds = row[preds_col]
        acts  = row[actuals_col]
        raw_acts = row[raw_actuals_col]
        flat_preds = [p for fold in preds for p in fold]
        flat_acts  = [a for fold in acts  for a in fold]
        flat_raw_acts = [r for fold in raw_acts for r in fold]
        all_preds_list.append(flat_preds)
        all_actuals_list.append(flat_acts)
        all_raw_actuals_list.append(flat_raw_acts)
        model_ids.append(int(row[id_col]))

    n_models = len(all_preds_list)

    # Calculate both-up, both-up-fp, and both-up-severe-fp counts
    both_up_counts = np.zeros((n_models, n_models), dtype=int)
    both_up_fp_counts = np.zeros((n_models, n_models), dtype=int)
    both_up_severe_fp_counts = np.zeros((n_models, n_models), dtype=int)

    for i in range(n_models):
        for j in range(n_models):
            if i <= j:
                # Binary "up" using pred_threshold
                pi = [1 if p >= pred_threshold else 0 for p in all_preds_list[i]]
                pj = [1 if p >= pred_threshold else 0 for p in all_preds_list[j]]

                both_up = sum((a == 1) and (b == 1) for a, b in zip(pi, pj))
                both_up_counts[i, j] = both_up_counts[j, i] = both_up

                # Get corresponding actual values for severity analysis
                acts_i = all_actuals_list[i]
                raw_acts_i = all_raw_actuals_list[i]
                
                both_up_fp = 0
                both_up_severe_fp = 0
                
                for idx, (a, b, act, raw_act) in enumerate(zip(pi, pj, acts_i, raw_acts_i)):
                    if (a == 1) and (b == 1) and (act < 0.5):
                        both_up_fp += 1
                        # Check if this is a severe false positive
                        severity = classify_severity_10_percent_pos(raw_act)
                        if severity == "Severe (<-0.12)":
                            both_up_severe_fp += 1
                
                both_up_fp_counts[i, j] = both_up_fp_counts[j, i] = both_up_fp
                both_up_severe_fp_counts[i, j] = both_up_severe_fp_counts[j, i] = both_up_severe_fp

    # Return pair metrics
    return_data = {}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model_pair = (model_ids[i], model_ids[j])
            both_up = both_up_counts[i, j]
            both_up_fp = both_up_fp_counts[i, j]
            both_up_severe_fp = both_up_severe_fp_counts[i, j]
            return_data[model_pair] = (both_up, both_up_fp, both_up_severe_fp)
    return return_data

# -------------------------------------------------------------------------------------------------
# Plot FP/TP ratio distributions side by side (original)
# -------------------------------------------------------------------------------------------------
def plot_fp_tp_ratios_side_by_side(master_df, title_prefix="", pred_threshold=0.5):
    """Plot FP/TP ratio distributions for V and T sets side by side"""
    corr_res_V = get_pairwise_metrics_with_severity(master_df, set_type="V", pred_threshold=pred_threshold)
    corr_res_T = get_pairwise_metrics_with_severity(master_df, set_type="T", pred_threshold=pred_threshold)

    def get_ratios(res):
        ratios = []
        for pair, tup in res.items():
            both_up = tup[0] or 0
            both_up_fp = tup[1] or 0
            if both_up > 0:
                ratios.append(both_up_fp / both_up)
        return np.array(ratios, dtype=float)

    rV = get_ratios(corr_res_V)
    rT = get_ratios(corr_res_T)

    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    
    # Plot V set distribution
    if rV.size:
        ax1.hist(rV, bins=10, alpha=0.9, edgecolor="black")
        ax1.set_title(f'V Set - {title_prefix}FP/TP Ratio (thr={pred_threshold})')
        ax1.set_xlabel('both_up_fp / both_up (pairs with both_up > 0)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, axis='y', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No data', transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title(f'V Set - {title_prefix}FP/TP Ratio (thr={pred_threshold})')

    # Plot T set distribution
    if rT.size:
        ax2.hist(rT, bins=10, alpha=0.9, edgecolor="black")
        ax2.set_title(f'T Set - {title_prefix}FP/TP Ratio (thr={pred_threshold})')
        ax2.set_xlabel('both_up_fp / both_up (pairs with both_up > 0)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No data', transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title(f'T Set - {title_prefix}FP/TP Ratio (thr={pred_threshold})')

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------------------------
# Plot Severe FP ratio distributions side by side
# -------------------------------------------------------------------------------------------------
def plot_severe_fp_ratios_side_by_side(master_df, title_prefix="", pred_threshold=0.5):
    """Plot severe FP ratio distributions for V and T sets side by side"""
    corr_res_V = get_pairwise_metrics_with_severity(master_df, set_type="V", pred_threshold=pred_threshold)
    corr_res_T = get_pairwise_metrics_with_severity(master_df, set_type="T", pred_threshold=pred_threshold)

    def get_severe_ratios(res):
        ratios = []
        for pair, tup in res.items():
            both_up = tup[0] or 0
            both_up_severe_fp = tup[2] or 0
            if both_up > 0:
                ratios.append(both_up_severe_fp / both_up)
        return np.array(ratios, dtype=float)

    rV = get_severe_ratios(corr_res_V)
    rT = get_severe_ratios(corr_res_T)

    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    
    # Plot V set distribution
    if rV.size:
        ax1.hist(rV, bins=10, alpha=0.9, edgecolor="black", color='red')
        ax1.set_title(f'V Set - {title_prefix}Severe FP/All Matches (thr={pred_threshold})' , fontsize=10)
        ax1.set_xlabel('both_up_severe_fp / both_up (pairs with both_up > 0)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, axis='y', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No data', transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title(f'V Set - {title_prefix}Severe FP/All Matches (thr={pred_threshold})' , fontsize=10)

    # Plot T set distribution
    if rT.size:
        ax2.hist(rT, bins=10, alpha=0.9, edgecolor="black", color='red')
        ax2.set_title(f'T Set - {title_prefix}Severe FP/All Matches (thr={pred_threshold})' , fontsize=10)
        ax2.set_xlabel('both_up_severe_fp / both_up (pairs with both_up > 0)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No data', transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title(f'T Set - {title_prefix}Severe FP/All Matches (thr={pred_threshold})' , fontsize=10)

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------------------------
# Enhanced single run to get master_df and plot both types of ratios
# -------------------------------------------------------------------------------------------------
def run_single_ratio_plot_enhanced(model_selection_map, results_dist_disc, results_dist_disc_Tset_same_seeds_organized, 
                                  select_models_by_criteria, threshold=0.7, pred_threshold=0.5, model_name=""):
    """
    Run enhanced ratio plotting for a single model selection
    
    Parameters:
    - model_selection_map: criteria for model selection
    - results_dist_disc: validation set results data
    - results_dist_disc_Tset_same_seeds_organized: test set results data
    - select_models_by_criteria: function to select models based on criteria
    - threshold: threshold for metrics
    - pred_threshold: prediction threshold
    - model_name: name for the model
    """
    # Build master_df (using existing functions)
    master_df = build_master_df_from_model_dict(model_selection_map, results_dist_disc, 
                                               results_dist_disc_Tset_same_seeds_organized, 
                                               select_models_by_criteria, threshold=threshold)
    if master_df is not None:
        prefix = f"{model_name} - " if model_name else ""
        
        # Plot original FP/TP ratios
        print(f"=== PLOTTING FP/TP RATIOS FOR {model_name} ===")
        plot_fp_tp_ratios_side_by_side(master_df, title_prefix=prefix, pred_threshold=pred_threshold)
        
        # Plot new Severe FP ratios
        print(f"=== PLOTTING SEVERE FP RATIOS FOR {model_name} ===")
        plot_severe_fp_ratios_side_by_side(master_df, title_prefix=prefix, pred_threshold=pred_threshold)
        
    else:
        print(f"Warning: No master_df generated for {model_name}")

# -------------------------------------------------------------------------------------------------
# Range masking utility function
# -------------------------------------------------------------------------------------------------
def range_mask(arr, i, L, H, ranges):
    """Create range mask for array based on index and bounds"""
    return ((arr >= L) & (arr < H)) if i < len(ranges) - 1 else ((arr >= L) & (arr <= H))


# ===================================================================================================================
#                               SEVERITY AND FP ANALYSIS PLOTTING FUNCTION
# ===================================================================================================================

def create_severity_fp_analysis_plots(tset_keep, tset_ignore, tset_flip, SEVERITY_CATS, SEVERITY_COLORS):
    """
    Create comprehensive severity and FP analysis plots for T-set models.
    
    Parameters:
    - tset_keep: T-set results with all FPs kept
    - tset_ignore: T-set results ignoring mild positive FPs  
    - tset_flip: T-set results flipping mild positive FPs to TPs
    - SEVERITY_CATS: List of severity categories
    - SEVERITY_COLORS: Dictionary mapping categories to colors
    
    Returns:
    - None (displays plots)
    """

    if not tset_keep:
        print("No T-set models to plot.")
        return
    
    # Pick which variant to visualize in the charts
    VIEW = "ignore"  # one of: "keep", "ignore", "flip_to_tp"
    current = {"keep": tset_keep, "ignore": tset_ignore, "flip_to_tp": tset_flip}[VIEW]
    title_suffix = {
        "keep": " (All FPs)",
        "ignore": " (Excluding Mild Positives)",
        "flip_to_tp": " (Reclassifying Mild Positives as TP)",
    }[VIEW]

    labels = [f"combo_{m['combo_index']}" for m in current]
    x = np.arange(len(labels))
    width = 0.35

    def get_ratio(models, key):
        return [m.get("fp_tp_ratios", {}).get(key, np.nan) for m in models]

    r_05_07 = get_ratio(current, "0.5-0.7")
    r_07_10 = get_ratio(current, "0.7-1.0")

    def get_counts(model, bracket_key, cat):
        return int(model.get("fp_severity_by_bracket", {}).get(bracket_key, {}).get(cat, 0))

    low_data  = {cat: [get_counts(m, "0.5-0.7", cat) for m in current] for cat in SEVERITY_CATS}
    high_data = {cat: [get_counts(m, "0.7-1.0", cat) for m in current] for cat in SEVERITY_CATS}

    # Precision distributions from mean_precision (already recomputed under each policy)
    precis_keep   = [m["mean_precision"] for m in tset_keep   if not np.isnan(m["mean_precision"])]
    precis_ignore = [m["mean_precision"] for m in tset_ignore if not np.isnan(m["mean_precision"])]
    precis_flip   = [m["mean_precision"] for m in tset_flip   if not np.isnan(m["mean_precision"])]

    # Severe FP counts per model (sum both brackets)
    severe_counts = [
        sum(get_counts(m, b, "Severe (<-0.12)") for b in ("0.5-0.7", "0.7-1.0"))
        for m in current
    ]

    # PLOTS (3x2 layout)
    fig, axes = plt.subplots(3, 2, figsize=(max(16, len(labels)*0.8), 18))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # Plot 1: FP ratios
    ax1.bar(x - width/2, r_05_07, width, label='FP/(TP+FP) 0.5–0.7',
            alpha=0.8, color='skyblue', edgecolor='navy', linewidth=1)
    ax1.bar(x + width/2, r_07_10, width, label='FP/(TP+FP) 0.7–1.0',
            alpha=0.8, color='lightcoral', edgecolor='darkred', linewidth=1)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('FP / (TP + FP) Ratio')
    ax1.set_title(f'T-set: FP Ratios by Prediction Bracket{title_suffix}', fontsize=12, fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3, axis='y')
    for i, (v1, v2) in enumerate(zip(r_05_07, r_07_10)):
        if not np.isnan(v1): ax1.text(i - width/2, v1 + 0.01, f'{v1:.2f}', ha='center', va='bottom', fontsize=8)
        if not np.isnan(v2): ax1.text(i + width/2, v2 + 0.01, f'{v2:.2f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Stacked severity by bracket
    bottom_low  = np.zeros(len(labels))
    bottom_high = np.zeros(len(labels))
    legend_handles = []
    for cat in SEVERITY_CATS:
        color = SEVERITY_COLORS.get(cat, 'gray')
        ax2.bar(x - width/2, low_data[cat],  width, bottom=bottom_low,  color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax2.bar(x + width/2, high_data[cat], width, bottom=bottom_high, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        bottom_low  += np.array(low_data[cat])
        bottom_high += np.array(high_data[cat])
        legend_handles.append(plt.Rectangle((0,0),1,1,fc=color,alpha=0.85,label=cat,edgecolor='black'))
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('Count of False Positives')
    ax2.set_title('T-set: False Positive Severity by Prediction Bracket', fontsize=12, fontweight='bold')
    ax2.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.15, 1))
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3–5: Precision distributions under each policy
    def plot_hist(ax, data, title, color, edge):
        if data:
            ax.hist(data, bins=20, alpha=0.7, color=color, edgecolor=edge)
            ax.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.3f}')
            ax.set_xlabel('Precision_up'); ax.set_ylabel('Frequency')
            ax.set_title(title, fontsize=12, fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No valid precision data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')

    plot_hist(ax3, precis_keep,   'Precision_up (All FPs)',                          'lightcoral', 'darkred')
    plot_hist(ax4, precis_ignore, 'Precision_up (Excluding Mild Positives)',         'lightgreen', 'darkgreen')
    plot_hist(ax5, precis_flip,   'Precision_up (Reclassifying Mild Positives as TP)','lightblue', 'darkblue')

    # Plot 6: Comparison boxplot
    pdata = [precis_keep, precis_ignore, precis_flip]
    if all(len(d) > 0 for d in pdata):
        ax6.boxplot(pdata, labels=['Original', 'Exclude', 'Reclassify'])
        ax6.set_ylabel('Precision_up')
        ax6.set_title('Comparison of Precision Calculation Methods', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
    else:
        ax6.text(0.5, 0.5, 'Insufficient data for comparison', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Comparison of Precision Calculation Methods', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

def create_metric_distribution_plots(selected_models):
    """
    Creates distribution plots for various model and seed metrics with percentile markers.
    
    Parameters:
    -----------
    selected_models : list
        List of selected model dictionaries containing metrics
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    def plot_metric_histogram(data_list, title, metric_name, percentiles=(35, 85), text_offset=1, figure_size=(8, 2)):
        """Helper function to create a single histogram with percentile markers"""
        plt.figure(figsize=figure_size)
        
        # Handle None values
        valid_data = [x for x in data_list if x is not None]
        if not valid_data:
            plt.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(title)
            plt.show()
            return
        
        zero_count = sum(1 for x in valid_data if x == 0)
        print(f'Number of {metric_name} with value of 0: {zero_count}')
        
        q1 = np.percentile(valid_data, percentiles[0])
        q2 = np.percentile(valid_data, percentiles[1])
        
        plt.hist(valid_data, bins=30)
        ax = plt.gca()
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Add percentile markers
        plt.text(q1 + text_offset, y_max * 0.7, f'{q1:.2f}', rotation=90)
        plt.text(q2 + text_offset, y_max * 0.7, f'{q2:.2f}', rotation=90)
        plt.axvline(x=q1, color='k')
        plt.axvline(x=q2, color='k')
        
        plt.title(title)
        plt.show()
    
    # Model-level metrics
    print("=== MODEL METRICS DISTRIBUTIONS ===")
    
    # Mean precision
    mean_precision_list = [model['mean_precision'] for model in selected_models]
    plot_metric_histogram(mean_precision_list, 'Model Mean Precision Distribution', 'models with mean precision')
    
    # Mean recall up
    mean_recall_up_list = [model['mean_recall_up'] for model in selected_models]
    plot_metric_histogram(mean_recall_up_list, 'Model Mean Recall Up Distribution', 'models with mean recall up')
    
    # Ratio difference
    ratio_difference_list = [model['ratio_difference'] for model in selected_models if model['ratio_difference'] is not None]
    plot_metric_histogram(ratio_difference_list, 'Model Ratio Difference Distribution', 'models with ratio difference', text_offset=0.01)
    
    # FP severe ratio FPs
    fp_severe_ratio_fps_list = [model['fp_severe_ratio_fps'] for model in selected_models if model['fp_severe_ratio_fps'] is not None]
    plot_metric_histogram(fp_severe_ratio_fps_list, 'Model FP Severe Ratio FPs Distribution', 'models with FP Severe Ratio FPs', text_offset=0.01)
    
    # FP severe ratio FPs and TPs
    fp_severe_ratio_fps_tps_list = [model['fp_severe_ratio_fps_tps'] for model in selected_models if model['fp_severe_ratio_fps_tps'] is not None]
    plot_metric_histogram(fp_severe_ratio_fps_tps_list, 'Model FP Severe Ratio FPs and TPs Distribution', 'models with FP Severe Ratio FPs and TPs', text_offset=0.01)
    
    # Seed-level metrics
    print("=== SEED METRICS DISTRIBUTIONS ===")
    
    # Seed precision
    all_seed_precisions = []
    for model in selected_models:
        for seed_result in model['selected_seeds']:
            all_seed_precisions.append(seed_result['precision'])
    plot_metric_histogram(all_seed_precisions, 'Seed Precision Distribution', 'seeds with precision')
    
    # Seed recall
    all_seed_recalls = []
    for model in selected_models:
        for seed_result in model['selected_seeds']:
            all_seed_recalls.append(seed_result['recall'])
    plot_metric_histogram(all_seed_recalls, 'Seed Recall Distribution', 'seeds with recall')
    
    # Seed TP minus FP
    all_seed_TPminusFP = []
    for model in selected_models:
        for seed_result in model['selected_seeds']:
            bracket_high = seed_result["seed_bracket_counts"].get("0.7-1.0", {"TP": 0, "FP": 0})
            bracket_low = seed_result["seed_bracket_counts"].get("0.5-0.7", {"TP": 0, "FP": 0})
            net_high = bracket_high["TP"] - bracket_high["FP"]
            net_low = bracket_low["TP"] - bracket_low["FP"]
            TPminusFP = net_high - net_low
            all_seed_TPminusFP.append(TPminusFP)
    plot_metric_histogram(all_seed_TPminusFP, 'Seed TP minus FP Distribution', 'seeds with TP minus FP')
    
    # Seed severe FPs in high bracket
    all_seed_severe_FPs_high_bracket = []
    for model in selected_models:
        for seed_result in model['selected_seeds']:
            all_seed_severe_FPs_high_bracket.append(seed_result['seed_severe_fp_high_bracket'])
    plot_metric_histogram(all_seed_severe_FPs_high_bracket, 'Seed Severe FPs in High Bracket Distribution', 'seeds with severe FP high bracket')
    
    # Seed TPs in high bracket
    all_seed_TPs_high_bracket = []
    for model in selected_models:
        for seed_result in model['selected_seeds']:
            if "0.7-1.0" in seed_result['seed_bracket_counts']:
                all_seed_TPs_high_bracket.append(seed_result['seed_bracket_counts']["0.7-1.0"]["TP"])
    plot_metric_histogram(all_seed_TPs_high_bracket, 'Seed TPs in High Bracket Distribution', 'seeds with TP high bracket')