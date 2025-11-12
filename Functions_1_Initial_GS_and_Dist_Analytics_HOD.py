import matplotlib.pyplot as plt

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



def TP_FP_brackets_analysis(top_models ,brackets , mdoel_name : str ):
        
    # Count TP and FP for each bracket
    bracket_counts = defaultdict(lambda: {'TP': 0, 'FP': 0})

    for score, entry in top_models:
        for pred_fold, actual_fold in zip(entry['all_preds'], entry['all_actuals']):
            for pred, actual in zip(pred_fold, actual_fold):
                if pred > 0.5:
                    for L, H in brackets:
                        if L <= pred <= H:
                            if actual > 0.5:
                                bracket_counts[f"{L}-{H}"]['TP'] += 1
                            else:
                                bracket_counts[f"{L}-{H}"]['FP'] += 1

    # Prepare data for plotting
    bracket_labels = sorted(bracket_counts.keys(), key=lambda x: float(x.split('-')[0]))
    tp_counts = [bracket_counts[b]['TP'] for b in bracket_labels]
    fp_counts = [bracket_counts[b]['FP'] for b in bracket_labels]

    # Create plot
    plt.figure(figsize=(4, 5))
    x = np.arange(len(bracket_labels))
    width = 0.35

    plt.bar(x - width/2, tp_counts, width, label='True Positives', color='green', alpha=0.7)
    plt.bar(x + width/2, fp_counts, width, label='False Positives', color='red', alpha=0.7)

    plt.xlabel('Confidence Brackets ')
    plt.ylabel('Count')
    plt.title(f'TP vs FP by Confidence Bracket  - {mdoel_name}')
    plt.xticks(x, bracket_labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def TP_FP_brackets_severity_analysis(top, brackets , name : str ):

    # Count FP severity by bracket
    bracket_severity_counts = defaultdict(lambda: {
        "Pos_high (0.05-0.1)":0 ,
        "Pos_low (0-0.05)":0 , 
        "Neg_low (-0.05) - 0": 0 ,
        "Neg_mid (-0.12) - (-0.05)" : 0 ,
        "Severe (<-0.12)" : 0 })


    for score, entry in top:
        all_preds = entry['all_preds']
        all_actuals = entry['all_actuals']
        raw_actuals = entry['raw_actuals']
        
        # Flatten all arrays
        flat_preds = [p for fold in all_preds for p in fold]
        flat_actuals = [a for fold in all_actuals for a in fold]
        flat_raw_actuals = [r for fold in raw_actuals for r in fold]
        
        for p, a, raw in zip(flat_preds, flat_actuals, flat_raw_actuals):
            # Only process FALSE POSITIVES
            if p > 0.5 and a < 0.5:
                severity = classify_severity_10_percent_pos(raw)

                # print(f"Pred: {p:.3f}, Actual: {a:.3f}, Raw: {raw:.3f} => Severity: {severity}")
                
                # Find which bracket this prediction falls into
                for L, H in brackets:
                    if L < p < H:
                        bracket_severity_counts[f"{L}-{H}"][severity] += 1
                        break

    # Prepare data for plotting
    bracket_labels = sorted(bracket_severity_counts.keys(), key=lambda x: float(x.split('-')[0]))
    severity_categories = ["Pos_high (0.05-0.1)", "Pos_low (0-0.05)", "Neg_low (-0.05) - 0", "Neg_mid (-0.12) - (-0.05)", "Severe (<-0.12)"]

    # Create the stacked bar plot
    plt.figure(figsize=(4, 5))
    x = np.arange(len(bracket_labels))
    width = 0.8
    bottom = np.zeros(len(bracket_labels))

    colors = ['lightblue', 'lightgreen', 'yellow' , 'orange', 'red']

    for i, category in enumerate(severity_categories):
        counts = [bracket_severity_counts[b][category] for b in bracket_labels]
        plt.bar(x, counts, width, bottom=bottom, label=category, alpha=0.8, color=colors[i])
        bottom += counts

    plt.xlabel('Confidence Brackets')
    plt.ylabel('Count of False Positives')
    plt.title(f'False Positives by Confidence Bracket vs Actual Severity - {name}')
    plt.xticks(x, bracket_labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add total counts and severity proportions on top
    for i, bracket in enumerate(bracket_labels):
        total = sum(bracket_severity_counts[bracket].values())
        severe_count = bracket_severity_counts[bracket]['Severe (<-0.12)']
        severe_proportion = (severe_count / total * 100) if total > 0 else 0
        
        plt.text(i, total + 0.5, f"{total}\n({severe_proportion:.0f}% severe)", 
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Print severity proportion analytics
    print("Severe Case Proportions by Confidence Bracket:")
    print("-" * 50)
    for bracket in bracket_labels:
        total = sum(bracket_severity_counts[bracket].values())
        severe_count = bracket_severity_counts[bracket]['Severe (<-0.12)']
        severe_proportion = (severe_count / total * 100) if total > 0 else 0
        
        print(f"{bracket}: {severe_proportion:.1f}% severe ({severe_count}/{total})")

    print("Severity Counts by Confidence Bracket:")
    print("-" * 50)
    for bracket in bracket_labels:
        print(f"{bracket}: {bracket_severity_counts[bracket]}")





########### 

import math
from collections import defaultdict


def classify_severity_10_percent_pos(actual_value: float) -> str:
    """Mirror your classify_severity_10_percent_pos() categories."""
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

def _labels_from_targets(targets):
    """
    Accepts 'pos_low', 'pos_high', ('pos_low','pos_high'), [] or None.
    Returns severity labels to match against classify_severity_10_percent_pos.
    """
    if targets is None:
        targets = []
    if isinstance(targets, str):
        targets = [targets]
    label_map = {
        "pos_low":  "Pos_low (0-0.05)",
        "pos_high": "Pos_high (0.05-0.1)",
    }
    out = set()
    for k in targets:
        k2 = str(k).strip().lower()
        if k2 in label_map:
            out.add(label_map[k2])
    return out

def _find_source_model(results_data, combo_index):
    for m in results_data:
        if m.get("combo_index") == combo_index:
            return m
    return None

def _find_seed_run(per_seed_all_results, seed_num):
    for run in per_seed_all_results:
        if run.get("seed") == seed_num:
            return run
    return None

def recompute_metrics_ignoring_pos_fp(
    selected_models_output,
    results_data,
    targets=("pos_low", "pos_high"),
    fp_treatment="ignore",
):
    """
    Recompute the SAME metrics as select_models_by_criteria(), with two options:
      1) Add FP severity-by-bracket data to the output (model-level).
      2) Handle Pos_low/Pos_high FPs in one of three ways via `fp_treatment`:
         - "ignore": do not count those FPs at all (removed from denominators)
         - "flip_to_tp": convert those FPs into TPs
         - "keep": leave them as FPs (baseline)

    Parameters
    ----------
    selected_models_output : list
        Output from select_models_by_criteria() (the chosen models + seeds).
    results_data : list
        Raw results (e.g., results_dist_disc or Tset) with folds/preds/actuals.
    targets : str | list | tuple
        Which FP severities to act on: "pos_low", "pos_high", both, or empty.
    fp_treatment : {"ignore","flip_to_tp","keep"}
        How to treat the targeted severities (default "ignore").

    Returns
    -------
    list
        Same schema as before, plus "fp_severity_by_bracket" at the model level:
        {
          "0.5-0.7": {"Pos_low (0-0.05)": n, "Pos_high (0.05-0.1)": m, ...},
          "0.7-1.0": {...}
        }
        Note: The counts reflect FPs that *actually remained FPs after treatment*.

    """
    BRACKETS = [(0.5, 0.7), (0.7, 1.0) , (0.9,1.0)]
    BRACKET_KEYS = [f"{L}-{H}" for (L, H) in BRACKETS]

    target_labels = _labels_from_targets(targets)
    fp_treatment = fp_treatment.lower()
    assert fp_treatment in {"ignore", "flip_to_tp", "keep"}, "fp_treatment must be 'ignore', 'flip_to_tp', or 'keep'"

    recomputed_models = []

    for model_stub in selected_models_output:
        combo_idx = model_stub["combo_index"]
        src_model = _find_source_model(results_data, combo_idx)
        if src_model is None:
            continue  # gracefully skip if not found

        # Aggregate at model level
        model_bracket_counts = defaultdict(lambda: {"TP": 0, "FP": 0})
        model_non_bracket_counts = {"TN": 0, "FN": 0}
        model_fp_severity_by_bracket = defaultdict(lambda: defaultdict(int))  # NEW
        model_fp_severity_by_bracket_raw_vals = defaultdict(lambda: defaultdict(list))  # NEW: 

        total_tp_all = 0
        total_fp_all = 0
        total_severe_fp = 0

        seed_outputs = []
        seed_precisions = []
        seed_recall_ups = []

        for seed_info in model_stub["selected_seeds"]:
            seed_num = seed_info["seed_num"]
            seed_run = _find_seed_run(src_model["per_seed_all_results"], seed_num)
            if seed_run is None:
                continue

            re = seed_run["result_entry"]
            seed_bracket_counts = defaultdict(lambda: {"TP": 0, "FP": 0})
            seed_non_bracket_counts = {"TN": 0, "FN": 0}
            seed_total_tp = 0
            seed_total_fp = 0
            seed_severe_fp = 0
            seed_severe_fp_high_bracket = 0

            for pred_fold, actual_fold, raw_fold in zip(
                re["all_preds"], re["all_actuals"], re["raw_actuals"]
            ):
                for pred, actual, raw in zip(pred_fold, actual_fold, raw_fold):
                    true_positive  = (actual > 0.5) and (pred >= 0.5)
                    false_positive = (actual <= 0.5) and (pred >= 0.5)
                    true_negative  = (actual <= 0.5) and (pred < 0.5)
                    false_negative = (actual > 0.5) and (pred < 0.5)

                    sev = None
                    # If it's an FP, decide how to treat it based on severity
                    if false_positive:
                        sev = classify_severity_10_percent_pos(raw)
                        if sev in target_labels:
                            if fp_treatment == "ignore":
                                # Drop this FP entirely
                                continue
                            elif fp_treatment == "flip_to_tp":
                                # Re-label this instance as TP instead of FP
                                false_positive = False
                                true_positive = True
                                # (sev stays computed but won't count toward FP severity)

                    assigned = False
                    for L, H in BRACKETS:
                        if L <= pred <= H:
                            assigned = True
                            key = f"{L}-{H}"
                            if true_positive:
                                seed_bracket_counts[key]["TP"] += 1
                                seed_total_tp += 1
                                total_tp_all += 1
                            elif false_positive:
                                seed_bracket_counts[key]["FP"] += 1
                                seed_total_fp += 1
                                total_fp_all += 1

                                # Track FP severity-by-bracket *after treatment*
                                # (i.e., only FPs that remained FPs)
                                if sev is None:
                                    sev = classify_severity_10_percent_pos(raw)
                                model_fp_severity_by_bracket[key][sev] += 1
                                model_fp_severity_by_bracket_raw_vals[key][sev].append(raw)  # NEW: keep raw vals


                                # Track severe FP
                                if sev == "Severe (<-0.12)":
                                    seed_severe_fp += 1
                                    total_severe_fp += 1
                                    if key == "0.7-1.0":
                                        seed_severe_fp_high_bracket += 1
                            # break ### NEW NEW removed the break to allow pred to be stored in multiple brackets since they are now overlapping 

                    if not assigned:
                        if true_negative:
                            seed_non_bracket_counts["TN"] += 1
                            model_non_bracket_counts["TN"] += 1
                        elif false_negative:
                            seed_non_bracket_counts["FN"] += 1
                            model_non_bracket_counts["FN"] += 1

            # accumulate into model totals
            for key, counts in seed_bracket_counts.items():
                model_bracket_counts[key]["TP"] += counts["TP"]
                model_bracket_counts[key]["FP"] += counts["FP"]

            # seed-level precision/recall after treatment
            tp_sum = sum(c["TP"] for c in seed_bracket_counts.values())
            fp_sum = sum(c["FP"] for c in seed_bracket_counts.values())
            fn_sum = seed_non_bracket_counts["FN"]

            seed_precision = float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else float("nan")
            seed_recall = float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else float("nan")

            # ratios per seed
            seed_fp_tp_ratios = {}
            for key, counts in seed_bracket_counts.items():
                tp, fp = counts["TP"], counts["FP"]
                denom = tp + fp
                seed_fp_tp_ratios[key] = (fp / denom) if denom > 0 else float("nan")

            seed_outputs.append({
                "seed_num": seed_num,
                "seed_order_index": seed_info.get("seed_order_index"),
                "precision": seed_precision,
                "recall": seed_recall,
                "seed_bracket_counts": dict(seed_bracket_counts),
                "seed_non_bracket_counts": dict(seed_non_bracket_counts),
                "seed_fp_tp_ratios": seed_fp_tp_ratios,
                "seed_total_tp": seed_total_tp,
                "seed_severe_fp": seed_severe_fp,
                "seed_severe_fp_high_bracket": seed_severe_fp_high_bracket,
            })

            if not math.isnan(seed_precision):
                seed_precisions.append(seed_precision)

            if not math.isnan(seed_recall):
                seed_recall_ups.append(seed_recall)

        # model-level aggregates
        fp_tp_ratios = {}
        for key, counts in model_bracket_counts.items():
            tp, fp = counts["TP"], counts["FP"]
            denom = tp + fp
            fp_tp_ratios[key] = (fp / denom) if denom > 0 else float("nan")

        ratio_difference = None
        if "0.7-1.0" in fp_tp_ratios and "0.5-0.7" in fp_tp_ratios:
            rh = fp_tp_ratios["0.7-1.0"]
            rl = fp_tp_ratios["0.5-0.7"]
            if not (math.isnan(rh) or math.isnan(rl)):
                ratio_difference = rl - rh

        # severe FP ratios (after treatment)
        fp_severe_ratio_fps = (total_severe_fp / total_fp_all) if total_fp_all > 0 else 0.0
        fp_severe_ratio_fps_tps = (total_severe_fp / (total_fp_all + total_tp_all)) if (total_fp_all + total_tp_all) > 0 else 0.0

        mean_precision = float(sum(seed_precisions) / len(seed_precisions)) if seed_precisions else float("nan")
        zero_precision_count = sum(1 for s in seed_outputs if (not math.isnan(s["precision"]) and s["precision"] == 0))

        mean_recall_up = float(sum(seed_recall_ups) / len(seed_recall_ups)) if seed_recall_ups else float("nan")

        recomputed_models.append({
            "combo_index": combo_idx,
            "parameters": model_stub["parameters"],  # keep original params
            "mean_precision": mean_precision,
            "mean_recall_up": mean_recall_up,
            "valid_seeds_count": len(seed_outputs),
            "total_seeds": len(seed_outputs),
            "zero_precision_count": zero_precision_count,
            "bracket_counts": dict(model_bracket_counts),
            "non_bracket_counts": dict(model_non_bracket_counts),
            "fp_tp_ratios": fp_tp_ratios,
            "ratio_difference": ratio_difference,
            "fp_severe_ratio_fps": fp_severe_ratio_fps,
            "fp_severe_ratio_fps_tps": fp_severe_ratio_fps_tps,
            "fp_severity_by_bracket": {k: dict(v) for k, v in model_fp_severity_by_bracket.items()},  # NEW
            "fp_severity_by_bracket_raw_vals": {k: dict(v) for k, v in model_fp_severity_by_bracket_raw_vals.items()},  # NEW: raw vals
            "selected_seeds": seed_outputs
        })

    return recomputed_models

############

import math
import numpy as np
from collections import defaultdict

def recompute_selected_models_on_Tset(results_dist_disc_Tset_same_seeds_organized,
                                      selected_models,
                                      brackets=((0.5, 0.7), (0.7, 1.0) , (0.9,1.0))):
    """
    Recompute the SAME analytics produced by `select_models_by_criteria`
    for the SAME models & seeds, but using the T-set dataset.
    """


    tset_by_combo = {m["combo_index"]: m for m in results_dist_disc_Tset_same_seeds_organized}

    # Helper to compute all counts/metrics for a single model on Tset (across ALL seeds)
    def _compute_model_metrics_on_tset(model_entry, brackets):
        per_seed = model_entry["per_seed_all_results"]

        zero_precision_count = 0
        bracket_counts = defaultdict(lambda: {'TP': 0, 'FP': 0})
        non_bracket_counts = {'TN': 0, 'FN': 0}
        seed_records_all = []

        total_severe_fp = 0
        total_fp_all = 0
        total_tp_all = 0

        # NEW: aggregate FP severity distribution by bracket (across ALL seeds)
        fp_severity_by_bracket = defaultdict(lambda: defaultdict(int))  # NEW
        fp_severity_by_bracket_raw_vals = defaultdict(lambda: defaultdict(list))  # NEW: FP severity counts per bracket

        for run_idx, run in enumerate(per_seed):
            re = run["result_entry"]
            om = re["cv_sets"]["overall_metrics"]
            seed_num = run["seed"]

            prec = om.get("precision_up")
            rec = om.get("recall_up")

            if isinstance(prec, (int, float)) and not math.isnan(prec):
                prec = float(prec)
                if prec == 0:
                    zero_precision_count += 1
            else:
                prec = None

            if isinstance(rec, (int, float)) and not math.isnan(rec):
                rec = float(rec)
            else:
                rec = None

            # Per-seed tallies for this model
            seed_bracket_counts = defaultdict(lambda: {'TP': 0, 'FP': 0})
            seed_non_bracket_counts = {'TN': 0, 'FN': 0}
            seed_total_tp = 0
            seed_total_fp = 0
            seed_severe_fp = 0
            seed_severe_fp_high_bracket = 0

            # Walk folds/timesteps exactly like selection function
            for pred_fold, actual_fold, raw_fold in zip(re["all_preds"], re["all_actuals"], re["raw_actuals"]):
                for pred, actual, raw in zip(pred_fold, actual_fold, raw_fold):
                    true_positive = (actual > 0.5) and (pred >= 0.5)
                    false_positive = (actual <= 0.5) and (pred >= 0.5)
                    true_negative = (actual <= 0.5) and (pred < 0.5)
                    false_negative = (actual > 0.5) and (pred < 0.5)

                    assigned = False
                    for (L, H) in brackets:
                        if L <= pred <= H:
                            assigned = True
                            key = f"{L}-{H}"
                            if true_positive:
                                bracket_counts[key]['TP'] += 1
                                seed_bracket_counts[key]['TP'] += 1
                                seed_total_tp += 1
                                total_tp_all += 1
                            elif false_positive:
                                bracket_counts[key]['FP'] += 1
                                seed_bracket_counts[key]['FP'] += 1
                                seed_total_fp += 1
                                total_fp_all += 1

                                # Severity classification & tallies
                                severity = classify_severity_10_percent_pos(raw)
                                fp_severity_by_bracket[key][severity] += 1   # NEW
                                fp_severity_by_bracket_raw_vals[key][severity].append(raw)  # NEW: keep raw vals
                                if severity == "Severe (<-0.12)":
                                    seed_severe_fp += 1
                                    total_severe_fp += 1
                                    if key == "0.7-1.0":
                                        seed_severe_fp_high_bracket += 1
                            # break ### break has been removed since brackets are now overlapping to allow multiple preds to be stored in diff bracekts 

                    if not assigned:
                        if true_negative:
                            non_bracket_counts['TN'] += 1
                            seed_non_bracket_counts['TN'] += 1
                        elif false_negative:
                            non_bracket_counts['FN'] += 1
                            seed_non_bracket_counts['FN'] += 1

            # Per-seed ratios
            seed_fp_tp_ratios = {}
            for key, counts in seed_bracket_counts.items():
                tp, fp = counts['TP'], counts['FP']
                denom = tp + fp
                seed_fp_tp_ratios[key] = (fp / denom) if denom > 0 else float("nan")

            seed_records_all.append({
                "seed_order_index": run_idx,  # position within per_seed_all_results
                "seed_num": seed_num,
                "precision": prec,
                "recall": rec,
                "seed_bracket_counts": dict(seed_bracket_counts),
                "seed_non_bracket_counts": dict(seed_non_bracket_counts),
                "seed_fp_tp_ratios": seed_fp_tp_ratios,
                "seed_total_tp": seed_total_tp,
                "seed_total_fp": seed_total_fp,
                "seed_severe_fp": seed_severe_fp,
                "seed_severe_fp_high_bracket": seed_severe_fp_high_bracket,
            })

        # Mean precision across ALL seeds
        non_none_precs = [r["precision"] for r in seed_records_all if r["precision"] is not None]
        non_none_recall_ups = [r["recall"] for r in seed_records_all if r["recall"] is not None]

        mean_precision = float(np.mean(non_none_precs)) if non_none_precs else float("nan")
        mean_recall_up = float(np.mean(non_none_recall_ups)) if non_none_recall_ups else float("nan")
        
        # Model-level fp/tp ratio by bracket
        fp_tp_ratios = {}
        for key, counts in bracket_counts.items():
            tp, fp = counts['TP'], counts['FP']
            denom = tp + fp
            fp_tp_ratios[key] = (fp / denom) if denom > 0 else float("nan")

        # rl - rh (low minus high)
        ratio_difference = None
        if "0.7-1.0" in fp_tp_ratios and "0.5-0.7" in fp_tp_ratios:
            rh = fp_tp_ratios["0.7-1.0"]
            rl = fp_tp_ratios["0.5-0.7"]
            if not math.isnan(rh) and not math.isnan(rl):
                ratio_difference = rl - rh

        # Severe FP ratios
        fp_severe_ratio_fps = (total_severe_fp / total_fp_all) if total_fp_all > 0 else 0
        denom_pos = (total_fp_all + total_tp_all)
        fp_severe_ratio_fps_tps = (total_severe_fp / denom_pos) if denom_pos > 0 else 0

        return {
            "mean_precision": mean_precision,
            "mean_recall_up": mean_recall_up,
            "zero_precision_count": zero_precision_count,
            "total_seeds": len(seed_records_all),
            "seed_records_all": seed_records_all,  # keep for seed picking below
            "bracket_counts": dict(bracket_counts),
            "non_bracket_counts": dict(non_bracket_counts),
            "fp_tp_ratios": fp_tp_ratios,
            "ratio_difference": ratio_difference,
            "fp_severe_ratio_fps": fp_severe_ratio_fps,
            "fp_severe_ratio_fps_tps": fp_severe_ratio_fps_tps,
            "fp_severity_by_bracket": {k: dict(v) for k, v in fp_severity_by_bracket.items()},  # NEW
            "fp_severity_by_bracket_raw": {k: dict(v) for k, v in fp_severity_by_bracket_raw_vals.items()},  # NEW
        }

    # Build output mirroring the original selection output
    mirrored_outputs = []

    for sel_model in selected_models:
        combo_idx = sel_model["combo_index"]
        # Find the corresponding Tset model
        if combo_idx not in tset_by_combo:
            continue

        tset_model = tset_by_combo[combo_idx]
        model_metrics = _compute_model_metrics_on_tset(tset_model, brackets)

        # Build a map from seed_num -> seed_record (computed from Tset)
        seed_map = {sr["seed_num"]: sr for sr in model_metrics["seed_records_all"]}

        # Recreate the "selected_seeds" list using the EXACT same seeds (and order) from training selection
        selected_seeds_Tset = []
        for s in sel_model["selected_seeds"]:
            seed_num = s["seed_num"]
            if seed_num not in seed_map:
                continue

            sr = seed_map[seed_num]
            # Mirror the schema exactly
            selected_seeds_Tset.append({
                "seed_num": sr["seed_num"],
                "seed_order_index": sr["seed_order_index"],
                "precision": sr["precision"],
                "recall": sr["recall"],
                "seed_bracket_counts": sr["seed_bracket_counts"],
                "seed_non_bracket_counts": sr["seed_non_bracket_counts"],
                "seed_fp_tp_ratios": sr["seed_fp_tp_ratios"],
                "seed_total_tp": sr["seed_total_tp"],
                "seed_severe_fp": sr["seed_severe_fp"],
                "seed_severe_fp_high_bracket": sr["seed_severe_fp_high_bracket"],
            })

        # Assemble model-level dict in the EXACT same shape as your selection output
        mirrored_outputs.append({
            "combo_index": combo_idx,
            "parameters": tset_model["combo"],  # should match
            "mean_precision": model_metrics["mean_precision"],
            "mean_recall_up": model_metrics["mean_recall_up"],
            "valid_seeds_count": len(selected_seeds_Tset),               # SAME selected seeds (on Tset)
            "total_seeds": model_metrics["total_seeds"],                  # all seeds present on Tset
            "zero_precision_count": model_metrics["zero_precision_count"],
            "bracket_counts": model_metrics["bracket_counts"],
            "non_bracket_counts": model_metrics["non_bracket_counts"],
            "fp_tp_ratios": model_metrics["fp_tp_ratios"],
            "ratio_difference": model_metrics["ratio_difference"],
            "fp_severe_ratio_fps": model_metrics["fp_severe_ratio_fps"],
            "fp_severe_ratio_fps_tps": model_metrics["fp_severe_ratio_fps_tps"],
            "fp_severity_by_bracket": model_metrics["fp_severity_by_bracket"],  # NEW
            "fp_severity_by_bracket_raw": model_metrics["fp_severity_by_bracket_raw"],  # NEW
            "selected_seeds": selected_seeds_Tset,
        })

    return mirrored_outputs


import random

# Modified function to store TP and FP counts for each seed
def select_models_by_criteria(results_data, 
                              
                             use_custom_thesh_loss_Fn = False,
                             use_custom_thesh_severity_loss_Fn = False,

                             mean_precision_range=(0, 100),
                            
                             mean_recall_up_range=(0, 100),

                             seed_precision_range=(0, 100),
                             seed_recall_range=(0, 100),
                             min_seeds_per_model=1,
                             max_models_to_return=10,
                             max_zero_precision_seeds=None,
                             min_ratio_difference=None,
                             max_ratio_difference=None,
                             seed_min_TPminusFP_greaterEqual=None,
                             max_FP_severe_ratio_FPs=None,         
                             max_FP_severe_ratio_FPs_andTPs=None, 
                             min_FP_severe_ratio_FPs=None,         
                             min_FP_severe_ratio_FPs_andTPs=None, 
                                                           
                             max_seed_severe_FPs_high_bracket=None,
                             min_seed_TPs_high_bracket=None,       
                             random_seed=None):
    
    if random_seed is not None:
        random.seed(random_seed)

    brackets = [(0.5, 0.7), (0.7, 1.0) , (0.9,1.0)]
    model_summaries = []

    for model in results_data:
        combo_idx = model["combo_index"]

        parameters = model["combo"]
        per_seed = model["per_seed_all_results"]

        zero_precision_count = 0
        bracket_counts = defaultdict(lambda: {'TP': 0, 'FP': 0})
        non_bracket_counts = {'TN': 0, 'FN': 0}
        seed_records = []
        fp_severity_by_bracket = defaultdict(lambda: defaultdict(int))  # NEW: FP severity counts per bracket

        fp_severity_by_bracket_raw_vals = defaultdict(lambda: defaultdict(list))  # NEW: FP severity counts per bracket


        
        # NEW: Track severe FP counts
        total_severe_fp = 0
        total_fp_all = 0
        total_tp_all = 0

        

        for run in per_seed:
            re = run["result_entry"]
            om = re["cv_sets"]["overall_metrics"]

            seed_num = run["seed"]

            prec = om.get("precision_up")
            rec = om.get("recall_up")
            if isinstance(prec, (int, float)) and not math.isnan(prec):
                prec = float(prec)
                if prec == 0:
                    zero_precision_count += 1
            else:
                prec = None

            if isinstance(rec, (int, float)) and not math.isnan(rec):
                rec = float(rec)
            else:
                rec = None

            # Store TP and FP counts for each bracket per seed
            seed_bracket_counts = defaultdict(lambda: {'TP': 0, 'FP': 0})
            seed_non_bracket_counts = {'TN': 0, 'FN': 0}
            seed_total_tp = 0
            seed_total_fp = 0
            seed_severe_fp = 0  # NEW: Track severe FPs per seed
            seed_severe_fp_high_bracket = 0  # NEW: Track severe FPs specifically in 0.7-1.0 bracket

            for pred_fold, actual_fold ,raw_fold in zip(re["all_preds"], re["all_actuals"], re["raw_actuals"]):
                for pred, actual, raw in zip(pred_fold, actual_fold, raw_fold):
                    true_positive = (actual > 0.5) and (pred >= 0.5)
                    false_positive = (actual <= 0.5) and (pred >= 0.5)
                    true_negative = (actual <= 0.5) and (pred < 0.5)
                    false_negative = (actual > 0.5) and (pred < 0.5)

                    assigned = False
                    for L, H in brackets:
                        if L <= pred <= H:
                            assigned = True
                            key = f"{L}-{H}"
                            if true_positive:
                                bracket_counts[key]['TP'] += 1
                                seed_bracket_counts[key]['TP'] += 1
                                seed_total_tp += 1
                                total_tp_all += 1  # NEW: Count all TPs
                            elif false_positive:
                                bracket_counts[key]['FP'] += 1
                                seed_bracket_counts[key]['FP'] += 1
                                seed_total_fp += 1
                                total_fp_all += 1  # NEW: Count all FPs
                                
                                # NEW: Check and record severity of this FP
                                severity = classify_severity_10_percent_pos(raw)
                                fp_severity_by_bracket[key][severity] += 1  # NEW
                                fp_severity_by_bracket_raw_vals[key][severity].append(raw)  # NEW: Store raw values for potential debugging
                                if severity == "Severe (<-0.12)":
                                    seed_severe_fp += 1
                                    total_severe_fp += 1
                                    # NEW: Check if this severe FP is in the high bracket (0.7-1.0)
                                    if key == "0.7-1.0":
                                        seed_severe_fp_high_bracket += 1

                            # break  ## NEW break removed to allow multiple bracket assignments for same pred since brackets now overlap

                    if not assigned:
                        if true_negative:
                            seed_non_bracket_counts['TN'] += 1
                            non_bracket_counts['TN'] += 1
                        elif false_negative:
                            seed_non_bracket_counts['FN'] += 1
                            non_bracket_counts['FN'] += 1

            seed_fp_tp_ratios = {}
            for key, counts in seed_bracket_counts.items():
                tp, fp = counts['TP'], counts['FP']
                denom = tp + fp
                seed_fp_tp_ratios[key] = (fp / denom) if denom > 0 else float("nan")

            seed_records.append({
                "seed_num": seed_num,
                "precision": prec,
                "recall": rec,
                "seed_bracket_counts": dict(seed_bracket_counts),
                "seed_non_bracket_counts": dict(seed_non_bracket_counts),
                "seed_fp_tp_ratios": seed_fp_tp_ratios,
                "seed_total_tp": seed_total_tp,
                "seed_total_fp": seed_total_fp,
                "seed_severe_fp": seed_severe_fp,  # NEW: Add severe FP count
                "seed_severe_fp_high_bracket": seed_severe_fp_high_bracket,  # NEW: Add severe FP count in high bracket
                "seed_raw": run,
            })

        non_none_precs = [r["precision"] for r in seed_records if r["precision"] is not None]
        non_none_recall_ups = [r["recall"] for r in seed_records if r["recall"] is not None]

        mean_precision = float(np.mean(non_none_precs)) if non_none_precs else float("nan")
        mean_recall_up = float(np.mean(non_none_recall_ups)) if non_none_recall_ups else float("nan")

        fp_tp_ratios = {}
        for key, counts in bracket_counts.items():
            tp, fp = counts['TP'], counts['FP']
            denom = tp + fp
            fp_tp_ratios[key] = (fp / denom) if denom > 0 else float("nan")

        ratio_difference = None
        if "0.7-1.0" in fp_tp_ratios and "0.5-0.7" in fp_tp_ratios:
            rh, rl = fp_tp_ratios["0.7-1.0"], fp_tp_ratios["0.5-0.7"]
            if not math.isnan(rh) and not math.isnan(rl):
                ratio_difference = rl - rh

        # NEW: Calculate severe FP ratios
        fp_severe_ratio_fps = total_severe_fp / total_fp_all if total_fp_all > 0 else 0
        fp_severe_ratio_fps_tps = total_severe_fp / (total_fp_all + total_tp_all) if (total_fp_all + total_tp_all) > 0 else 0

        model_summaries.append({
            "combo_index": combo_idx,
            "parameters": parameters,
            "mean_precision": mean_precision,
            "mean_recall_up": mean_recall_up,
            "zero_precision_count": zero_precision_count,
            "total_seeds": len(seed_records),
            "seed_records": seed_records,
            "bracket_counts": dict(bracket_counts),
            "non_bracket_counts": dict(non_bracket_counts),
            "fp_tp_ratios": fp_tp_ratios,
            "ratio_difference": ratio_difference,
            "fp_severe_ratio_fps": fp_severe_ratio_fps,           # NEW: Add severe FP ratios
            "fp_severe_ratio_fps_tps": fp_severe_ratio_fps_tps,   # NEW: Add severe FP ratios
            "fp_severity_by_bracket": {k: dict(v) for k, v in fp_severity_by_bracket.items()},  # NEW: FP severity by bracket
            "fp_severity_by_bracket_raw": {k: dict(v) for k, v in fp_severity_by_bracket_raw_vals.items()},  # NEW: Raw values for debugging
        })

    filtered_models = []
    min_mean_prec, max_mean_prec = mean_precision_range
    min_mean_rec, max_mean_rec = mean_recall_up_range
    min_seed_prec, max_seed_prec = seed_precision_range
    min_seed_rec, max_seed_rec = seed_recall_range

    for m in model_summaries:

        mean_prec = m["mean_precision"] if not math.isnan(m["mean_precision"]) else -float("inf")
        if not (min_mean_prec <= mean_prec <= max_mean_prec):
            continue


        if use_custom_thesh_loss_Fn:
            if not (m["parameters"]["use_custom_loss_function_BCE_THRESH"] == True):
                continue
        
        if use_custom_thesh_severity_loss_Fn:
            if not (m["parameters"]["use_custom_loss_function_BCE_THRESH_AND_SEVERITY"] == True):
                continue

        mean_rec = m["mean_recall_up"] if not math.isnan(m["mean_recall_up"]) else -float("inf")
        if not (min_mean_rec <= mean_rec <= max_mean_rec):
            continue

        # Check zero precision count
        if max_zero_precision_seeds is not None and m["zero_precision_count"] > max_zero_precision_seeds:
            continue

        # Check ratio difference
        if min_ratio_difference is not None or max_ratio_difference is not None:
            if m["ratio_difference"] is None:
                continue
            if min_ratio_difference is not None and m["ratio_difference"] < min_ratio_difference:
                continue
        if max_ratio_difference is not None:    
            if m["ratio_difference"] is None:
                continue
            if m["ratio_difference"] > max_ratio_difference:
                continue

        # NEW: Check severe FP ratio among all FPs
        if max_FP_severe_ratio_FPs is not None:
            if not (m["fp_severe_ratio_fps"] < max_FP_severe_ratio_FPs):
                continue

        if min_FP_severe_ratio_FPs is not None:
            if not (m["fp_severe_ratio_fps"] > min_FP_severe_ratio_FPs):
                continue

        # NEW: Check severe FP ratio among all positive predictions (TPs + FPs)
        if max_FP_severe_ratio_FPs_andTPs is not None:
            if not (m["fp_severe_ratio_fps_tps"] < max_FP_severe_ratio_FPs_andTPs):
                continue
        
        if min_FP_severe_ratio_FPs_andTPs is not None:
            if not (m["fp_severe_ratio_fps_tps"] > min_FP_severe_ratio_FPs_andTPs):
                continue

        valid_seeds = []
        for idx, r in enumerate(m["seed_records"]):
            prec, rec = r["precision"], r["recall"]

            if prec is None or rec is None:
                continue
            if (min_seed_prec <= prec <= max_seed_prec) and (min_seed_rec <= rec <= max_seed_rec):

                # TP-FP difference filtering
                bracket_high = r["seed_bracket_counts"].get("0.7-1.0", {"TP": 0, "FP": 0})
                bracket_low = r["seed_bracket_counts"].get("0.5-0.7", {"TP": 0, "FP": 0})
                
                net_high = bracket_high["TP"] - bracket_high["FP"]
                net_low = bracket_low["TP"] - bracket_low["FP"]

                if seed_min_TPminusFP_greaterEqual is not None:
                    if (bracket_high["TP"] == 0):
                        continue
                    if (bracket_high["FP"] > bracket_high["TP"]):
                        continue
                    if not (net_high - net_low >= seed_min_TPminusFP_greaterEqual):
                        continue

                # NEW: Check max severe FPs in high bracket for this seed
                if max_seed_severe_FPs_high_bracket is not None:
                    if not (r["seed_severe_fp_high_bracket"] <= max_seed_severe_FPs_high_bracket):
                        continue

                # NEW: Check min TPs in high bracket for this seeda
                if min_seed_TPs_high_bracket is not None:
                    if not (bracket_high["TP"] >= min_seed_TPs_high_bracket):
                        continue

                valid_seeds.append({
                    "seed_order_index": idx,
                    "seed_num": r["seed_num"],
                    "precision": prec,
                    "recall": rec,
                    "seed_bracket_counts": r["seed_bracket_counts"],
                    "seed_non_bracket_counts": r["seed_non_bracket_counts"],
                    "seed_fp_tp_ratios": r["seed_fp_tp_ratios"],
                    "seed_total_tp": r["seed_total_tp"],
                    "seed_severe_fp": r["seed_severe_fp"],  # NEW: Include severe FP count
                    "seed_severe_fp_high_bracket": r["seed_severe_fp_high_bracket"],  # NEW: Include severe FP count in high bracket
                })

        if len(valid_seeds) >= min_seeds_per_model:
            filtered_models.append({
                "combo_index": m["combo_index"],
                "parameters": m["parameters"],
                "mean_precision": m["mean_precision"],
                "mean_recall_up": m["mean_recall_up"],
                "valid_seeds_count": len(valid_seeds),
                "valid_seeds": valid_seeds,
                "total_seeds": m["total_seeds"],
                "zero_precision_count": m["zero_precision_count"],
                "bracket_counts": m["bracket_counts"],
                "non_bracket_counts": m["non_bracket_counts"],
                "fp_tp_ratios": m["fp_tp_ratios"],
                "ratio_difference": m["ratio_difference"],
                "fp_severe_ratio_fps": m["fp_severe_ratio_fps"],           # NEW: Include severe FP ratios
                "fp_severe_ratio_fps_tps": m["fp_severe_ratio_fps_tps"],   # NEW: Include severe FP ratios
                "fp_severity_by_bracket": m["fp_severity_by_bracket"],     # NEW: Carry through
                "fp_severity_by_bracket_raw": m["fp_severity_by_bracket_raw"],  # NEW: Carry through
            })

    if max_models_to_return < len(filtered_models):
        n_pick = min(max_models_to_return, len(filtered_models))
        selected_models = random.sample(filtered_models, n_pick) if n_pick > 0 else []
    else:
        selected_models = filtered_models

    output = []
    for m in selected_models:
        output.append({
            "combo_index": m["combo_index"],
            "parameters": m["parameters"],
            "mean_precision": m["mean_precision"],
            "mean_recall_up": m["mean_recall_up"],
            "valid_seeds_count": m["valid_seeds_count"],
            "total_seeds": m["total_seeds"],
            "zero_precision_count": m["zero_precision_count"],
            "bracket_counts": m["bracket_counts"],
            "non_bracket_counts": m["non_bracket_counts"],
            "fp_tp_ratios": m["fp_tp_ratios"],
            "ratio_difference": m["ratio_difference"],
            "fp_severe_ratio_fps": m["fp_severe_ratio_fps"],           # NEW: Include in output
            "fp_severe_ratio_fps_tps": m["fp_severe_ratio_fps_tps"],   # NEW: Include in output
            "fp_severity_by_bracket": m["fp_severity_by_bracket"],     # NEW: Include in output
            "fp_severity_by_bracket_raw": m["fp_severity_by_bracket_raw"],  # NEW: Include in output
            "selected_seeds": [
                {
                    "seed_num": s["seed_num"],
                    "seed_order_index": s["seed_order_index"],
                    "precision": s["precision"],
                    "recall": s["recall"],
                    "seed_bracket_counts": s["seed_bracket_counts"],
                    "seed_non_bracket_counts": s["seed_non_bracket_counts"],
                    "seed_fp_tp_ratios": s["seed_fp_tp_ratios"],
                    "seed_total_tp": s["seed_total_tp"],
                    "seed_severe_fp": s["seed_severe_fp"],  # NEW: Include in seed output
                    "seed_severe_fp_high_bracket": s["seed_severe_fp_high_bracket"],  # NEW: Include in seed output
                }
                for s in m["valid_seeds"]
            ]
        })

    return output



