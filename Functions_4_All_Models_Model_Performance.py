#------------------------------------------------------------------------------------------------------------------------
#                                             ABOUT 

#       - funtions used in the (4) CLEAN Model Performance Assessment HOD 20_01 - 21_12.ipynb 
#       - used to analye the performance of different models and seeds in clean manner 

#------------------------------------------------------------------------------------------------------------------------

    

import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data helpers
# -----------------------------
def _flatten_list_of_lists(lol):
    """Flatten a list of lists; ignore None and non-iterables safely."""
    out = []
    if not lol:
        return out
    for sub in lol:
        if sub is None:
            continue
        # assume sub is iterable of numbers
        try:
            out.extend(sub)
        except TypeError:
            # if sub isn't iterable, treat as single value
            out.append(sub)
    return out

def _nan_to_num_list(vals):
    """Replace NaN/inf with 0 for stable sums; drop None."""
    clean = []
    for v in vals:
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(vf):
            vf = 0.0
        clean.append(vf)
    return clean

def compute_six_metrics(entry):
    """
    entry: dict with keys:
      - "REG_UNIQUE_ALL": list[list[number]]
      - "THR_UNIQUE_ALL": list[list[number]]
    Returns (reg_sum, reg_cnt, reg_ret, thr_sum, thr_cnt, thr_ret)
    where *_ret = sum(unique)/count(unique) (0 if count==0).
    """
    reg_flat = _nan_to_num_list(_flatten_list_of_lists(entry.get("REG_UNIQUE_ALL")))
    thr_flat = _nan_to_num_list(_flatten_list_of_lists(entry.get("THR_UNIQUE_ALL")))

    reg_unique = set(reg_flat)
    thr_unique = set(thr_flat)

    reg_sum = sum(reg_unique)
    reg_cnt = len(reg_unique)
    reg_ret = (reg_sum / reg_cnt) if reg_cnt > 0 else 0.0

    thr_sum = sum(thr_unique)
    thr_cnt = len(thr_unique)
    thr_ret = (thr_sum / thr_cnt) if thr_cnt > 0 else 0.0

    return reg_sum, reg_cnt, reg_ret, thr_sum, thr_cnt, thr_ret


# -----------------------------
# Plotting
# -----------------------------
def plot_model_six_bars(
    models_dict,
    title="Unique-Value Metrics per Model",
    annotate=True,
    annotate_position="above",  # "above" or "below"
    xtick_rotation=45,
    save_path=None
):
    """
    models_dict: dict like
        {
          "ModelA": {"REG_UNIQUE_ALL":[[...], ...], "THR_UNIQUE_ALL":[[...], ...]},
          "ModelB": {...},
          ...
        }
    annotate_position:
        - "above": values above each bar
        - "below": all values aligned along a single baseline under the bars
    """
    # 1) Compute matrix (n_models x 6)
    model_names = list(models_dict.keys())
    metrics = [compute_six_metrics(models_dict[name]) for name in model_names]
    data = np.array(metrics, dtype=float) if metrics else np.zeros((0, 6), dtype=float)

    labels = ["REG_SUM", "REG_NUM_PREDS", "REG_RETURN",
              "THR_SUM", "THR_NUM_PREDS", "THR_RETURN"]

    n_models = len(model_names)
    n_bars = 6
    x = np.arange(n_models, dtype=float)

    # 2) Layout
    width = min(0.12, 0.7 / n_bars)  # keep groups compact
    fig_w = max(10, n_models * 1.9)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    # 3) Bars
    series = []
    for i in range(n_bars):
        series.append(ax.bar(x + i * width, data[:, i], width, label=labels[i]))

    # 4) Axes, ticks, legend
    ax.set_xticks(x + width * (n_bars - 1) / 2)
    ax.set_xticklabels(model_names, rotation=xtick_rotation, ha="right" if xtick_rotation else "center")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(ncols=3, frameon=False)
    ax.margins(x=0.02)

    # 5) Annotations
    if annotate:
        if annotate_position.lower() == "below":
            # One common baseline under all bars
            ymin, ymax = ax.get_ylim()
            yr = ymax - ymin if ymax > ymin else 1.0
            # Make extra space at bottom so text isn't clipped
            ax.set_ylim(ymin - 0.2 * yr, ymax)
            y0, y1 = ax.get_ylim()
            y_text = y0 + 0.04 * (y1 - y0)  # common baseline

            for i in range(n_bars):
                for rect in series[i]:
                    h = rect.get_height()
                    # We annotate the numeric value but position it at y_text
                    x_center = rect.get_x() + rect.get_width() / 2.0
                    ax.annotate(f"{h:.3g}",
                                xy=(x_center, y_text),
                                xytext=(0, 15),
                                rotation = 85 , 
                                rotation_mode = "anchor" , 
                                textcoords="offset points",
                                ha='center', va='top', fontsize=10)


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()


def plot_model_realization_distributions(
    realizations_list,
    model_name,
    title_prefix="Distribution of Metrics Across Realizations",
    bins=10,
    save_path=None
):
    """
    Plot distributions of all 6 metrics for a given model across multiple realizations.
    
    Parameters:
    -----------
    realizations_list : list of dict
        List of dictionaries, each representing one realization of model results.
        Each dict should have structure: {model_name: {"REG_UNIQUE_ALL": [...], "THR_UNIQUE_ALL": [...]}}
    model_name : str
        Name of the model to plot distributions for
    title_prefix : str
        Prefix for the plot title
    bins : int
        Number of bins for histograms
    save_path : str, optional
        Path to save the figure
    """
    
    # Extract all 6 metrics for each realization
    reg_sums = []
    reg_counts = []
    reg_returns = []
    thr_sums = []
    thr_counts = []
    thr_returns = []
    
    for realization_dict in realizations_list:
        if model_name in realization_dict:
            model_data = realization_dict[model_name]
            reg_sum, reg_cnt, reg_ret, thr_sum, thr_cnt, thr_ret = compute_six_metrics(model_data)
            reg_sums.append(reg_sum)
            reg_counts.append(reg_cnt)
            reg_returns.append(reg_ret)
            thr_sums.append(thr_sum)
            thr_counts.append(thr_cnt)
            thr_returns.append(thr_ret)
    
    if not reg_sums:
        print(f"No data found for model '{model_name}' in the realizations.")
        return
    
    # Create 6-subplot distribution plots
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    
    # Define metrics data, labels, and colors
    metrics_data = [reg_sums, reg_counts, reg_returns, thr_sums, thr_counts, thr_returns]
    metric_labels = ['REG_UNIQUE_SUM', 'REG_NUM_PREDS_UNIQUE', 'REG_RETURN_PER_TRADE', 
                     'THR_UNIQUE_SUM', 'THR_NUM_PREDS_UNIQUE', 'THR_RETURN_PER_TRADE']
    colors = ['blue', 'lightblue', 'navy', 'red', 'lightcoral', 'darkred']
    
    for i, (ax, data, label, color) in enumerate(zip(axes, metrics_data, metric_labels, colors)):
        # Plot histogram
        ax.hist(data, bins=bins, alpha=0.7, color=color, edgecolor='black')
        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{model_name}\n{label}', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add stats text
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.text(0.05, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nN: {len(data)}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
    
    fig.suptitle(f'{title_prefix}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    
    plt.show()
    
    return {
        'reg_sums': reg_sums,
        'reg_counts': reg_counts, 
        'reg_returns': reg_returns,
        'thr_sums': thr_sums,
        'thr_counts': thr_counts,
        'thr_returns': thr_returns
    }


def plot_all_models_realization_distributions(
    realizations_list,
    title_prefix="Distribution of Metrics Across Realizations",
    bins=10,
    save_path_prefix=None
):
    """
    Plot distributions of all 6 metrics for all models across multiple realizations.
    
    Parameters:
    -----------
    realizations_list : list of dict
        List of dictionaries, each representing one realization of model results
    title_prefix : str
        Prefix for the plot titles
    bins : int
        Number of bins for histograms
    save_path_prefix : str, optional
        Prefix for save paths (will append model name)
    """
    
    # Get all unique model names from all realizations
    all_models = []
    for realization_dict in realizations_list:
        for key in realization_dict.keys():
            if key not in all_models:
                all_models.append(key)

    print(f"Found {len(all_models)} models: {all_models}")
    print(f"Across {len(realizations_list)} realizations")
    
    # Plot distributions for each model
    results = {}
    for model_name in all_models:

        print(f"\nPlotting distributions for model: {model_name}")
        save_path = f"{save_path_prefix}_{model_name}.png" if save_path_prefix else None
        model_metrics = plot_model_realization_distributions(
            realizations_list, model_name, title_prefix, bins, save_path
        )
        results[model_name] = model_metrics
    
    return results

