# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import json
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
import seaborn as sns
from plot_style import apply_style

colors = apply_style()

# %%
models = [
    dict(model_name="Gemma 2 2B", 
         model_id="google/gemma-2-2b-it"),
    dict(model_name="Gemma 2 9B",
         model_id="google/gemma-2-9b-it"),
    dict(model_name="Llama 3 8B",
         model_id="meta-llama/Meta-Llama-3-8B-Instruct"),
    dict(model_name="Qwen 2.5 1.5B",
         model_id="Qwen/Qwen2.5-1.5B-Instruct"),
    dict(model_name="Qwen 2.5 3B",
         model_id="Qwen/Qwen2.5-3B-Instruct"),
    dict(model_name="Qwen 2.5 7B",
         model_id="Qwen/Qwen2.5-7B-Instruct"),
    dict(model_name="Qwen 2.5 14B",
         model_id="Qwen/Qwen2.5-14B-Instruct"),
]

# load baseline scores
dim_dir = os.path.join("/ceph/hdd/students/elsj/paper_results/dim_directions")
for model in models:
    print(model)
    model_id = model["model_id"].split("/")[-1]
    datasets = ["jailbreakbench", "strongreject", "sorrybench", "xstest"]
    for dataset in datasets:
        key = 'StrongREJECT_score' if dataset != "xstest" else "xstest_judgements"
        path = os.path.join(dim_dir, f"{model_id}/completions/{dataset}_ablation_evaluations.json")
        with open(path, "r") as f:
            data = json.load(f)
        model[f"{dataset}_ablation_asr"] = data[key]
        path = os.path.join(dim_dir, f"{model_id}/completions/{dataset}_baseline_evaluations.json")
        with open(path, "r") as f:
            data = json.load(f)
        model[f"{dataset}_baseline_asr"] = data[key]
        path = os.path.join(dim_dir, f"{model_id}/completions/{dataset}_actadd_evaluations.json")
        with open(path, "r") as f:
            data = json.load(f)
        model[f"{dataset}_actadd_asr"] = data[key]
    path = os.path.join(dim_dir, f"{model_id}/completions/harmless_actadd_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["harmless_actadd_asr"] = data["substring_matching_success_rate"]
    path = os.path.join(dim_dir, f"{model_id}/completions/harmless_baseline_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["harmless_baseline_asr"] = data["substring_matching_success_rate"]
    path = os.path.join(dim_dir, f"{model_id}/direction.pt")
    refusal_direction = torch.load(path, map_location=torch.device('cpu'))
    model["refusal_direction"] = refusal_direction.clone()

    api = wandb.Api()
    groups = [
        dict(group="sb_data_with_retain", name="retain"),
    ]
    for group in groups:
        group_name = f"{group['group']}_{model_id}"
        run_name = "run_4" if (model_id == "Qwen2.5-3B-Instruct" and "retain" in group_name) else "run_5" # run 5 crashed for Qwen2.5-3B-Instruct
        runs = api.runs("refusal-representations/robust_refusal_vector", {"group": group_name, "display_name": run_name})
        if runs:
            run = runs[0]
            summary = run.summary._json_dict
            key = "retain_summary"
            model[f"{key}"] = summary
# %%
def plot_scores(y_label: str, dataset: str):
    # Extract scores from models
    dataset_key = dataset.lower().replace("-", "")
    ablation_scores = [model[f"{dataset_key}_ablation_asr"] for model in models]
    actadd_scores = [model[f"{dataset_key}_actadd_asr"] for model in models]
    baseline_scores = [model[f"{dataset_key}_baseline_asr"] for model in models]
    if 'xstest' in dataset_key: 
        retain_ablation_scores = [model[f"retain_summary"][f"{dataset_key}_ablation_xstest_judgements"] for model in models]
        retain_actadd_scores = [model[f"retain_summary"][f"{dataset_key}_actadd_xstest_judgements"] for model in models]
    else:
        retain_ablation_scores = [model[f"retain_summary"][f"{dataset_key}_ablation_StrongREJECT_score"] for model in models]
        retain_actadd_scores = [model[f"retain_summary"][f"{dataset_key}_actadd_StrongREJECT_score"] for model in models]
    model_names = [model["model_name"] for model in models]

    x = np.arange(len(models))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.12  # Reduced width to accommodate more bars

    # Create proxy artists for the legend
    from matplotlib.patches import Patch, Rectangle
    
    # Method labels (solid colors)
    method_patches = [
        Rectangle((0,0), 1, 1, facecolor='gray', label='Baseline\n(No Intervention)', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[0], label='DIM', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[1], label='RDO (Ours)', edgecolor='black', linewidth=0.5)
    ]
    
    # Condition labels (patterns)
    condition_patches = [
        Rectangle((0,0), 1, 1, facecolor='white', label='Directional\nAblation', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor='white', label='Activation\nSubtraction', edgecolor='black', linewidth=0.5, hatch='////')
    ]

    # Plot baseline scores
    ax.bar(x - 3*bar_width/2, baseline_scores, bar_width,
        color='gray',
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Plot ablation scores side by side
    ax.bar(x - bar_width/2, ablation_scores, bar_width,
        color=colors[0],
        alpha=1.0, edgecolor='black', linewidth=0.5)
    
    ax.bar(x + bar_width/2, retain_ablation_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Plot actadd scores side by side with hatching
    ax.bar(x + 3*bar_width/2, actadd_scores, bar_width,
        color=colors[0],
        alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

    ax.bar(x + 5*bar_width/2, retain_actadd_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

    # Customization
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='center')  # Rotated labels
    ax.tick_params(axis='both', which='major')

    # Grid styling
    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray', linewidth=0.5)

    # Adjust spines to be thinner
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Make tick marks thinner and shorter
    ax.tick_params(axis='both', width=0.5, length=3)

    # Y-axis limits and spines
    ax.set_ylim(0, 1)  # Slightly higher to accommodate error bars
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Create two-part legend with better positioning
    legend1 = ax.legend(handles=method_patches, title='Method', 
                    bbox_to_anchor=(1.0, 1.0),  # Position to the right
                    loc='upper left', ncol=1,
                    frameon=True, fancybox=False, 
                    edgecolor='black',
                    borderaxespad=0)
    ax.add_artist(legend1)  # Add first legend
    
    ax.legend(handles=condition_patches, title='Operation',
              bbox_to_anchor=(1.0, 0.5),  # Position to the right, below the first legend
              loc='upper left', ncol=1,
              frameon=True, fancybox=False,
              edgecolor='black', 
              borderaxespad=0)
    ax.set_title(f"{dataset}")
    # Save
    # plt.tight_layout()
    os.makedirs("results/plots/rdo/combined", exist_ok=True)
    path = f"results/plots/rdo/combined/{dataset}_scores.png"
    plt.savefig(path, dpi=300)
    plt.show()

plot_scores(y_label="Attack Success Rate", dataset="JailbreakBench")
plot_scores(y_label="Attack Success Rate", dataset="StrongREJECT")
plot_scores(y_label="Attack Success Rate", dataset="SORRY-Bench")
plot_scores(y_label="Refusal Rate", dataset="XSTest")

# %% 
# plot the asr but reversed and call it safety score
def plot_safety_scores(y_label: str):
    # Extract scores from models
    ablation_scores = [1 - model["ablation_asr"] for model in models]
    actadd_scores = [1 - model["actadd_asr"] for model in models]
    baseline_scores = [1 - model["baseline_asr"] for model in models]
    retain_ablation_scores = [1 - model[f"retain_summary"]["jailbreakbench_ablation_StrongREJECT_score"] for model in models]
    retain_actadd_scores = [1 - model[f"retain_summary"]["jailbreakbench_actadd_StrongREJECT_score"] for model in models]
    model_names = [model["model_name"] for model in models]

    x = np.arange(len(models))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.12  # Reduced width to accommodate more bars

    # Create proxy artists for the legend
    from matplotlib.patches import Patch, Rectangle
    
    # Method labels (solid colors)
    method_patches = [
        Rectangle((0,0), 1, 1, facecolor='gray', label='Baseline\n(No Intervention)', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[0], label='DIM', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[1], label='RDO (Ours)', edgecolor='black', linewidth=0.5)
    ]
    
    # Condition labels (patterns)
    condition_patches = [
        Rectangle((0,0), 1, 1, facecolor='white', label='Directional\nAblation', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor='white', label='Activation\nSubtraction', edgecolor='black', linewidth=0.5, hatch='////')
    ]

    # Plot baseline scores
    ax.bar(x - 3*bar_width/2, baseline_scores, bar_width,
        color='gray',
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Plot ablation scores side by side
    ax.bar(x - bar_width/2, ablation_scores, bar_width,
        color=colors[0],
        alpha=1.0, edgecolor='black', linewidth=0.5)
    
    ax.bar(x + bar_width/2, retain_ablation_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Plot actadd scores side by side with hatching
    ax.bar(x + 3*bar_width/2, actadd_scores, bar_width,
        color=colors[0],
        alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

    ax.bar(x + 5*bar_width/2, retain_actadd_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

    # Customization
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='center')  # Rotated labels
    ax.tick_params(axis='both', which='major')

    # Grid styling
    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray', linewidth=0.5)

    # Adjust spines to be thinner
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Make tick marks thinner and shorter
    ax.tick_params(axis='both', width=0.5, length=3)

    # Y-axis limits and spines
    ax.set_ylim(0, 1)  # Slightly higher to accommodate error bars
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Create two-part legend with better positioning
    legend1 = ax.legend(handles=method_patches, title='Method', 
                    bbox_to_anchor=(1.0, 1.0),  # Position to the right
                    loc='upper left', ncol=1,
                    frameon=True, fancybox=False, 
                    edgecolor='black',
                    borderaxespad=0)
    ax.add_artist(legend1)  # Add first legend
    
    ax.legend(handles=condition_patches, title='Operation',
              bbox_to_anchor=(1.0, 0.5),  # Position to the right, below the first legend
              loc='upper left', ncol=1,
              frameon=True, fancybox=False,
              edgecolor='black', 
              borderaxespad=0)

    # Save
    # plt.tight_layout()
    os.makedirs("results/plots/single_direction/combined", exist_ok=True)
    path = "results/plots/single_direction/combined/jailbreak_scores.png"
    plt.savefig(path, dpi=300)
    plt.show()

plot_safety_scores(y_label="Safety Score")

# %%
from tabulate import tabulate

def create_latex_table():
    # Start of the LaTeX table
    latex_output = [
        "\\begin{table*}[htpb]",
        "\\small",
        "\\centering",
        "\\begin{tabular}{ l c c c c c c c }",
        "\\toprule"
    ]
    
    # Multirow headers
    latex_output.append("\\multirow{2}{*}{ } & \\multicolumn{3}{c}{\\textbf{Jailbreaking}} & " +
                       "\\multicolumn{4}{c}{\\textbf{General Capability}} \\\\")
    
    # Cmidrule separators
    latex_output.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-8}")
    
    # Subheaders
    latex_output.append(" & \\textbf{JailbreakBench} & \\textbf{StrongREJECT} & \\textbf{SORRY-Bench} & " +
                       "\\textbf{MMLU} & \\textbf{ARC-C} & \\textbf{GSM8K} & \\textbf{TruthfulQA} \\\\")
    
    # Direction indicators
    latex_output.append(" & ASR \\(\\uparrow\\) & ASR \\(\\uparrow\\) & ASR \\(\\uparrow\\) & " +
                       "Acc \\(\\uparrow\\) & Acc \\(\\uparrow\\) & Acc \\(\\uparrow\\) & Acc \\(\\uparrow\\) \\\\")
    
    # Midrule
    latex_output.append("\\midrule")
    
    for model in models:
        model_name = model["model_name"]
        
        # Base model row
        base_mmlu = model.get('baseline_mmlu_acc', 0)*100
        base_arc = model.get('baseline_arc_challenge_acc', 0)*100
        base_gsm8k = model.get('baseline_gsm8k_acc', 0)*100
        base_truthfulqa = model.get('baseline_truthfulqa_mc2_acc', 0)*100
        avg_benchmark = (base_mmlu + base_arc + base_gsm8k + base_truthfulqa) / 4
        
        latex_output.append(f"\\textsc{{{model_name}}} & " +
                          f"{model.get('jailbreakbench_baseline_asr', 0)*100:.1f} & " +
                          f"{model.get('strongreject_baseline_asr', 0)*100:.1f} & " +
                          f"{model.get('sorrybench_baseline_asr', 0)*100:.1f} & " +
                          f"{base_mmlu:.1f} & " +
                          f"{base_arc:.1f} & " +
                          f"{base_gsm8k:.1f} & " +
                          f"{base_truthfulqa:.1f} \\\\")
        
        # DIM row (system prompt)
        dim_jbb_dir = model.get('jailbreakbench_ablation_asr', 0)*100
        dim_jbb_act = model.get('jailbreakbench_actadd_asr', 0)*100
        dim_strong_dir = model.get('strongreject_ablation_asr', 0)*100
        dim_strong_act = model.get('strongreject_actadd_asr', 0)*100
        dim_sorry_dir = model.get('sorrybench_ablation_asr', 0)*100
        dim_sorry_act = model.get('sorrybench_actadd_asr', 0)*100
        
        dim_mmlu = model.get('original_mmlu_acc', 0)*100
        dim_arc = model.get('original_arc_challenge_acc', 0)*100
        dim_gsm8k = model.get('original_gsm8k_acc', 0)*100
        dim_truthfulqa = model.get('original_truthfulqa_mc2_acc', 0)*100
        
        # RDO values for comparison
        rdo_jbb_dir = 0
        rdo_jbb_act = 0
        rdo_strong_dir = 0
        rdo_strong_act = 0
        rdo_sorry_dir = 0
        rdo_sorry_act = 0
        rdo_mmlu = model.get('wandb_mmlu_acc', 0)*100
        rdo_arc = model.get('wandb_arc_challenge_acc', 0)*100
        rdo_gsm8k = model.get('wandb_gsm8k_acc', 0)*100
        rdo_truthfulqa = model.get('wandb_truthfulqa_mc2_acc', 0)*100
        
        if "retain_summary" in model:
            rdo_jbb_dir = model['retain_summary'].get('jailbreakbench_ablation_StrongREJECT_score', 0)*100
            rdo_jbb_act = model['retain_summary'].get('jailbreakbench_actadd_StrongREJECT_score', 0)*100
            rdo_strong_dir = model['retain_summary'].get('strongreject_ablation_StrongREJECT_score', 0)*100
            rdo_strong_act = model['retain_summary'].get('strongreject_actadd_StrongREJECT_score', 0)*100
            rdo_sorry_dir = model['retain_summary'].get('sorrybench_ablation_StrongREJECT_score', 0)*100
            rdo_sorry_act = model['retain_summary'].get('sorrybench_actadd_StrongREJECT_score', 0)*100
        
        # Bold DIM values if better than RDO for safety metrics
        dim_jbb_dir_text = f"\\textbf{{{dim_jbb_dir:.1f}}}" if dim_jbb_dir > rdo_jbb_dir else f"{dim_jbb_dir:.1f}"
        dim_jbb_act_text = f"\\textbf{{{dim_jbb_act:.1f}}}" if dim_jbb_act > rdo_jbb_act else f"{dim_jbb_act:.1f}"
        dim_strong_dir_text = f"\\textbf{{{dim_strong_dir:.1f}}}" if dim_strong_dir > rdo_strong_dir else f"{dim_strong_dir:.1f}"
        dim_strong_act_text = f"\\textbf{{{dim_strong_act:.1f}}}" if dim_strong_act > rdo_strong_act else f"{dim_strong_act:.1f}"
        dim_sorry_dir_text = f"\\textbf{{{dim_sorry_dir:.1f}}}" if dim_sorry_dir > rdo_sorry_dir else f"{dim_sorry_dir:.1f}"
        dim_sorry_act_text = f"\\textbf{{{dim_sorry_act:.1f}}}" if dim_sorry_act > rdo_sorry_act else f"{dim_sorry_act:.1f}"
        
        # For capability metrics, bold the highest score
        dim_mmlu_text = f"\\textbf{{{dim_mmlu:.1f}}}" if dim_mmlu > rdo_mmlu else f"{dim_mmlu:.1f}"
        dim_arc_text = f"\\textbf{{{dim_arc:.1f}}}" if dim_arc > rdo_arc else f"{dim_arc:.1f}"
        dim_gsm8k_text = f"\\textbf{{{dim_gsm8k:.1f}}}" if dim_gsm8k > rdo_gsm8k else f"{dim_gsm8k:.1f}"
        dim_truthfulqa_text = f"\\textbf{{{dim_truthfulqa:.1f}}}" if dim_truthfulqa > rdo_truthfulqa else f"{dim_truthfulqa:.1f}"
        
        latex_output.append(f"\\multicolumn{{1}}{{c}}{{\\: DIM}} & " +
                          f"{dim_jbb_dir_text} / {dim_jbb_act_text} & " +
                          f"{dim_strong_dir_text} / {dim_strong_act_text} & " +
                          f"{dim_sorry_dir_text} / {dim_sorry_act_text} & " +
                          f"{dim_mmlu_text} & " +
                          f"{dim_arc_text} & " +
                          f"{dim_gsm8k_text} & " +
                          f"{dim_truthfulqa_text} \\\\")
        
        # RDO row (vector ablation)
        if "retain_summary" in model:
            jbb_dir_text = f"\\textbf{{{rdo_jbb_dir:.1f}}}" if rdo_jbb_dir > dim_jbb_dir else f"{rdo_jbb_dir:.1f}"
            jbb_act_text = f"\\textbf{{{rdo_jbb_act:.1f}}}" if rdo_jbb_act > dim_jbb_act else f"{rdo_jbb_act:.1f}"
            strong_dir_text = f"\\textbf{{{rdo_strong_dir:.1f}}}" if rdo_strong_dir > dim_strong_dir else f"{rdo_strong_dir:.1f}"
            strong_act_text = f"\\textbf{{{rdo_strong_act:.1f}}}" if rdo_strong_act > dim_strong_act else f"{rdo_strong_act:.1f}"
            sorry_dir_text = f"\\textbf{{{rdo_sorry_dir:.1f}}}" if rdo_sorry_dir > dim_sorry_dir else f"{rdo_sorry_dir:.1f}"
            sorry_act_text = f"\\textbf{{{rdo_sorry_act:.1f}}}" if rdo_sorry_act > dim_sorry_act else f"{rdo_sorry_act:.1f}"
            
            jbb_cell = f"{jbb_dir_text} / {jbb_act_text}"
            strong_cell = f"{strong_dir_text} / {strong_act_text}"
            sorry_cell = f"{sorry_dir_text} / {sorry_act_text}"
        else:
            jbb_cell = "N/A"
            strong_cell = "N/A"
            sorry_cell = "N/A"
        
        mmlu_text = f"\\textbf{{{rdo_mmlu:.1f}}}" if rdo_mmlu > dim_mmlu else f"{rdo_mmlu:.1f}"
        arc_text = f"\\textbf{{{rdo_arc:.1f}}}" if rdo_arc > dim_arc else f"{rdo_arc:.1f}"
        gsm8k_text = f"\\textbf{{{rdo_gsm8k:.1f}}}" if rdo_gsm8k > dim_gsm8k else f"{rdo_gsm8k:.1f}"
        truthfulqa_text = f"\\textbf{{{rdo_truthfulqa:.1f}}}" if rdo_truthfulqa > dim_truthfulqa else f"{rdo_truthfulqa:.1f}"
        
        latex_output.append(f"\\multicolumn{{1}}{{c}}{{\\: RDO}} & " +
                           f"{jbb_cell} & {strong_cell} & {sorry_cell} & " +
                           f"{mmlu_text} & {arc_text} & {gsm8k_text} & {truthfulqa_text} \\\\")
        
        # Add midrule between models (except after the last one)
        if model != models[-1]:
            latex_output.append("\\midrule")
    
    # Bottom rule and table end
    latex_output.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "\\caption{Combined evaluation results showing safety metrics (DirectionalAblation/ActivationSubtraction) and general capability benchmarks. Each safety cell contains two values: the first for directional ablation and the second for activation subtraction.}",
        "\\label{tab:combined_evaluation}",
        "\\end{table*}"
    ])
    
    # Join and print the LaTeX code
    latex_table = "\n".join(latex_output)
    print(latex_table)
    
    # Save the LaTeX table to a file
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/combined_evaluation.tex", "w") as f:
        f.write(latex_table)

# Call the function to create the LaTeX table
create_latex_table()

# %%
def plot_harmless_scores(y_label: str):
    # Extract scores from models
    ablation_scores = [1 - model["harmless_actadd_asr"] for model in models]  # Changed to harmless scores
    retain_ablation_scores = [1 - model[f"retain_summary"]["harmless_actadd_substring_matching_success_rate"] for model in models]

    model_names = [model["model_name"] for model in models]

    x = np.arange(len(models))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.15  # Slightly wider since we have fewer bars

    # Create proxy artists for the legend
    from matplotlib.patches import Patch, Rectangle
    
    # Method labels (solid colors)
    method_patches = [
        Rectangle((0,0), 1, 1, facecolor=colors[0], label='DIM', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[1], label='RDO (Ours)', edgecolor='black', linewidth=0.5)
    ]

    # Plot bars without transparency
    ax.bar(x - bar_width/2, ablation_scores, bar_width, 
        color=colors[0], 
        alpha=1.0, edgecolor='black', linewidth=0.5)
    
    ax.bar(x + bar_width/2, retain_ablation_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Customization
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='center')  # Rotated labels
    ax.tick_params(axis='both', which='major')

    # Grid styling
    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray', linewidth=0.5)

    # Adjust spines to be thinner
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Make tick marks thinner and shorter
    ax.tick_params(axis='both', width=0.5, length=3)

    # Y-axis limits and spines
    ax.set_ylim(0, 1)  # Slightly higher to accommodate error bars
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Create legend
    legend = ax.legend(handles=method_patches,
                    bbox_to_anchor=(0.5, 1.15),
                    loc='center', ncol=2,
                    frameon=True, fancybox=False, 
                    edgecolor='black',
                    borderaxespad=0)

    # Save
    os.makedirs("results/plots/single_direction/combined", exist_ok=True)
    path = "results/plots/single_direction/combined/harmless_scores.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')

# Call the function
plot_harmless_scores(y_label="Refusal Score")
# %%
print(models[-1])
# %%
for model in models:
    print(model["model_name"], model["harmless_actadd_asr"])

# %%
models = [
    # dict(model_name="Gemma 2 2B", 
    #      model_id="google/gemma-2-2b-it"),
    # dict(model_name="Gemma 2 9B",
    #      model_id="google/gemma-2-9b-it"),
    dict(model_name="Qwen 2.5 1.5B",
         model_id="Qwen/Qwen2.5-1.5B-Instruct"),
    dict(model_name="Qwen 2.5 3B",
         model_id="Qwen/Qwen2.5-3B-Instruct"),
    dict(model_name="Qwen 2.5 7B",
         model_id="Qwen/Qwen2.5-7B-Instruct"),
    dict(model_name="Qwen 2.5 14B",
         model_id="Qwen/Qwen2.5-14B-Instruct"),
    # dict(model_name="Llama 3 8B",
    #      model_id="meta-llama/Meta-Llama-3-8B-Instruct"),
]
base_dir = "/ceph/hdd/students/elsj/paper_results/subspace_samples_eval"
api = wandb.Api()

# Get all runs once
base_runs = {}
for model in models:
    model_id = model["model_id"].split("/")[-1]
    base_group_name = "join_subspace"
    # base_group_name = "repind_subspace_retain1_repind200_fixed_sum_no_dim"
    base_runs[model_id] = api.runs("refusal-representations/robust_refusal_subspace", 
                                 {"group": f"{base_group_name}_{model_id}"})

for model in models:
    model_id = model["model_id"].split("/")[-1]
    for dim in range(2, 9):
        matching_runs = []
        for r in base_runs[model_id]:
            if r.display_name == f"dim_{dim}":
                matching_runs.append(r)
        if not matching_runs:
            print(f"No runs found for dim_{dim} for model {model_id}")
            continue
            
        run = matching_runs[-1]
        summary = run.summary._json_dict
        if "candidate_idx" not in summary and "use_repind_loss" not in summary:
            print(f"No candidate_idx found for dim_{dim} for model {model_id}")
            continue
            
        basis_scores = []
        basis_scores_harmless = []
        for basis_dim in range(1, dim + 1):
            basis_scores.append(summary[f"jailbreakbench_ablation_basis_{basis_dim}_StrongREJECT_score"])
            basis_scores_harmless.append(summary[f"harmless_actadd_basis_{basis_dim}_substring_matching_success_rate"])
        model[f"dim_{dim}_basis"] = basis_scores
        model[f"dim_{dim}_basis_harmless"] = basis_scores_harmless

# %%
import os
import json
from concurrent.futures import ThreadPoolExecutor

def read_file_field(filepath, field):
    with open(filepath) as f:
        data = json.load(f)
    return data[field]

def load_evaluations(model_path):
    """Load all evaluation files for a model path concurrently and efficiently."""
    model_data = {}
    missing_samples = {}
    
    with ThreadPoolExecutor() as executor:
        futures = []
        # Iterate through dimensions we expect to find (dim 2 to 8)
        for dim in range(2, 9):
            attack_key = f"dim_{dim}_samples"
            harmless_key = f"dim_{dim}_samples_harmless"
            model_data[attack_key] = [None] * 256
            model_data[harmless_key] = [None] * 256
            missing_samples[attack_key] = set(range(1, 257))
            missing_samples[harmless_key] = set(range(1, 257))
            
            # List the completions directory once per dimension
            completions_dir = os.path.join(model_path, f"dim_{dim}", "completions")
            if not os.path.isdir(completions_dir):
                continue
            try:
                files = os.listdir(completions_dir)
            except Exception as e:
                print(f"Error listing {completions_dir}: {e}")
                continue
            
            for file in files:
                # Process attack samples
                if file.startswith("jailbreakbench_ablation_sample_") and file.endswith("_evaluations.json"):
                    sample_str = file[len("jailbreakbench_ablation_sample_"):-len("_evaluations.json")]
                    try:
                        sample_num = int(sample_str)
                    except ValueError:
                        continue
                    file_path = os.path.join(completions_dir, file)
                    futures.append((executor.submit(read_file_field, file_path, "StrongREJECT_score"), attack_key, sample_num))
                # Process harmless samples
                elif file.startswith("harmless_actadd_sample_") and file.endswith("_evaluations.json"):
                    sample_str = file[len("harmless_actadd_sample_"):-len("_evaluations.json")]
                    try:
                        sample_num = int(sample_str)
                    except ValueError:
                        continue
                    file_path = os.path.join(completions_dir, file)
                    futures.append((executor.submit(read_file_field, file_path, "substring_matching_success_rate"), harmless_key, sample_num))
        
        # Gather results from all concurrent file reads
        for future, key, sample_num in futures:
            try:
                value = future.result()
            except Exception:
                value = None
            if value is not None:
                if 1 <= sample_num <= len(model_data[key]):
                    model_data[key][sample_num - 1] = value
                    missing_samples[key].discard(sample_num)
                else:
                    print(f"Warning: sample number {sample_num} is out of range for {key} in {model_path}. Skipping assignment.")
    
    # Remove dimensions with no data and filter out missing entries
    keys_to_remove = []
    for key in list(model_data.keys()):
        if all(x is None for x in model_data[key]):
            print(f"No data found for {key} for {model_path}")
            keys_to_remove.append(key)
            missing_samples.pop(key, None)
        else:
            model_data[key] = [s for s in model_data[key] if s is not None]
    for key in keys_to_remove:
        model_data.pop(key, None)
    
    # Report any missing samples
    for key, missing in missing_samples.items():
        if missing:
            model_name = os.path.basename(model_path)
            print(f"Model {model_name} - Missing samples for {key}: {sorted(missing)}")
            
    return model_data

# Process each model
for model in models:
    model_id = model["model_id"].split("/")[-1]
    dir_name = f"join_subspace_{model_id}" if "repind" not in base_group_name else f"repind_subspace_retain1_repind200_fixed_sum_no_dim_{model_id}"
    samples_folder = os.path.join(base_dir, dir_name)
    
    # Load all evaluations for this model concurrently
    model_data = load_evaluations(samples_folder)
    
    if not any(key.endswith("_samples") or key.endswith("_samples_harmless") for key in model_data.keys()):
        print(f"Model {model['model_name']} Warning: No samples found")
        continue
        
    model.update(model_data)
    
    for key, value in model.items():
        if key.endswith("_samples") or key.endswith("_samples_harmless"):
            if len(value) != 256:
                print(f"Model {model['model_name']} Warning: {key} has {len(value)} samples instead of expected 256")

# %%
for model in models:
    print(model)
    model_id = model["model_id"].split("/")[-1]
    path = os.path.join(f"results/refusal_dir/{model_id}/completions/jailbreakbench_ablation_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["ablation_asr"] = data["StrongREJECT_score"]
    model["dim_1_basis"] = [data["StrongREJECT_score"]]  # Add dim 1 baseline
    path = os.path.join(f"results/refusal_dir/{model_id}/completions/jailbreakbench_baseline_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["baseline_asr"] = data["StrongREJECT_score"]
    path = os.path.join(f"results/refusal_dir/{model_id}/completions/jailbreakbench_actadd_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["actadd_asr"] = data["StrongREJECT_score"]
    path = os.path.join(f"results/refusal_dir/{model_id}/completions/harmless_actadd_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["harmless_actadd_asr"] = data["substring_matching_success_rate"]
    model["dim_1_basis_harmless"] = [data["substring_matching_success_rate"]]  # Add dim 1 harmless baseline
    path = os.path.join(f"results/refusal_dir/{model_id}/completions/harmless_baseline_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["harmless_baseline_asr"] = data["substring_matching_success_rate"]
    path = os.path.join(f"results/refusal_dir/{model_id}/direction.pt")
    refusal_direction = torch.load(path, map_location=torch.device('cpu'))
    model["refusal_direction"] = refusal_direction.clone()

    api = wandb.Api()
    groups = [
        dict(group="sb_data_with_retain", name="retain"),
    ]
    for group in groups:
        group_name = f"{group['group']}_{model_id}"
        run_name = "run_4" if (model_id == "Qwen2.5-3B-Instruct" and "retain" in group_name) else "run_5"
        runs = api.runs("refusal-representations/robust_refusal_vector", {"group": group_name, "display_name": run_name})
        if runs:
            run = runs[0]
            summary = run.summary._json_dict
            key = "retain_summary"
            model[f"{key}"] = summary

# %%
for key in models[0].keys():
    print(key)
# %%
# Find min and max dimensions from available data
# Group models by family
# Configure font sizes

model_families = {}
for model in models:
    family = model["model_id"].split("/")[0]
    # Rename google family to Gemma
    if family == "google":
        family = "Gemma"
    elif family == "meta-llama":
        family = "Llama 3.1"
    if family not in model_families:
        model_families[family] = []
    model_families[family].append(model)

# Create a plot for each model family
min_dim = 2
qwen_max_dim = 7
gemma_max_dim = 5
llama_max_dim = 5
for family, family_models in model_families.items():
    # Find min and max dimensions from available data across all models in family
    if family == "Qwen":
        max_dim = qwen_max_dim
    elif family == "Gemma":
        max_dim = gemma_max_dim
    elif family == "Llama 3.1":
        max_dim = llama_max_dim
    subspace_dims = list(range(min_dim, max_dim + 1))

    # Define colors
    # model_colors = ['#E67E22'] 

    # Create attack success plot - make it wider
    plot_size = (12, 5)  # Increased width to accommodate right legend
    fig1 = plt.figure(figsize=plot_size)
    ax1 = fig1.add_subplot(111)

    # Create harmless scores plot
    fig2 = plt.figure(figsize=plot_size)
    ax2 = fig2.add_subplot(111)

    # Increase group spacing
    group_spacing = 1.25
    box_spacing = 0.2

    # Plot each model in the family
    for model_idx, model in enumerate(family_models):
        # Calculate offset for this model's boxes
        offset = model_idx * box_spacing - (len(family_models)-1) * box_spacing/2
        positions = [x * group_spacing + offset for x in range(len(subspace_dims))]

        if "dim_1_basis" in model and "repind" not in base_group_name:
            # Plot dim 1 star with model color and black edge
            dim1_pos = -group_spacing + offset
            ax1.scatter([dim1_pos], model["dim_1_basis"], 
                    color=colors[model_idx],
                    marker='o',
                    s=10,
                    zorder=2,
                    alpha=1.0,
                    edgecolors='black',
                    linewidth=0.5)
            ax2.scatter([dim1_pos], [1 - x for x in model["dim_1_basis_harmless"]], 
                    color=colors[model_idx],
                    marker='o',
                    s=10,
                    zorder=2,
                    alpha=1.0,
                    edgecolors='black',
                    linewidth=0.5)

        # Extract performances for both plots
        performances = []
        performances_harmless = []
        basis_performances = []
        basis_performances_harmless = []
        for dim in range(min_dim, max_dim + 1):
            # Get basis data
            basis = model.get(f"dim_{dim}_basis", [])
            basis_harmless = [1 - x for x in model.get(f"dim_{dim}_basis_harmless", [])]
            basis_performances.append(basis)
            basis_performances_harmless.append(basis_harmless)
            
            # Get sample data if available
            samples = model.get(f"dim_{dim}_samples", [])
            samples_harmless = [1 - x for x in model.get(f"dim_{dim}_samples_harmless", [])]
            performances.append(samples)
            performances_harmless.append(samples_harmless)

        # Only plot boxplots if we have sample data
        if any(len(p) > 0 for p in performances):
            # Plot on first subplot (attack success) - reduce outlier visibility
            bp1 = ax1.boxplot(performances, positions=positions, 
                            widths=0.12,  # Slightly reduced width
                            patch_artist=True,
                            medianprops={'color': 'black', 'linewidth': 1},
                            flierprops={'marker': '.', 
                                      'markerfacecolor': colors[model_idx], 
                                      'markersize': 4,
                                      'alpha': 0.3,
                                      'markeredgecolor': 'none'},
                            whiskerprops={'linewidth': 1},
                            boxprops={'facecolor': colors[model_idx], 
                                    'alpha': 0.8,
                                    'linewidth': 1})

            # Plot on second subplot (harmless scores)
            bp2 = ax2.boxplot(performances_harmless, positions=positions, 
                            widths=0.12,
                            patch_artist=True,
                            medianprops={'color': 'black', 'linewidth': 1},
                            flierprops={'marker': '.', 
                                      'markerfacecolor': colors[model_idx], 
                                      'markersize': 4,
                                      'alpha': 0.3,
                                      'markeredgecolor': 'none'},
                            whiskerprops={'linewidth': 1},
                            boxprops={'facecolor': colors[model_idx], 
                                    'alpha': 0.8,
                                    'linewidth': 1})

        # Plot basis vectors on both subplots
        basis_scatter1 = None
        basis_scatter2 = None
        for i, (basis, basis_harmless) in enumerate(zip(basis_performances, basis_performances_harmless)):
            if not basis:  # Skip if no basis data for this dimension
                continue
            scatter1 = ax1.scatter([positions[i]] * len(basis), basis, 
                       color=colors[model_idx],
                       marker='o', 
                       s=10,  
                       zorder=2,
                       alpha=1.0,
                       edgecolors='black',
                       linewidth=0.5)
            scatter2 = ax2.scatter([positions[i]] * len(basis_harmless), basis_harmless,
                       color=colors[model_idx],
                       marker='o', 
                       s=10,  
                       zorder=2,
                       alpha=1.0,
                       edgecolors='black',
                       linewidth=0.5)
            if i == 0:
                basis_scatter1 = scatter1
                basis_scatter2 = scatter2

        # Collect handles and labels for models
        if any(len(p) > 0 for p in performances):
            if model_idx == 0:
                handles = [bp1['boxes'][0]]
                labels = [f'{model["model_name"]}']
            else:
                handles.append(bp1['boxes'][0])
                labels.append(f'{model["model_name"]}')

    # Create a color-neutral basis vector marker for the legend
    basis_legend_marker = ax1.scatter([], [], 
                                    color='gray',
                                    marker='o',
                                    s=10,
                                    zorder=2,
                                    alpha=1.0,
                                    edgecolors='black',
                                    linewidth=0.5)

    # Add basis vectors to legend last
    if 'handles' in locals():
        handles.append(basis_legend_marker)
        labels.append('Basis Vectors')

    # Customize both plots
    for ax, fig in [(ax1, fig1), (ax2, fig2)]:
        ax.set_xlabel("Cone Dimension")
        # Adjust x-axis ticks for wider spacing
        if "repind" in base_group_name:
            ax.set_xticks([x * group_spacing for x in range(len(subspace_dims))])  # Only subspace dims for repind
            ax.set_xticklabels(subspace_dims)  # Only subspace dims for repind
        else:
            ax.set_xticks([-group_spacing] + [x * group_spacing for x in range(len(subspace_dims))])  # Add dim 1 tick
            ax.set_xticklabels([1] + subspace_dims)  # Add dim 1 label
        ax.tick_params(axis='both', which='major')
        ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', width=0.5, length=3)
        ax.set_ylim(0, 1)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # Add light vertical gridlines to separate dimensions
        ax.grid(axis='x', linestyle='-', alpha=0.1, color='gray', linewidth=0.5)
        
        # Place legend on the right
        if 'handles' in locals():
            legend = ax.legend(handles=handles, labels=labels,
                          frameon=True, fancybox=False, edgecolor='black',
                          loc='center left', bbox_to_anchor=(1.02, 0.5),
                          borderaxespad=0, borderpad=0.5)

        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

    # Set specific y-labels
    ax1.set_ylabel("Attack Success Rate")
    ax2.set_ylabel("Refusal Score")
    
    # Show plots
    plt.figure(fig1.number)
    plt.show()
    plt.figure(fig2.number) 
    plt.show()
    
    # Save plots with extra space for legend
    if "repind" in base_group_name:
        os.makedirs(os.path.dirname(f"results/plots/subspace_performance/{family.lower()}_repind"), exist_ok=True)
        fig1.savefig(f"results/plots/subspace_performance/{family.lower()}_repind_dim_attack.png", dpi=300, bbox_inches='tight')
        fig2.savefig(f"results/plots/subspace_performance/{family.lower()}_repind_dim_harmless.png", dpi=300, bbox_inches='tight')
    else:
        os.makedirs(os.path.dirname(f"results/plots/subspace_performance/{family.lower()}"), exist_ok=True)
        fig1.savefig(f"results/plots/subspace_performance/{family.lower()}_attack.png", dpi=300, bbox_inches='tight')
        fig2.savefig(f"results/plots/subspace_performance/{family.lower()}_harmless.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    plt.close(fig1)

# %%
# Plot comparison between Gemma 9B, Qwen 14B and Llama 8B
selected_models = [
    model for model in models 
    if model["model_name"] in ["Gemma 2 9B", "Qwen 2.5 7B", "Llama 3 8B"]
]

max_dim = 5

min_dim = 2

subspace_dims = list(range(min_dim, max_dim + 1))

# Create attack success plot
figsize = (10, 5)
fig1 = plt.figure(figsize=figsize)
ax1 = fig1.add_subplot(111)

# Create harmless scores plot
fig2 = plt.figure(figsize=figsize)
ax2 = fig2.add_subplot(111)

# Spacing parameters
group_spacing = 1
box_spacing = 0.2

# Plot each model
for model_idx, model in enumerate(selected_models):
    # Calculate offset for this model's boxes
    offset = model_idx * box_spacing - (len(selected_models)-1) * box_spacing/2
    positions = [-group_spacing + offset] + [x * group_spacing + offset for x in range(len(subspace_dims))]  # Add dim 1 position

    # Extract performances for both plots
    performances = []
    performances_harmless = []
    basis_performances = []
    basis_performances_harmless = []
    
    # Add dim 1 basis data
    basis_performances.append(model.get("dim_1_basis", []))
    basis_performances_harmless.append([1 - x for x in model.get("dim_1_basis_harmless", [])])
    
    for dim in range(min_dim, max_dim + 1):
        # Get basis data
        basis = model.get(f"dim_{dim}_basis", [])
        basis_harmless = [1 - x for x in model.get(f"dim_{dim}_basis_harmless", [])]
        basis_performances.append(basis)
        basis_performances_harmless.append(basis_harmless)
        
        # Get sample data if available
        samples = model.get(f"dim_{dim}_samples", [])
        samples_harmless = [1 - x for x in model.get(f"dim_{dim}_samples_harmless", [])]
        performances.append(samples)
        performances_harmless.append(samples_harmless)

    # Only plot boxplots if we have sample data
    if any(len(p) > 0 for p in performances):
        # Plot on first subplot (attack success)
        bp1 = ax1.boxplot(performances, positions=positions[1:], 
                        widths=0.12,
                        patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 1},
                        flierprops={'marker': '.', 
                                  'markerfacecolor': colors[model_idx], 
                                  'markersize': 4,
                                  'alpha': 0.3,
                                  'markeredgecolor': 'none'},
                        whiskerprops={'linewidth': 1},
                        boxprops={'facecolor': colors[model_idx], 
                                'alpha': 0.8,
                                'linewidth': 1})

        # Plot on second subplot (harmless scores)
        bp2 = ax2.boxplot(performances_harmless, positions=positions[1:], 
                        widths=0.12,
                        patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 1},
                        flierprops={'marker': '.', 
                                  'markerfacecolor': colors[model_idx], 
                                  'markersize': 4,
                                  'alpha': 0.3,
                                  'markeredgecolor': 'none'},
                        whiskerprops={'linewidth': 1},
                        boxprops={'facecolor': colors[model_idx], 
                                'alpha': 0.8,
                                'linewidth': 1})

    # Plot basis vectors on both subplots
    basis_scatter1 = None
    basis_scatter2 = None
    for i, (basis, basis_harmless) in enumerate(zip(basis_performances, basis_performances_harmless)):
        if not basis:
            continue
        scatter1 = ax1.scatter([positions[i]] * len(basis), basis, 
                   color='gray',
                   marker='o', 
                   s=10,  
                   zorder=2,
                   alpha=1.0,
                   edgecolors='black',
                   linewidth=0.5)
        scatter2 = ax2.scatter([positions[i]] * len(basis_harmless), basis_harmless,
                   color='gray',
                   marker='o', 
                   s=10,  
                   edgecolors='black',
                   linewidth=0.5,
                   zorder=2,
                   alpha=1.0)
        if i == 0:
            basis_scatter1 = scatter1
            basis_scatter2 = scatter2

    # Collect handles and labels for models
    if any(len(p) > 0 for p in performances):
        if model_idx == 0:
            handles = [bp1['boxes'][0]]
            labels = [f'{model["model_name"]}']
        else:
            handles.append(bp1['boxes'][0])
            labels.append(f'{model["model_name"]}')
        
        if model_idx == 0:
            basis_legend_scatter = basis_scatter1

# Add basis vectors to legend last
if 'handles' in locals():
    handles.append(basis_legend_scatter)
    labels.append('Basis Vectors')

# Customize both plots
for ax, fig in [(ax1, fig1), (ax2, fig2)]:
    ax.set_xlabel("Cone Dimension")
    ax.set_xticks([-group_spacing] + [x * group_spacing for x in range(len(subspace_dims))])  # Add dim 1 tick
    ax.set_xticklabels([1] + subspace_dims)  # Add dim 1 label
    ax.tick_params(axis='both', which='major')
    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(axis='both', width=0.5, length=3)
    ax.set_ylim(0, 1)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Add light vertical gridlines
    ax.grid(axis='x', linestyle='-', alpha=0.1, color='gray', linewidth=0.5)
    
    # Adjust legend position and layout
    if 'handles' in locals():
        legend = ax.legend(handles=handles, labels=labels,
                      frameon=True, fancybox=False, edgecolor='black',
                      loc='center left', bbox_to_anchor=(1.02, 0.5),
                      borderaxespad=0, borderpad=0.5)

    plt.tight_layout()

# Set specific y-labels
ax1.set_ylabel("Attack Success Rate")
ax2.set_ylabel("Refusal Score")

# Save plots
plt.figure(fig1.number)
plt.tight_layout()
plt.show()
plt.figure(fig2.number)
plt.tight_layout()
plt.show()
os.makedirs("results/plots/subspace_performance/model_comparison", exist_ok=True)
fig1.savefig("results/plots/subspace_performance/model_comparison_attack.png", dpi=300, bbox_inches='tight')
fig2.savefig("results/plots/subspace_performance/model_comparison_harmless.png", dpi=300, bbox_inches='tight')

plt.close(fig2)
plt.close(fig1)


            
import glob
dim = 4
model_path = "/ceph/hdd/students/elsj/paper_results/subspace_samples_eval/join_subspace_gemma-2-2b-it/dim_4"
gemma_sample_performances = []
for sample_num in range(1, 257):
    # Check for attack sample
    attack_file = os.path.join(model_path, "completions", f"jailbreakbench_ablation_sample_{sample_num}_evaluations.json")
    attack_files = glob.glob(attack_file)
    if attack_files:
        with open(attack_files[0]) as f:
            data = json.load(f)
        scores = [d["is_jailbreak_strongreject"] for d in data["completions"]]
        gemma_sample_performances.append(scores)
np.array(gemma_sample_performances).shape
# %%
models = [
    dict(model_name="Gemma 2 2B", 
         model_id="google/gemma-2-2b-it",
         run_names=["rep_ind_1_run_1", "rep_ind_2_run_3", "rep_ind_3_run_1", "rep_ind_4_run_1", "rep_ind_5_run_1"],
         group="repind_symmetric_sum_loss_diff_cutoff09_gemma-2-2b-it"),
    dict(model_name="Llama 3 8B",
         model_id="meta-llama/Meta-Llama-3-8B-Instruct",
         run_names=["rep_ind_1_run_3", "rep_ind_2_run_1", "rep_ind_3_run_1", "rep_ind_4_run_3", "rep_ind_5_run_1"],
         group="repind_iterative_Meta-Llama-3-8B-Instruct"),
    dict(model_name="Qwen 2.5 3B",
         model_id="Qwen/Qwen2.5-3B-Instruct",
         run_names=["rep_ind_1_run_3", "rep_ind_2_run_1", "rep_ind_3_run_1", "rep_ind_4_run_3", "rep_ind_5_run_3"],
         group="repind_iterative_Qwen2.5-3B-Instruct"),
    dict(model_name="Qwen 2.5 7B",
         model_id="Qwen/Qwen2.5-7B-Instruct",
         run_names=["rep_ind_1_run_2", "rep_ind_2_run_1", "rep_ind_3_run_1", "rep_ind_4_run_2", "rep_ind_5_run_3"],
         group="repind_iterative_Qwen2.5-7B-Instruct"),
]

# Create figure for plotting ASRs across all models
fig, ax = plt.subplots(figsize=(8, 5))  # Increased figure size for better readability with diagonal labels

# Collect data for all models
all_model_data = []
model_names = []
x_categories = []
category_counts = {}

for i, model in enumerate(models):
    model_id = model["model_id"].split("/")[-1]
    model_name = model["model_name"]
    model_names.append(model_name)
    
    # Get baseline ASR
    baseline_path = f"results/refusal_dir/{model_id}/completions/jailbreakbench_baseline_evaluations.json"
    if os.path.exists(baseline_path):
        baseline_data = json.load(open(baseline_path))
        baseline_score = baseline_data["StrongREJECT_score"]
    else:
        baseline_score = 0
        
    # Get DIM ASR
    dim_path = f"results/refusal_dir/{model_id}/completions/jailbreakbench_ablation_evaluations.json"
    if os.path.exists(dim_path):
        dim_data = json.load(open(dim_path))
        dim_score = dim_data["StrongREJECT_score"]
    else:
        dim_score = 0
    
    # Get RepInd ASRs (for all RepInd directions)
    api = wandb.Api()
    matching_runs = api.runs("refusal-representations/robust_refusal_vector", 
                            filters={"group": model["group"]})
    
    repind_scores = []
    for run_name in model["run_names"]:
        for run in matching_runs:
            if run_name == run.name:
                file_path = "completions/jailbreakbench_ablation_evaluations.json"
                download_dir = os.path.join("wandb_downloads", run.group, run.name)
                os.makedirs(download_dir, exist_ok=True)
                run.file(file_path).download(root=download_dir, exist_ok=True)
                file_path = os.path.join(download_dir, file_path)
                eval_result = json.load(open(file_path))
                score = eval_result["StrongREJECT_score"]
                repind_scores.append(score)
    
    # Sort RepInd scores in descending order
    repind_scores.sort(reverse=True)
    
    # Create data points
    model_data = {
        "DIM": dim_score,
        "No intervention": baseline_score
    }
    
    # Add RepInd scores
    for j, score in enumerate(repind_scores):
        category = f"RepInd {j+1}"
        model_data[category] = score
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1
    
    all_model_data.append(model_data)

# Determine which categories to include (those that exist for all models)
common_categories = ["DIM"]
for category, count in category_counts.items():
    if count == len(models):
        common_categories.append(category)
common_categories.append("No intervention")

# Set up bar positions
bar_width = 0.2
x = np.arange(len(common_categories))

# Plot bars for each model
for i, (model_data, model_name) in enumerate(zip(all_model_data, model_names)):
    values = [model_data.get(category, 0) for category in common_categories]
    position = x + (i - len(models)/2 + 0.5) * bar_width
    ax.bar(position, values, width=bar_width, color=colors[i], label=model_name, edgecolor='black', linewidth=0.5)

# Add labels and styling
ax.set_xlabel('Directional Ablation Vector')
ax.set_ylabel('Attack Success Rate')
ax.set_xticks(x)

# Set x-tick labels diagonally
plt.setp(ax.set_xticklabels(common_categories), rotation=45, ha='right', rotation_mode='anchor')

ax.legend(loc='upper right')
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# Clean up spines
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()  # Adjust layout to make room for diagonal labels
os.makedirs("results/plots/model_comparison", exist_ok=True)
plt.savefig("results/plots/model_comparison/asr_comparison_bars.png", dpi=300, bbox_inches='tight')
plt.show()


# %%
models = [
    dict(model_name="Gemma 2 2B", 
         model_id="google/gemma-2-2b-it"),
    dict(model_name="Gemma 2 9B",
         model_id="google/gemma-2-9b-it"),
    dict(model_name="Llama 3 8B",
         model_id="meta-llama/Meta-Llama-3-8B-Instruct"),
    dict(model_name="Qwen 2.5 1.5B",
         model_id="Qwen/Qwen2.5-1.5B-Instruct"),
    dict(model_name="Qwen 2.5 3B",
         model_id="Qwen/Qwen2.5-3B-Instruct"),
    dict(model_name="Qwen 2.5 7B",
         model_id="Qwen/Qwen2.5-7B-Instruct"),
    dict(model_name="Qwen 2.5 14B",
         model_id="Qwen/Qwen2.5-14B-Instruct"),
]

# load baseline scores
dim_dir = os.path.join("/ceph/hdd/students/elsj/paper_results/dim_directions")
for model in models:
    print(model)
    model_id = model["model_id"].split("/")[-1]
    datasets = ["jailbreakbench", "strongreject", "sorrybench", "xstest"]
    for dataset in datasets:
        key = 'StrongREJECT_score' if dataset != "xstest" else "xstest_judgements"
        path = os.path.join(dim_dir, f"{model_id}/completions/{dataset}_ablation_evaluations.json")
        with open(path, "r") as f:
            data = json.load(f)
        model[f"{dataset}_ablation_asr"] = data[key]
        path = os.path.join(dim_dir, f"{model_id}/completions/{dataset}_baseline_evaluations.json")
        with open(path, "r") as f:
            data = json.load(f)
        model[f"{dataset}_baseline_asr"] = data[key]
        path = os.path.join(dim_dir, f"{model_id}/completions/{dataset}_actadd_evaluations.json")
        with open(path, "r") as f:
            data = json.load(f)
        model[f"{dataset}_actadd_asr"] = data[key]
    path = os.path.join(dim_dir, f"{model_id}/completions/harmless_actadd_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["harmless_actadd_asr"] = data["substring_matching_success_rate"]
    path = os.path.join(dim_dir, f"{model_id}/completions/harmless_baseline_evaluations.json")
    with open(path, "r") as f:
        data = json.load(f)
    model["harmless_baseline_asr"] = data["substring_matching_success_rate"]
    path = os.path.join(dim_dir, f"{model_id}/direction.pt")
    refusal_direction = torch.load(path, map_location=torch.device('cpu'))
    model["refusal_direction"] = refusal_direction.clone()

    api = wandb.Api()
    groups = [
        dict(group="sb_data_with_retain", name="retain"),
    ]
    for group in groups:
        group_name = f"{group['group']}_{model_id}"
        run_name = "run_4" if (model_id == "Qwen2.5-3B-Instruct" and "retain" in group_name) else "run_5"
        runs = api.runs("refusal-representations/robust_refusal_vector", {"group": group_name, "display_name": run_name})
        if runs:
            run = runs[0]
            summary = run.summary._json_dict
            key = "retain_summary"
            model[f"{key}"] = summary
# %%
def plot_scores(y_label: str, dataset: str):
    # Extract scores from models
    dataset_key = dataset.lower().replace("-", "")
    ablation_scores = [model[f"{dataset_key}_ablation_asr"] for model in models]
    actadd_scores = [model[f"{dataset_key}_actadd_asr"] for model in models]
    baseline_scores = [model[f"{dataset_key}_baseline_asr"] for model in models]
    if 'xstest' in dataset_key: 
        retain_ablation_scores = [model[f"retain_summary"][f"{dataset_key}_ablation_xstest_judgements"] for model in models]
        retain_actadd_scores = [model[f"retain_summary"][f"{dataset_key}_actadd_xstest_judgements"] for model in models]
    else:
        retain_ablation_scores = [model[f"retain_summary"][f"{dataset_key}_ablation_StrongREJECT_score"] for model in models]
        retain_actadd_scores = [model[f"retain_summary"][f"{dataset_key}_actadd_StrongREJECT_score"] for model in models]
    model_names = [model["model_name"] for model in models]

    x = np.arange(len(models))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.12  # Reduced width to accommodate more bars

    # Create proxy artists for the legend
    from matplotlib.patches import Patch, Rectangle
    
    # Method labels (solid colors)
    method_patches = [
        Rectangle((0,0), 1, 1, facecolor='gray', label='Baseline\n(No Intervention)', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[0], label='DIM', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=colors[1], label='RDO (Ours)', edgecolor='black', linewidth=0.5)
    ]
    
    # Condition labels (patterns)
    condition_patches = [
        Rectangle((0,0), 1, 1, facecolor='white', label='Directional\nAblation', edgecolor='black', linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor='white', label='Activation\nSubtraction', edgecolor='black', linewidth=0.5, hatch='////')
    ]

    # Plot baseline scores
    ax.bar(x - 3*bar_width/2, baseline_scores, bar_width,
        color='gray',
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Plot ablation scores side by side
    ax.bar(x - bar_width/2, ablation_scores, bar_width,
        color=colors[0],
        alpha=1.0, edgecolor='black', linewidth=0.5)
    
    ax.bar(x + bar_width/2, retain_ablation_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5)

    # Plot actadd scores side by side with hatching
    ax.bar(x + 3*bar_width/2, actadd_scores, bar_width,
        color=colors[0],
        alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

    ax.bar(x + 5*bar_width/2, retain_actadd_scores, bar_width,
        color=colors[1],
        alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

    # Customization
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='center')  # Rotated labels
    ax.tick_params(axis='both', which='major')

    # Grid styling
    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray', linewidth=0.5)

    # Adjust spines to be thinner
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Make tick marks thinner and shorter
    ax.tick_params(axis='both', width=0.5, length=3)

    # Y-axis limits and spines
    ax.set_ylim(0, 1)  # Slightly higher to accommodate error bars
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Create two-part legend with better positioning
    legend1 = ax.legend(handles=method_patches, title='Method', 
                    bbox_to_anchor=(1.0, 1.0),  # Position to the right
                    loc='upper left', ncol=1,
                    frameon=True, fancybox=False, 
                    edgecolor='black',
                    borderaxespad=0)
    ax.add_artist(legend1)  # Add first legend
    
    ax.legend(handles=condition_patches, title='Operation',
              bbox_to_anchor=(1.0, 0.5),  # Position to the right, below the first legend
              loc='upper left', ncol=1,
              frameon=True, fancybox=False,
              edgecolor='black', 
              borderaxespad=0)
    ax.set_title(f"{dataset}")
    # Save
    # plt.tight_layout()
    os.makedirs("results/plots/rdo/combined", exist_ok=True)
    path = f"results/plots/rdo/combined/{dataset}_scores.png"
    plt.savefig(path, dpi=300)
    plt.show()

plot_scores(y_label="Attack Success Rate", dataset="JailbreakBench")
plot_scores(y_label="Attack Success Rate", dataset="StrongREJECT")
plot_scores(y_label="Attack Success Rate", dataset="SORRY-Bench")