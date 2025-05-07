# %%
import torch
import torch.nn as nn
import json
from torch import Tensor
from jaxtyping import Float
from tqdm import tqdm
import os
import wandb
import argparse
import sys

MODEL_PATH = 'google/gemma-2-2b-it'
CACHE_DIR = '/ceph/hdd/students/elsj/huggingface'

assert "gemma" in MODEL_PATH.lower() or "qwen2.5" in MODEL_PATH.lower() or "llama-3" in MODEL_PATH.lower(), "Model not supported"

#%%
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
sns.set_style("white")
sns.set_context("paper", font_scale=1, rc={
        "lines.linewidth": 1.2,
        "xtick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.major.size": 0,
        "ytick.minor.size": 0
    })

matplotlib.rcParams["mathtext.fontset"] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['figure.autolayout'] = True

plt.rc('font', size=13)
plt.rc('font', size=13)
plt.rc('axes', titlesize=13)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', title_fontsize=13)
plt.rc('legend', fontsize=13)
plt.rc('figure', titlesize=16)

colors = sns.color_palette('colorblind')
colors[0], colors[-1] = colors[-1], colors[0]

colors = sns.color_palette('colorblind')
colors[0], colors[-1] = colors[-1], colors[0]
# %%
from nnsight import LanguageModel
dtype = torch.bfloat16
model = LanguageModel(MODEL_PATH, cache_dir=CACHE_DIR, device_map='auto', torch_dtype=dtype)
model.requires_grad_(False)

# %%
with model.trace("Hello") as tracer:
    pass

# %%
model_id = MODEL_PATH.split("/")[-1]
refusal_directions = torch.load(f"results/refusal_dir/{model_id}/generate_directions/mean_diffs.pt")
refusal_results = json.load(open(f"results/refusal_dir/{model_id}/direction_metadata.json"))
best_layer = refusal_results["layer"]
best_token = refusal_results["pos"]
best_refusal_direction = torch.load(f"results/refusal_dir/{model_id}/direction.pt").to(model.dtype)

# %%
SAVE_DIR = f"results/directopt/{MODEL_PATH.split('/')[-1]}/"
os.makedirs(SAVE_DIR, exist_ok=True)

add_layer = best_layer
alpha = best_refusal_direction.norm().detach().clone()
print(f"add_layer: {add_layer}, alpha: {alpha}")

# %%
GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""
QWEN25_CHAT_TEMPLATE = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def apply_chat_template(tokenizer, instructions: list[str]):
    if 'llama-3' in MODEL_PATH.lower():
        prompts = [LLAMA3_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    elif "gemma" in MODEL_PATH.lower():
        prompts = [GEMMA_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
    elif "qwen2.5" in MODEL_PATH.lower():
        prompts = [QWEN25_CHAT_TEMPLATE.format(instruction=inst) for inst in instructions]
        # prompts = [tokenizer.apply_chat_template([{"role": "user", "content": inst}], add_generation_prompt=True, tokenize=False) for inst in instructions]
    else:
        raise ValueError(f"Model {MODEL_PATH} not supported")
    return prompts

# %%
# get first token completion
def generate_completions(model, dataset, max_new_tokens=1):
    instructions = apply_chat_template(model.tokenizer, [d['instruction'] for d in dataset])
    print(instructions[0])
    decoded = []
    tokens = []
    for instruction in instructions:
        input_tokens = model.tokenizer(instruction, add_special_tokens=True, padding=True, truncation=False)["input_ids"]
        start_token = len(input_tokens)
        with model.generate(instruction, max_new_tokens=max_new_tokens, do_sample=False) as generator:
            out = model.generator.output.save()
        decoded.append(model.tokenizer.decode(out[0][start_token:], skip_special_tokens=True))
        tokens.append(out[0][start_token:])
    return decoded, tokens

# %%
if "gemma" in MODEL_PATH.lower():
    refusal_tokens = [235285]
elif "qwen2.5" in MODEL_PATH.lower():
    refusal_tokens = [40, 2121]
elif "llama-3" in MODEL_PATH.lower():
    refusal_tokens = [40]
else:
    raise ValueError(f"Model {MODEL_PATH} not supported")

# %%
val_data = json.load(open("data/saladbench_splits/harmful_val.json"))
harmful_val_instructions = apply_chat_template(model.tokenizer, [d['instruction'] for d in val_data])

# %%
import os
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from scoring import get_induce_scores

# List of dimensions to evaluate (e.g. dims 2 through 6)
dims_to_eval = [1, 2, 3, 4, 5]
file_path = "samples.pt"
batch_size = 32
default_alpha = alpha.item()
n_alphas = 10
alphas = list(np.linspace(-default_alpha, default_alpha, n_alphas))
alphas.insert(len(alphas)//2, 0)  # Insert 0 in the middle
n_samples = 64
n_instructions = 64
# %%

results_by_dim = {}
base_parent_dir = '/ceph/hdd/students/elsj/paper_results/subspace_samples_eval/join_subspace_gemma-2-2b-it'

for dim in dims_to_eval:
    # Build the directory path for the given dimension
    if dim == 1:
        samples = [best_refusal_direction]
    else:
        base_dir = os.path.join(base_parent_dir, f"dim_{dim}")
        samples = torch.load(os.path.join(base_dir, file_path))
    
    mean_scores = []
    std_scores = []
    # Record per-alpha scores for individual sample plotting
    sample_scores_matrix = []  # Will be a list of lists (one per alpha value)
    
    for a in tqdm(alphas, desc=f"Alpha values (Dim {dim})"):
        alpha_scores = []
        for sample in tqdm(samples[:n_samples], desc=f"Samples (α={a})", leave=False):
            sample = sample / sample.norm()
            scores = get_induce_scores(model, harmful_val_instructions[:n_instructions], refusal_tokens, add_layer, a * sample, batch_size=batch_size)
            alpha_scores.append(scores.mean().item())
        sample_scores_matrix.append(alpha_scores)
        mean_scores.append(np.mean(alpha_scores))
        std_scores.append(np.std(alpha_scores))
    
    results_by_dim[dim] = {
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "sample_scores_matrix": np.array(sample_scores_matrix)  # shape: (len(alphas), n_samples)
    }

# %%
os.makedirs("results/properties", exist_ok=True)
torch.save(results_by_dim, "results/properties/properties.pt")
# %%
results_by_dim = torch.load("results/properties/properties.pt")
# %%
# Define professional colors and plot style sizes
# Use a smaller figure size as in the example
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Define markers for each dimension
markers = ['o', 'x', 's', 'D', '^']  # Ensure you have at least as many markers as dimensions

# Loop over dimensions with index for color & marker assignment
for idx, (dim, res) in enumerate(sorted(results_by_dim.items())):
    marker = markers[idx % len(markers)]
    if dim == 1:
        # Plot dimension 1 without error bars
        ax.plot(
            alphas,
            res['mean_scores'],
            marker=marker,
            color=colors[idx],
            label="DIM direction"  # as in your original code
        )
    else:
        # Plot other dimensions with adjusted error bars
        ax.errorbar(
            alphas, 
            res['mean_scores'], 
            yerr=res['std_scores'],
            fmt=marker + '-',         # marker with a line
            capsize=3,                # reduced cap size for error bars
            elinewidth=0.8,           # thinner error bar lines
            alpha=0.8,                # apply slight transparency to error bars
            color=colors[idx],
            label=f"Dimension {dim} samples"
        )

# Customize axes labels with professional font sizes
ax.set_xlabel('Alpha $\\alpha$')
ax.set_ylabel('bypass_score')

# Customize grid: use a dashed grid with slight transparency
ax.grid(True, linestyle='--', alpha=0.6)

# Add reference horizontal and vertical lines
ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)

# Customize legend appearance similarly to the temperature sampling example
ax.legend(
    frameon=True,
    fancybox=False,
    edgecolor='black',
    loc='best',
)

# Set the tick parameters for a cleaner look
ax.tick_params(axis='both', which='major')

# Clean up spines for a professional look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Adjust layout and save the figure with consistent parameters
fig.tight_layout()
# plt.show()
os.makedirs("results/plots", exist_ok=True)
plt.savefig("results/plots/properties_new.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()