# %%
import torch
from nnsight import LanguageModel
import json
import numpy as np
import os
from generate_utils import projection_einops

torch.set_grad_enabled(False)

# %%
MODEL_PATH = 'google/gemma-2-2b-it'

model = LanguageModel(MODEL_PATH, cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"), device_map='auto', torch_dtype=torch.bfloat16)

# %%
with model.trace("Hello") as _:
    pass
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

if "gemma" in MODEL_PATH.lower():
    TEMPLATE = GEMMA_CHAT_TEMPLATE
elif "qwen2.5" in MODEL_PATH.lower():
    TEMPLATE = QWEN25_CHAT_TEMPLATE
elif "llama-3" in MODEL_PATH.lower():
    TEMPLATE = LLAMA3_CHAT_TEMPLATE
else:
    raise ValueError(f"Model {MODEL_PATH} not supported")

SAVE_DIR = f"results/directopt/{MODEL_PATH.split('/')[-1]}/"
os.makedirs(SAVE_DIR, exist_ok=True)

harmful_val = json.load(open('data/processed/sorrybench.json'))
harmless_val = json.load(open('data/saladbench_splits/harmless_val.json'))[:len(harmful_val)]
print(len(harmful_val), len(harmless_val))

# %%
module = model.model

# %%
raw_harmful_instructions = [d["instruction"] for d in harmful_val]
harmful_instructions = [TEMPLATE.format(instruction=d["instruction"]) for d in harmful_val]
raw_harmless_instructions = [d["instruction"] for d in harmless_val]
harmless_instructions = [TEMPLATE.format(instruction=d["instruction"]) for d in harmless_val]
print(len(raw_harmful_instructions), len(raw_harmless_instructions))

# %%
from scoring import get_bypass_scores
if "gemma" in MODEL_PATH.lower():
    refusal_tokens = [235285]
elif "qwen2.5" in MODEL_PATH.lower():
    refusal_tokens = [40, 2121]
elif "llama-3" in MODEL_PATH.lower():
    refusal_tokens = [40]
else:
    raise ValueError(f"Model {MODEL_PATH} not supported")

filter_data = True
batch_size = 16
if filter_data:
    print("Filtering data")
    harmful_scores = get_bypass_scores(model, harmful_instructions, refusal_tokens, batch_size=batch_size)
    harmless_scores = get_bypass_scores(model, harmless_instructions, refusal_tokens, batch_size=batch_size)
    filtered_harmful_instructions = [d for d, score in zip(harmful_instructions, harmful_scores) if score > 0]

    filtered_harmless_instructions = [d for d, score in zip(harmless_instructions, harmless_scores) if score < 0]
    print(f"Remaining harmful instances: {len(filtered_harmful_instructions)}")
    filtered_harmless_instructions = filtered_harmless_instructions[:len(filtered_harmful_instructions)]
    print(f"Remaining harmless instances: {len(filtered_harmless_instructions)}")

    harmful_instructions = filtered_harmful_instructions
    harmless_instructions = filtered_harmless_instructions

# %%
def new_get_cosine_similarities(model, prompts, measure_vector, intervention_vector=None, measure="after", batch_size=16):
    # Initialize list of lists to store cosine similarities per layer
    layer_cosine_sims = [[] for _ in range(len(model.model.layers))]
    
    if intervention_vector is not None:
        norm_intervention_vector = intervention_vector / intervention_vector.norm()
        
    for i in range(0, len(prompts), batch_size):
        instructions = prompts[i:i+batch_size]
        with model.trace(instructions) as _:
            for layer_idx, layer in enumerate(model.model.layers):
                act = layer.input
                if intervention_vector is not None:
                    ablated_act = act - projection_einops(act, norm_intervention_vector)
                    layer.input = ablated_act 
                    layer.self_attn.output[0][:] -= projection_einops(layer.self_attn.output[0][:], norm_intervention_vector)
                    layer.mlp.output[:] -= projection_einops(layer.mlp.output[:], norm_intervention_vector)
                else:
                    ablated_act = act
                measure_act = ablated_act if measure == 'after' else act
                measure_act = measure_act[:, -1]
                cosine_sim = torch.nn.functional.cosine_similarity(measure_act, measure_vector, dim=-1).detach().save()
                layer_cosine_sims[layer_idx].append(cosine_sim)
    
    # Concatenate all values for each layer
    all_values = []
    for layer_sims in layer_cosine_sims:
        layer_values = torch.cat([l.value for l in layer_sims])
        all_values.append(layer_values)

    return torch.stack(all_values, dim=1).to(torch.float32).cpu().numpy()

# %%
model_id = MODEL_PATH.split("/")[-1]
dim_dir = os.path.join(os.getenv("SAVE_DIR"), os.getenv("DIM_DIR"), model_id)   
refusal_directions = torch.load(f"{dim_dir}/generate_directions/mean_diffs.pt")
refusal_results = json.load(open(f"{dim_dir}/direction_metadata.json"))
best_layer = refusal_results["layer"]
best_token = refusal_results["pos"]
best_refusal_direction = torch.load(f"{dim_dir}/direction.pt").to(model.dtype)
# %%
import wandb
wandb_entity = os.getenv("WANDB_ENTITY")
wandb_project = os.getenv("WANDB_PROJECT")

vector_configs = [
    {"run_id": "0t6z11if", "run_name": "RDO$_\perp$"},
    {"run_id": "f1nl5qvp", "run_name": "Rep. Ind."},
]
api = wandb.Api()
vectors = []
for vector_config in vector_configs:
    run_id = vector_config["run_id"]
    run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
    artifact = api.artifact(f'{wandb_entity}/{wandb_project}/trained_vectors_run_{run_id}:v0', type='vector')
    artifact_dir = artifact.download()
    print(f"Artifact dir: {artifact_dir}")
    print(f"Vector config: {vector_config}")
    print("Loading lowest loss vector")
    direction = torch.load(os.path.join(artifact_dir, "lowest_loss_vector.pt"))
    direction = (direction.to(direction.dtype) / direction.norm().cpu()).to(model.dtype)
    vectors.append(direction)

# %%
vector_names = ["DIM"] + [c["run_name"] for c in vector_configs]
intervention_vectors = [best_refusal_direction] + vectors

# %%
batch_size = 16

baseline_cosine_sims = {}
baseline_cosine_sims_harmless = {}

for vector_name, vector in zip(vector_names, intervention_vectors):
    # Get similarities for harmful instructions
    cosine_sims = new_get_cosine_similarities(model, harmful_instructions, vector, batch_size=batch_size)
    baseline_cosine_sims[vector_name] = cosine_sims
    
    # # Get similarities for harmless instructions
    cosine_sims = new_get_cosine_similarities(model, harmless_instructions, vector, batch_size=batch_size)
    baseline_cosine_sims_harmless[vector_name] = cosine_sims

# %%
# Compute similarities after intervention
all_cosine_sims = {}
cosine_sim_changes = {}
cosine_sim_stds = {}

for intervention_vector, intervention_name in zip(intervention_vectors, vector_names):
    all_cosine_sims[intervention_name] = {}
    cosine_sim_changes[intervention_name] = {}
    cosine_sim_stds[intervention_name] = {}
    for measure_vector, measure_name in zip(intervention_vectors, vector_names):
        cosine_sims = new_get_cosine_similarities(model, harmful_instructions, measure_vector=measure_vector, intervention_vector=intervention_vector, batch_size=batch_size)
        all_cosine_sims[intervention_name][measure_name] = cosine_sims.mean(axis=0)
        
        # Calculate change from baseline and std
        cosine_sim_changes[intervention_name][measure_name] = cosine_sims.mean(axis=0) - baseline_cosine_sims[measure_name].mean(axis=0)
        cosine_sim_stds[intervention_name][measure_name] = cosine_sims.std(axis=0)

# %%
# Create figure with more spacing between subplots
# Get min/max across all relevant data for consistent y-axis
dim_baseline = baseline_cosine_sims[vector_names[0]].mean(axis=0)
orthogonal_baseline = baseline_cosine_sims[vector_names[1]].mean(axis=0)
repind_baseline = baseline_cosine_sims[vector_names[2]].mean(axis=0)
orthogonal_dim_ablated = all_cosine_sims[vector_names[0]][vector_names[1]]
dim_orthogonal_ablated = all_cosine_sims[vector_names[1]][vector_names[0]]
repind_dim_ablated = all_cosine_sims[vector_names[0]][vector_names[2]]
dim_repind_ablated = all_cosine_sims[vector_names[2]][vector_names[0]]

# max between dim_baseline and dim_repind_ablated
diffs = dim_baseline - dim_repind_ablated
print(max(diffs), np.argmax(diffs))

y_min = min(min(dim_baseline), min(orthogonal_baseline), min(repind_baseline),
            min(orthogonal_dim_ablated), min(dim_orthogonal_ablated),
            min(repind_dim_ablated), min(dim_repind_ablated))
y_max = max(max(dim_baseline), max(orthogonal_baseline), max(repind_baseline),
            max(orthogonal_dim_ablated), max(dim_orthogonal_ablated),
            max(repind_dim_ablated), max(dim_repind_ablated))
y_padding = (y_max - y_min) * 0.04
y_min -= y_padding
y_max += y_padding

x = range(model.config.num_hidden_layers)

# %%
from plot_style import apply_style
from matplotlib import pyplot as plt
colors = apply_style()

# Styling parameters
MARKER_SIZE = 0  # Increased marker size
LINE_WIDTH = 2

# Create second version without first subplot
fig3, ((ax2, ax3), (ax4, ax5)) = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=True)

# Ax2: RDO$_\perp$ baseline with DIM ablated
ax2.set_title('(a)', pad=10)
ax2.plot(x, orthogonal_baseline, color=colors[2], marker='o',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='RDO$_\perp$')
ax2.fill_between(x, orthogonal_baseline - baseline_cosine_sims[vector_names[1]].std(axis=0),
                 orthogonal_baseline + baseline_cosine_sims[vector_names[1]].std(axis=0),
                 color=colors[2], alpha=0.15)
ax2.plot(x, orthogonal_dim_ablated, color=colors[2], marker='s',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle='--',
         label='RDO$_\perp$ with DIM abl.')
ax2.fill_between(x, orthogonal_dim_ablated - cosine_sim_stds[vector_names[0]][vector_names[1]],
                 orthogonal_dim_ablated + cosine_sim_stds[vector_names[0]][vector_names[1]],
                 color=colors[2], alpha=0.25)  # Lighter red for better contrast

# Ax3: DIM baseline with RDO$_\perp$ ablated
ax3.set_title('(b)', pad=10)
ax3.plot(x, dim_baseline, color=colors[0], marker='o',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='DIM')
ax3.fill_between(x, dim_baseline - baseline_cosine_sims[vector_names[0]].std(axis=0),
                 dim_baseline + baseline_cosine_sims[vector_names[0]].std(axis=0),
                 color=colors[0], alpha=0.15)
ax3.plot(x, dim_orthogonal_ablated, color=colors[0], marker='s',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle='--',
         label='DIM with\nRDO$_\perp$ abl.')
ax3.fill_between(x, dim_orthogonal_ablated - cosine_sim_stds[vector_names[1]][vector_names[0]],
                 dim_orthogonal_ablated + cosine_sim_stds[vector_names[1]][vector_names[0]],
                 color=colors[0], alpha=0.25)  # Lighter blue for better contrast

# Ax4: RepInd baseline with DIM ablated
ax4.set_title('(c)', pad=10)
ax4.plot(x, repind_baseline, color=colors[1], marker='o',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='RepInd')
ax4.fill_between(x, repind_baseline - baseline_cosine_sims[vector_names[2]].std(axis=0),
                 repind_baseline + baseline_cosine_sims[vector_names[2]].std(axis=0),
                 color=colors[1], alpha=0.15)
ax4.plot(x, repind_dim_ablated, color=colors[1], marker='s',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle='--',
         label='RepInd with DIM abl.')
ax4.fill_between(x, repind_dim_ablated - cosine_sim_stds[vector_names[0]][vector_names[2]],
                 repind_dim_ablated + cosine_sim_stds[vector_names[0]][vector_names[2]],
                 color=colors[1], alpha=0.25)  # Lighter orange for better contrast

# Ax5: DIM baseline with RepInd ablated
ax5.set_title('(d)', pad=10)
ax5.plot(x, dim_baseline, color=colors[0], marker='o',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='DIM')
ax5.fill_between(x, dim_baseline - baseline_cosine_sims[vector_names[0]].std(axis=0),
                 dim_baseline + baseline_cosine_sims[vector_names[0]].std(axis=0),
                 color=colors[0], alpha=0.15)
ax5.plot(x, dim_repind_ablated, color=colors[0], marker='s',
         markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle='--',
         label='DIM with\nRepInd abl.')
ax5.fill_between(x, dim_repind_ablated - cosine_sim_stds[vector_names[2]][vector_names[0]],
                 dim_repind_ablated + cosine_sim_stds[vector_names[2]][vector_names[0]],
                 color=colors[0], alpha=0.25)  # Lighter blue for better contrast

# Configure all axes with enhanced styling
for ax in [ax2, ax3, ax4, ax5]:
    # Enhanced grid
    ax.grid(True, linestyle='--')
    
    # Better legend with larger font
    ax.legend(frameon=True, fancybox=False, framealpha=0.95,
              shadow=False,
              loc='upper left', bbox_to_anchor=(0.01, 1), ncol=1, handlelength=1.5)
    
    # Labels with larger fonts
    if ax in [ax4, ax5]:
        ax.set_xlabel('Layer')
    if ax in [ax2, ax4]:
        ax.set_ylabel('Cosine similarity')
    
    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tick parameters with larger font
    ax.tick_params(width=0.5)
    ax.set_xticks(range(0, len(x), 5))  # x ticks every 5 layers
    ax.set_ylim(y_min, y_max + 0.1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))  # y ticks every 0.1

# Adjust layout with increased vertical spacing between rows
plt.tight_layout(rect=[0, 0, 1, 0.85], h_pad=1.0, w_pad=0.3)

save_dir = f'{os.getenv("SAVE_DIR")}/plots/cosine_sims'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(f'{save_dir}/cosine_sims_four_plots.png', dpi=300, bbox_inches='tight')
plt.show()