import json
import wandb
import torch
import functools
import math
import matplotlib.pyplot as plt
import os
import wandb

from typing import List, Optional
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from einops import rearrange

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.hook_utils import add_hooks, get_activation_addition_input_pre_hook, get_direction_ablation_input_pre_hook, get_direction_ablation_output_hook

def refusal_score(
    logits: Float[Tensor, 'batch seq d_vocab_out'],
    refusal_toks: Int[Tensor, 'batch seq'],
    epsilon: Float = 1e-8,
):
    logits = logits.to(torch.float64)

    # we only care about the last tok position
    logits = logits[:, -1, :]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_probs = probs[:, refusal_toks].sum(dim=-1)

    nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)

def get_refusal_scores(model, instructions, tokenize_instructions_fn, refusal_toks, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32):
    refusal_score_fn = functools.partial(refusal_score, refusal_toks=refusal_toks)

    refusal_scores = torch.zeros(len(instructions), device=model.device)

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
            ).logits

        refusal_scores[i:i+batch_size] = refusal_score_fn(logits=logits)

    return refusal_scores

def get_last_position_logits(model, tokenizer, instructions, tokenize_instructions_fn, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32) -> Float[Tensor, "n_instructions d_vocab"]:
    last_position_logits = None

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
            ).logits

        if last_position_logits is None:
            last_position_logits = logits[:, -1, :]
        else:
            last_position_logits = torch.cat((last_position_logits, logits[:, -1, :]), dim=0)

    return last_position_logits

def plot_refusal_scores(
    refusal_scores: Float[Tensor, 'n_pos n_layer'],
    baseline_refusal_score: Optional[float],
    token_labels: List[str],
    title: str,
    artifact_dir: str,
    artifact_name: str,
):
    n_pos, n_layer = refusal_scores.shape

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(9, 5))  # width and height in inches

    # Add a trace for each position to extract
    for i in range(-n_pos, 0):
        ax.plot(
            list(range(n_layer)),
            refusal_scores[i].cpu().numpy(),
            label=f'{i}: {repr(token_labels[i])}'
        )

    if baseline_refusal_score is not None:
        # Add a horizontal line for the baseline
        ax.axhline(y=baseline_refusal_score, color='black', linestyle='--')
        ax.annotate('Baseline', xy=(1, baseline_refusal_score), xytext=(8, 10), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points',
                    horizontalalignment='right', verticalalignment='center')

    ax.set_title(title)
    ax.set_xlabel('Layer source of direction (resid_pre)')
    ax.set_ylabel('Refusal score')
    ax.legend(title='Position source of direction', loc='lower left')

    plt.savefig(f"{artifact_dir}/{artifact_name}.png")

import seaborn as sns
def plot_density_plot(
    scores: Float[Tensor, '...'],  # can be either [candidate_idx] or [subspace_dim candidate_idx]
    title: str,
    artifact_dir: str,
    artifact_name: str,
):
    # Convert scores to numpy array
    scores_np = scores.cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    if scores_np.ndim == 1:
        # Single dimension case
        sns.kdeplot(data=scores_np, fill=True)
    else:
        # Multiple subspace case
        for i in range(scores_np.shape[0]):
            sns.kdeplot(data=scores_np[i], fill=True, alpha=0.3, label=f'Subspace {i}')
        plt.legend()
    
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Density')
    
    # Save plot
    plt.savefig(f"{artifact_dir}/{artifact_name}.png")
    wandb.save(f"{artifact_dir}/{artifact_name}.png")
    plt.close()

# returns True if the direction should be filtered out
def filter_fn(refusal_score, steering_score, kl_div_score, layer, n_layer, kl_threshold=None, induce_refusal_threshold=None, prune_layer_percentage=0.20) -> bool:
    if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    if induce_refusal_threshold is not None and steering_score < induce_refusal_threshold:
        return True
    return False

def select_direction(
    model_base: ModelBase,
    harmful_instructions,
    harmless_instructions,
    candidate_directions: Float[Tensor, 'n_pos n_layer d_model'],
    artifact_dir,
    kl_threshold=0.1, # directions larger KL score are filtered out
    induce_refusal_threshold=0.0, # directions with a lower inducing refusal score are filtered out
    prune_layer_percentage=0.2, # discard the directions extracted from the last 20% of the model
    batch_size=32
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_pos, n_layer, d_model = candidate_directions.shape

    baseline_refusal_scores_harmful = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_hooks=[], batch_size=batch_size)
    baseline_refusal_scores_harmless = get_refusal_scores(model_base.model, harmless_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_hooks=[], batch_size=batch_size)

    ablation_kl_div_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)
    ablation_refusal_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)
    steering_refusal_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)

    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing KL for source position {source_pos}"):

            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]

            intervention_logits: Float[Tensor, "n_instructions 1 d_vocab"] = get_last_position_logits(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                instructions=harmless_instructions,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                batch_size=batch_size
            )

            ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal ablation for source position {source_pos}"):

            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]

            refusal_scores = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            ablation_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal addition for source position {source_pos}"):

            refusal_vector = candidate_directions[source_pos, source_layer]
            coeff = torch.tensor(1.0)

            fwd_pre_hooks = [(model_base.model_block_modules[source_layer], get_activation_addition_input_pre_hook(vector=refusal_vector, coeff=coeff))]
            fwd_hooks = []

            refusal_scores = get_refusal_scores(model_base.model, harmless_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            steering_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

    plot_refusal_scores(
        refusal_scores=ablation_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmful.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Ablating direction on harmful instructions',
        artifact_dir=artifact_dir,
        artifact_name='ablation_scores'
    )

    plot_refusal_scores(
        refusal_scores=steering_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmless.mean().item(),
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Adding direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='actadd_scores'
    )

    plot_refusal_scores(
        refusal_scores=ablation_kl_div_scores,
        baseline_refusal_score=0.0,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='KL Divergence when ablating direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='kl_div_scores'
    )

    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):

            json_output_all_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'refusal_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'steering_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })

            refusal_score = ablation_refusal_scores[source_pos, source_layer].item()
            steering_score = steering_refusal_scores[source_pos, source_layer].item()
            kl_div_score = ablation_kl_div_scores[source_pos, source_layer].item()

            # we sort the directions in descending order (from highest to lowest score)
            # the intervention is better at bypassing refusal if the refusal score is low, so we multiply by -1
            sorting_score = -refusal_score

            # we filter out directions if the KL threshold 
            discard_direction = filter_fn(
                refusal_score=refusal_score,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                layer=source_layer,
                n_layer=n_layer,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if discard_direction:
                continue

            filtered_scores.append((sorting_score, source_pos, source_layer))

            json_output_filtered_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'refusal_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'steering_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })   

    with open(f"{artifact_dir}/direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    json_output_filtered_scores = sorted(json_output_filtered_scores, key=lambda x: x['refusal_score'], reverse=False)

    with open(f"{artifact_dir}/direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    assert len(filtered_scores) > 0, "All scores have been filtered out!"

    # sorted in descending order
    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)

    # now return the best position, layer, and direction
    score, pos, layer = filtered_scores[0]

    print(f"Selected direction: position={pos}, layer={layer}")
    print(f"Refusal score: {ablation_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
    print(f"Steering score: {steering_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
    print(f"KL Divergence: {ablation_kl_div_scores[pos, layer]:.4f}")
    
    return pos, layer, candidate_directions[pos, layer]


def select_rdo_direction(
    model_base: ModelBase,
    harmful_instructions,
    harmless_instructions,
    candidate_directions: Float[Tensor, 'n_candidates d_model'],
    artifact_dir,
    add_layer,
    kl_threshold=0.1, # directions larger KL score are filtered out
    induce_refusal_threshold=0.0, # directions with a lower inducing refusal score are filtered out
    batch_size=32,
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_candidates, d_model = candidate_directions.shape

    baseline_refusal_scores_harmful = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_hooks=[], batch_size=batch_size)
    baseline_refusal_scores_harmless = get_refusal_scores(model_base.model, harmless_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_hooks=[], batch_size=batch_size)

    ablation_kl_div_scores = torch.zeros((n_candidates), device=model_base.model.device, dtype=torch.float64)
    ablation_refusal_scores = torch.zeros((n_candidates), device=model_base.model.device, dtype=torch.float64)
    steering_refusal_scores = torch.zeros((n_candidates), device=model_base.model.device, dtype=torch.float64)

    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )
    for candidate_idx in tqdm(range(n_candidates), desc=f"Computing KL for candidate directions"):
        ablation_dir = candidate_directions[candidate_idx]
        fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
        fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
        fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]

        intervention_logits: Float[Tensor, "n_instructions 1 d_vocab"] = get_last_position_logits(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            instructions=harmless_instructions,
            tokenize_instructions_fn=model_base.tokenize_instructions_fn,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            batch_size=batch_size
        )

        ablation_kl_div_scores[candidate_idx] = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()

    for candidate_idx in tqdm(range(n_candidates), desc=f"Computing refusal ablation for candidate directions"):

        ablation_dir = candidate_directions[candidate_idx]
        fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
        fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
        fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]

        refusal_scores = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
        ablation_refusal_scores[candidate_idx] = refusal_scores.mean().item()

    for candidate_idx in tqdm(range(n_candidates), desc=f"Computing refusal addition for candidate directions"):
        refusal_vector = candidate_directions[candidate_idx]
        coeff = torch.tensor(1.0)

        fwd_pre_hooks = [(model_base.model_block_modules[add_layer], get_activation_addition_input_pre_hook(vector=refusal_vector, coeff=coeff))]
        fwd_hooks = []

        refusal_scores = get_refusal_scores(model_base.model, harmless_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
        steering_refusal_scores[candidate_idx] = refusal_scores.mean().item()

    plot_density_plot(
        scores=ablation_kl_div_scores,
        title='KL Divergence for candidate directions',
        artifact_dir=artifact_dir,
        artifact_name='kl_div_scores'
    )
    plot_density_plot(
        scores=ablation_refusal_scores,
        title='Refusal score for candidate directions',
        artifact_dir=artifact_dir,
        artifact_name='refusal_scores'
    )
    plot_density_plot(
        scores=steering_refusal_scores,
        title='Steering score for candidate directions',
        artifact_dir=artifact_dir,
        artifact_name='steering_scores'
    )   

    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for candidate_idx in range(n_candidates):

        json_output_all_scores.append({
            'candidate_idx': candidate_idx,
            'layer': add_layer,
            'refusal_score': ablation_refusal_scores[candidate_idx].item(),
            'steering_score': steering_refusal_scores[candidate_idx].item(),
            'kl_div_score': ablation_kl_div_scores[candidate_idx].item()
        })

        refusal_score = ablation_refusal_scores[candidate_idx].item()
        steering_score = steering_refusal_scores[candidate_idx].item()
        kl_div_score = ablation_kl_div_scores[candidate_idx].item()

        # we sort the directions in descending order (from highest to lowest score)
        # the intervention is better at bypassing refusal if the refusal score is low, so we multiply by -1
        sorting_score = -refusal_score

        # we filter out directions if the KL threshold 
        discard_direction = filter_fn(
            refusal_score=refusal_score,
            steering_score=steering_score,
            kl_div_score=kl_div_score,
            layer=add_layer,
            n_layer=None,
            kl_threshold=kl_threshold,
            induce_refusal_threshold=induce_refusal_threshold,
            prune_layer_percentage=None,
        )

        if discard_direction:
            continue

        filtered_scores.append((sorting_score, candidate_idx))

        json_output_filtered_scores.append({
            'candidate_idx': candidate_idx,
            'layer': add_layer,
            'refusal_score': ablation_refusal_scores[candidate_idx].item(),
            'steering_score': steering_refusal_scores[candidate_idx].item(),
            'kl_div_score': ablation_kl_div_scores[candidate_idx].item()
        })   

    with open(f"{artifact_dir}/direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)
    wandb.save(f"{artifact_dir}/direction_evaluations.json")

    json_output_filtered_scores = sorted(json_output_filtered_scores, key=lambda x: x['refusal_score'], reverse=False)

    with open(f"{artifact_dir}/direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)
    wandb.save(f"{artifact_dir}/direction_evaluations_filtered.json")

    assert len(filtered_scores) > 0, "All scores have been filtered out!"

    # sorted in descending order
    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)

    # now return the best position, layer, and direction
    score, candidate_idx = filtered_scores[0]

    print(f"Selected direction: candidate_idx={candidate_idx}")
    print(f"Refusal score: {ablation_refusal_scores[candidate_idx]:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
    print(f"Steering score: {steering_refusal_scores[candidate_idx]:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
    print(f"KL Divergence: {ablation_kl_div_scores[candidate_idx]:.4f}")

    if wandb.run is not None:
        wandb.summary["val/candidate_idx"] = candidate_idx
        wandb.summary["val/refusal_score"] = ablation_refusal_scores[candidate_idx].item()
        wandb.summary["val/steering_score"] = steering_refusal_scores[candidate_idx].item()
        wandb.summary["val/kl_div_score"] = ablation_kl_div_scores[candidate_idx].item()

    return candidate_idx, candidate_directions[candidate_idx]

def sample_prob_vectors(batch_size, dim):
    rng_state = torch.get_rng_state()
    torch.manual_seed(42)
    samples = torch.exp(torch.randn(batch_size, dim))
    samples = samples / samples.sum(dim=1, keepdim=True)
    torch.set_rng_state(rng_state)
    return samples

def sample_hypersphere_vectors(batch_size, dim):
    print("Sampling hypersphere vectors")
    rng_state = torch.get_rng_state()
    torch.manual_seed(42)
    samples = torch.randn(batch_size, dim).abs()
    samples = samples / torch.norm(samples, dim=1, keepdim=True)
    torch.set_rng_state(rng_state)
    return samples

def select_cone_basis(
    model_base: ModelBase,
    harmful_instructions,
    harmless_instructions,
    candidate_directions: Float[Tensor, 'n_candidates subspace_dim d_model'],
    artifact_dir,
    add_layer,
    kl_threshold=0.1, # directions larger KL score are filtered out
    induce_refusal_threshold=0.0, # directions with a lower inducing refusal score are filtered out
    batch_size=32,
    n_samples=8,
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_candidates, subspace_dim, d_model = candidate_directions.shape

    baseline_refusal_scores_harmful = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_hooks=[], batch_size=batch_size)
    baseline_refusal_scores_harmless = get_refusal_scores(model_base.model, harmless_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_hooks=[], batch_size=batch_size)

    # ablation_kl_div_scores = torch.zeros((n_candidates, subspace_dim), device=model_base.model.device, dtype=torch.float64)
    # ablation_refusal_scores = torch.zeros((n_candidates, subspace_dim), device=model_base.model.device, dtype=torch.float64)
    # steering_refusal_scores = torch.zeros((n_candidates, subspace_dim), device=model_base.model.device, dtype=torch.float64)

    # # candidate_directions = torch.stack(torch.load("../subspace.pt"))[-20:].cuda()

    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )

    if wandb.config["sampling_method"] == "interpolation":
        samples = sample_prob_vectors(n_samples, subspace_dim).to(model_base.model.device)
    else:
        samples = sample_hypersphere_vectors(n_samples, subspace_dim).to(model_base.model.device)

    ablation_kl_div_scores = torch.zeros((n_candidates, subspace_dim), device=model_base.model.device, dtype=torch.float64)
    ablation_refusal_scores = torch.zeros((n_candidates, subspace_dim), device=model_base.model.device, dtype=torch.float64)
    steering_refusal_scores = torch.zeros((n_candidates, subspace_dim), device=model_base.model.device, dtype=torch.float64)

    sample_ablation_kl_div_scores = torch.zeros((n_candidates, n_samples), device=model_base.model.device, dtype=torch.float64)
    sample_ablation_refusal_scores = torch.zeros((n_candidates, n_samples), device=model_base.model.device, dtype=torch.float64)
    sample_steering_refusal_scores = torch.zeros((n_candidates, n_samples), device=model_base.model.device, dtype=torch.float64)

    for candidate_idx in range(n_candidates):

        # for subspace_idx in tqdm(range(subspace_dim), desc=f"Computing KL for candidate directions {candidate_idx}"):
        #     ablation_dir = candidate_directions[candidate_idx, subspace_idx]
        #     fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
        #     fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
        #     fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]

        #     intervention_logits: Float[Tensor, "n_instructions 1 d_vocab"] = get_last_position_logits(
        #         model=model_base.model,
        #         tokenizer=model_base.tokenizer,
        #         instructions=harmless_instructions,
        #         tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        #         fwd_pre_hooks=fwd_pre_hooks,
        #         fwd_hooks=fwd_hooks,
        #         batch_size=batch_size
        #     )
        #     ablation_kl_div_scores[candidate_idx, subspace_idx] = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()


        for subspace_idx in tqdm(range(subspace_dim), desc=f"Computing refusal ablation for candidate directions {candidate_idx}"):
            ablation_dir = candidate_directions[candidate_idx, subspace_idx]
            fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=ablation_dir)) for layer in range(model_base.model.config.num_hidden_layers)]

            refusal_scores = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            ablation_refusal_scores[candidate_idx, subspace_idx] = refusal_scores.mean().item()

        for subspace_idx in tqdm(range(subspace_dim), desc=f"Computing refusal addition for candidate directions {candidate_idx}"):
            refusal_vector = candidate_directions[candidate_idx, subspace_idx]
            coeff = torch.tensor(1.0)

            fwd_pre_hooks = [(model_base.model_block_modules[add_layer], get_activation_addition_input_pre_hook(vector=refusal_vector, coeff=coeff))]
            fwd_hooks = []

            refusal_scores = get_refusal_scores(model_base.model, harmless_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            steering_refusal_scores[candidate_idx, subspace_idx] = refusal_scores.mean().item()
        
        direction = candidate_directions[candidate_idx, :]
        norm_best_direction = direction / direction.norm(dim=-1, keepdim=True)
        transformed_samples = [torch.matmul(sample, norm_best_direction) for sample in samples]
        transformed_samples = [sample / torch.norm(sample) for sample in transformed_samples]
        transformed_samples = [sample * wandb.config["alpha"] for sample in transformed_samples]
        transformed_samples = [sample.to(model_base.model.dtype) for sample in transformed_samples]

        for sample_idx in tqdm(range(n_samples), desc=f"Computing KL divergence for sample directions for candidate {candidate_idx}"):
            sample_vector = transformed_samples[sample_idx]
            fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=sample_vector)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=sample_vector)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=sample_vector)) for layer in range(model_base.model.config.num_hidden_layers)]

            intervention_logits: Float[Tensor, "n_instructions 1 d_vocab"] = get_last_position_logits(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                instructions=harmless_instructions,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                batch_size=batch_size
            )
            sample_ablation_kl_div_scores[candidate_idx, sample_idx] = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()

        for sample_idx in tqdm(range(n_samples), desc=f"Computing refusal ablation for sample directions for candidate {candidate_idx}"):
            sample_vector = transformed_samples[sample_idx]
            fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=sample_vector)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=sample_vector)) for layer in range(model_base.model.config.num_hidden_layers)]
            fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=sample_vector)) for layer in range(model_base.model.config.num_hidden_layers)]

            refusal_scores = get_refusal_scores(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            sample_ablation_refusal_scores[candidate_idx, sample_idx] = refusal_scores.mean().item()
    
        coeff = torch.tensor(1.0)
        for sample_idx in tqdm(range(n_samples), desc=f"Computing steering refusal for sample directions for candidate {candidate_idx}"):
            sample_vector = transformed_samples[sample_idx]
            fwd_pre_hooks = [(model_base.model_block_modules[add_layer], get_activation_addition_input_pre_hook(vector=sample_vector, coeff=coeff))]
            fwd_hooks = []

            refusal_scores = get_refusal_scores(model_base.model, harmless_instructions, model_base.tokenize_instructions_fn, model_base.refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            sample_steering_refusal_scores[candidate_idx, sample_idx] = refusal_scores.mean().item()
    
    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    basis_discards = []
    sample_discards = []
    for candidate_idx in range(n_candidates):
        mean_refusal_score = sample_ablation_refusal_scores[candidate_idx, :].mean().item()
        mean_steering_score = sample_steering_refusal_scores[candidate_idx, :].mean().item()
        mean_kl_div_score = sample_ablation_kl_div_scores[candidate_idx, :].max().item()

        basis_refusal_scores = ablation_refusal_scores[candidate_idx, :]
        basis_steering_scores = steering_refusal_scores[candidate_idx, :]
        max_basis_refusal_score = basis_refusal_scores.max().item()
        min_basis_steering_score = basis_steering_scores.min().item()

        discard_basis = filter_fn(
            refusal_score=max_basis_refusal_score,
            steering_score=min_basis_steering_score,
            kl_div_score=9999,
            layer=add_layer,
            n_layer=None,
            induce_refusal_threshold=induce_refusal_threshold,
            prune_layer_percentage=None,
        )
        basis_discards.append(discard_basis)

        discard_samples = filter_fn(
            refusal_score=mean_refusal_score,
            steering_score=mean_steering_score,
            kl_div_score=mean_kl_div_score,
            layer=add_layer,
            n_layer=None,
            kl_threshold=kl_threshold,
            induce_refusal_threshold=induce_refusal_threshold,
            prune_layer_percentage=None,
        )
        sample_discards.append(discard_samples)

    all_basis_discarded = all(basis_discards)
    all_sample_discarded = all(sample_discards)
    for candidate_idx in range(n_candidates):
        mean_refusal_score = sample_ablation_refusal_scores[candidate_idx, :].mean().item()
        mean_steering_score = sample_steering_refusal_scores[candidate_idx, :].mean().item()
        mean_kl_div_score = sample_ablation_kl_div_scores[candidate_idx, :].max().item()

        basis_refusal_scores = ablation_refusal_scores[candidate_idx, :]
        basis_steering_scores = steering_refusal_scores[candidate_idx, :]
        max_basis_refusal_score = basis_refusal_scores.max().item()
        min_basis_steering_score = basis_steering_scores.min().item()

        json_output_all_scores.append({
            'candidate_idx': candidate_idx,
            'layer': add_layer,
            'mean_refusal_score': mean_refusal_score,
            'mean_steering_score': mean_steering_score,
            'mean_kl_div_score': mean_kl_div_score,
            'max_basis_refusal_score': max_basis_refusal_score,
            'min_basis_steering_score': min_basis_steering_score,
            'basis_refusal_scores': basis_refusal_scores.tolist(),
            'basis_steering_scores': basis_steering_scores.tolist()
        })

        # we sort the directions in descending order (from highest to lowest score)
        # the intervention is better at bypassing refusal if the refusal score is low, so we multiply by -1
        sorting_score = -mean_refusal_score

        # we filter out directions if the KL threshold 
        if not all_basis_discarded:
            discard_basis = filter_fn(
                refusal_score=max_basis_refusal_score,
                steering_score=min_basis_steering_score,
                kl_div_score=9999,
                layer=add_layer,
                n_layer=None,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=None,
            )
        else:
            discard_basis = False # ignore the basis if nothing works
            print("="*100)
            print("No appropriate basis found, skipping basis filtering")
            print("="*100)

        if all_sample_discarded:
            print("="*100)
            print("No appropriate samples found, skipping sample filtering")
            print("="*100)

        discard_samples = filter_fn(
            refusal_score=mean_refusal_score,
            steering_score=mean_steering_score,
            kl_div_score=mean_kl_div_score,
            layer=add_layer,
            n_layer=None,
            kl_threshold=kl_threshold,
            induce_refusal_threshold=induce_refusal_threshold if not all_sample_discarded else -999,
            prune_layer_percentage=None,
        )

        if discard_basis or discard_samples:
            continue

        filtered_scores.append((sorting_score, candidate_idx))

        json_output_filtered_scores.append({
            'candidate_idx': candidate_idx,
            'layer': add_layer,
            'mean_refusal_score': sample_ablation_refusal_scores[candidate_idx, :].mean().item(),
            'max_refusal_score': sample_ablation_refusal_scores[candidate_idx, :].max().item(),
            'min_refusal_score': sample_ablation_refusal_scores[candidate_idx, :].min().item(),
            'mean_steering_score': sample_steering_refusal_scores[candidate_idx, :].mean().item(),
            'min_steering_score': sample_steering_refusal_scores[candidate_idx, :].min().item(),
            'max_steering_score': sample_steering_refusal_scores[candidate_idx, :].max().item(),
            'mean_kl_div_score': sample_ablation_kl_div_scores[candidate_idx, :].mean().item(),
            'max_kl_div_score': sample_ablation_kl_div_scores[candidate_idx, :].max().item(),
            'min_kl_div_score': sample_ablation_kl_div_scores[candidate_idx, :].min().item(),
            'basis_refusal_scores': basis_refusal_scores.tolist(),
            'basis_steering_scores': basis_steering_scores.tolist(),
            'max_basis_refusal_score': max_basis_refusal_score,
            'min_basis_steering_score': min_basis_steering_score
        })   

    with open(f"{artifact_dir}/direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)
    wandb.save(f"{artifact_dir}/direction_evaluations.json")

    json_output_filtered_scores = sorted(json_output_filtered_scores, key=lambda x: x['mean_refusal_score'], reverse=False)

    with open(f"{artifact_dir}/direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)
    wandb.save(f"{artifact_dir}/direction_evaluations_filtered.json")

    assert len(filtered_scores) > 0, "All scores have been filtered out!"

    # sorted in descending order
    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)

    # now return the best position, layer, and direction
    score, candidate_idx = filtered_scores[0]

    print(f"Selected candidate direction: candidate_idx={candidate_idx}")
    print(f"Mean Refusal score: {sample_ablation_refusal_scores[candidate_idx].mean().item():.4f} ± {sample_ablation_refusal_scores[candidate_idx].std().item():.4f}, Min: {sample_ablation_refusal_scores[candidate_idx].min().item():.4f}, Max: {sample_ablation_refusal_scores[candidate_idx].max().item():.4f}")
    print(f"Mean Steering score: {sample_steering_refusal_scores[candidate_idx].mean().item():.4f} ± {sample_steering_refusal_scores[candidate_idx].std().item():.4f}, Min: {sample_steering_refusal_scores[candidate_idx].min().item():.4f}, Max: {sample_steering_refusal_scores[candidate_idx].max().item():.4f}")
    print(f"Mean KL Divergence score: {sample_ablation_kl_div_scores[candidate_idx].mean().item():.4f} ± {sample_ablation_kl_div_scores[candidate_idx].std().item():.4f}, Min: {sample_ablation_kl_div_scores[candidate_idx].min().item():.4f}, Max: {sample_ablation_kl_div_scores[candidate_idx].max().item():.4f}")

    print(f"Max basis refusal score: {ablation_refusal_scores[candidate_idx].max().item():.4f}")
    print(f"Min basis steering score: {steering_refusal_scores[candidate_idx].min().item():.4f}")
    print(f"Max KL divergence score: {ablation_kl_div_scores[candidate_idx].max().item():.4f}")

    direction = candidate_directions[candidate_idx, :]
    norm_best_direction = direction / direction.norm(dim=-1, keepdim=True)

    samples = sample_hypersphere_vectors(512, subspace_dim).to(model_base.model.device)
    transformed_samples = [torch.matmul(sample, norm_best_direction) for sample in samples]
    transformed_samples = [sample / torch.norm(sample) for sample in transformed_samples]
    transformed_samples = [sample * wandb.config["alpha"] for sample in transformed_samples]

    if wandb.run is not None:
        wandb.summary["candidate_idx"] = candidate_idx
        torch.save(transformed_samples, f'{wandb.run.dir}/samples.pt')
        wandb.save(f'samples.pt')
    return candidate_idx, candidate_directions[candidate_idx, :]

def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if mask is None:
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean

def kl_div_fn(
    logits_a: Float[Tensor, 'batch seq_pos d_vocab'],
    logits_b: Float[Tensor, 'batch seq_pos d_vocab'],
    mask: Int[Tensor, "batch seq_pos"]=None,
    epsilon: Float=1e-6
) -> Float[Tensor, 'batch']:
    """
    Compute the KL divergence loss between two tensors of logits.
    """
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)

    if mask is None:
        return torch.mean(kl_divs, dim=-1)
    else:
        return masked_mean(kl_divs, mask).mean(dim=-1)