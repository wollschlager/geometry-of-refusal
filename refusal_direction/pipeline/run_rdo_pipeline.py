import torch
import random
import json
import os
import argparse
import wandb

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.select_direction import select_rdo_direction, select_cone_basis, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--wandb_run', type=str, required=True, help='wandb run')
    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_train = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)[:len(harmful_train)]
    harmful_val = load_dataset_split(harmtype='harmful', split='val', instructions_only=True)
    harmless_val = load_dataset_split(harmtype='harmless', split='val', instructions_only=True)[:len(harmful_val)]
    return harmful_train, harmless_train, harmful_val, harmless_val

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        print("Filtering train dataset")
        print(f"Number of harmful examples: {len(harmful_train)}")
        print(f"Number of harmless examples: {len(harmless_train)}")
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)

        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        print(len([score for score in harmful_train_scores.tolist() if score > 0]))
        print(len([score for score in harmless_train_scores.tolist() if score < 0]))
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)[:len(harmful_train)]
        print(f"Filtered {len(harmful_train)} harmful examples and {len(harmless_train)} harmless examples")

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
    
    return harmful_train, harmless_train, harmful_val, harmless_val

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, add_layer):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(wandb.run.dir, 'select_direction')):
        os.makedirs(os.path.join(wandb.run.dir, 'select_direction'))

    is_subspace = len(candidate_directions.shape) > 2
    if is_subspace:
        candidate_idx, direction = select_cone_basis(
            model_base,
            harmful_val,
            harmless_val,
            candidate_directions,
            artifact_dir=os.path.join(wandb.run.dir, "select_direction"),
            add_layer=add_layer,
            n_samples=cfg.subspace_n_samples
        )
    else:
        kl_threshold = 0.1 if wandb.config.get("use_retain_loss", False) and not "kl_ablation" in wandb.run.group else 9999
        if candidate_directions.shape[0] > 1:
            candidate_idx, direction = select_rdo_direction(
                model_base,
                harmful_val,
                harmless_val,
                candidate_directions,
                artifact_dir=os.path.join(wandb.run.dir, "select_direction"),
                add_layer=add_layer,
                kl_threshold=kl_threshold
            )
        else:
            candidate_idx = 0
            direction = candidate_directions[0]

    with open(f'{wandb.run.dir}/direction_metadata.json', "w") as f:
        json.dump({"candidate_idx": candidate_idx}, f, indent=4)

    torch.save(direction, f'{wandb.run.dir}/direction.pt')
    wandb.save(f'direction.pt')
    wandb.save(f'direction_metadata.json')
    return candidate_idx, direction

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(wandb.run.dir, 'completions')):
        os.makedirs(os.path.join(wandb.run.dir, 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens, batch_size=cfg.completions_batch_size)
    
    with open(f'{wandb.run.dir}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)
    wandb.save(f'{wandb.run.dir}/completions/{dataset_name}_{intervention_label}_completions.json')

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(wandb.run.dir, f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(wandb.run.dir, "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{wandb.run.dir}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)
    wandb.save(f'{wandb.run.dir}/completions/{dataset_name}_{intervention_label}_evaluations.json')

def run_pipeline(wandb_run):
    """Run the full pipeline."""
    import dotenv
    dotenv.load_dotenv("..")

    # Resume the existing wandb run
    entity = os.getenv("WANDB_ENTITY")
    project_name = os.getenv("WANDB_PROJECT")
    run_name = wandb_run # The desired run name passed as argument
    api = wandb.Api()

    # Construct the full path for querying runs
    path = f"{entity}/{project_name}"
    runs = api.runs(path, filters={"display_name": run_name})

    if not runs:
        raise ValueError(f"No run found with name '{run_name}' in project '{path}'.")

    # If multiple runs match the name, select the most recent one.
    if len(runs) > 1:
        print(f"Warning: Found {len(runs)} runs with name '{run_name}'. Selecting the most recent one.")
        # Sort by created_at timestamp (descending) and take the first one
        target_run = sorted(runs, key=lambda x: x.created_at, reverse=True)[0]
    else:
        # Exactly one run found
        target_run = runs[0]

    print(f"Found run '{target_run.name}' (ID: {target_run.id}, created at {target_run.created_at}). Resuming this run.")
    # Assign to newest_run as the subsequent code expects this variable name
    newest_run = target_run
    run = wandb.init(project=project_name, id=newest_run.id, resume="allow")

    # Get model_path from wandb config
    model_path = wandb.config.get("model")
    if not model_path:
        raise ValueError("model_path not found in wandb config for the resumed run.")
    
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)

    is_subspace = wandb.config["cone_dim"] > 1
    artifact_name = f"trained_vectors_run_{wandb.run.id}:v0"
    filename = "lowest_loss_vector.pt"
    artifact = run.use_artifact(f"{entity}/{project_name}/{artifact_name}")
    artifact_dir = artifact.download()
    print(f"Artifact dir: {artifact_dir}")
    directions = torch.load(os.path.join(artifact_dir, filename))
    directions = directions.squeeze().unsqueeze(0).cuda()
    print(directions.shape)

    add_layer = wandb.config["add_layer"]
    alpha = wandb.config["alpha"]
    
    candidate_directions = alpha * directions
    print(candidate_directions.shape)

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)

    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)

    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    candidate_idx, direction = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, add_layer=add_layer)
    if not is_subspace:
        eval_vectors = [{'vector': direction, 'name_postfix': ''}]
    else:
        subspace_dim = len(direction)
        print(f"Subspace dim: {subspace_dim}")
        eval_vectors = []
        for i in range(subspace_dim):
            eval_vectors.append({'vector': direction[i], 'name_postfix': f'_basis_{i+1}'})

    for e in eval_vectors:
        direction = e['vector']
        name_postfix = e['name_postfix']
        ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
        actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[add_layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []

        # 3a. Generate and save completions on harmful evaluation datasets
        for dataset_name in cfg.evaluation_datasets:
            if cfg.evaluate_ablation:
                generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, f'ablation{name_postfix}', dataset_name)
            if cfg.evaluate_actadd:
                generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, f'actadd{name_postfix}', dataset_name)

        # 4a. Generate and save completions on harmless evaluation dataset
        if cfg.evaluate_harmless:
            print("Evaluating on harmless instructions")

            actadd_refusal_pre_hooks, actadd_refusal_hooks = [(model_base.model_block_modules[add_layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))], []
            generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, f'actadd{name_postfix}', 'harmless', dataset=harmless_test)

    for e in eval_vectors:
        print("Evaluating completions")
        name_postfix = e['name_postfix']
        model_base.del_model()
        torch.cuda.empty_cache()

        for dataset_name in cfg.evaluation_datasets:
            if cfg.evaluate_ablation:
                evaluate_completions_and_save_results_for_dataset(cfg, f'ablation{name_postfix}', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
            if cfg.evaluate_actadd:
                evaluate_completions_and_save_results_for_dataset(cfg, f'actadd{name_postfix}', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

        if cfg.evaluate_harmless:
            evaluate_completions_and_save_results_for_dataset(cfg, f'actadd{name_postfix}', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(wandb_run=args.wandb_run)

