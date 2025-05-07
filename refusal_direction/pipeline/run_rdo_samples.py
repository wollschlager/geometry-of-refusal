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

from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--wandb_run', type=str, required=True, help='wandb run')
    parser.add_argument('--sample_start_idx', type=int, required=True, help='')
    parser.add_argument('--sample_end_idx', type=int, required=True, help='')
    parser.add_argument('--save_dir', type=str, default='sample_results', help='Base directory to save results.')

    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    if cfg.sample:
        harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
        harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
        harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
        harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    else:
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

def generate_and_save_completions_for_dataset(cfg, download_dir, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(download_dir, 'completions')):
        os.makedirs(os.path.join(download_dir, 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens, batch_size=cfg.completions_batch_size)
    
    with open(os.path.join(download_dir, 'completions', f'{dataset_name}_{intervention_label}_completions.json'), "w") as f:
        json.dump(completions, f, indent=4)

def evaluate_completions_and_save_results_for_dataset(cfg, download_dir, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(download_dir, 'completions', f'{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(download_dir, "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(os.path.join(download_dir, "completions", f"{dataset_name}_{intervention_label}_evaluations.json"), "w") as f:
        json.dump(evaluation, f, indent=4)

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(wandb.run.dir, 'loss_evals')):
        os.makedirs(os.path.join(wandb.run.dir, 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{wandb.run.dir}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)

def run_pipeline(wandb_run, sample_start_idx, sample_end_idx, save_dir):
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

    # Get model_path from wandb config
    model_path = newest_run.config.get("model")
    if not model_path:
        raise ValueError("model_path not found in wandb config for the resumed run.")
    
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)

    file_path = "samples.pt"
    base_dir = save_dir # Use the argument here
    dir_name = "cone_samples_eval"
    download_dir = os.path.join(base_dir, dir_name, newest_run.group, newest_run.name)

    os.makedirs(download_dir, exist_ok=True)
    newest_run.file(file_path).download(root=download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, file_path)
    samples = torch.load(file_path)

    eval_vectors = []
    print(f"Evaluating samples {sample_start_idx} to {sample_end_idx-1}")
    for i in range(sample_start_idx, sample_end_idx):
        eval_vectors.append({'vector': samples[i], 'name_postfix': f'_sample_{i+1}'})
    
    add_layer = newest_run.config["add_layer"]

    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    for e in eval_vectors:
        direction = e['vector']
        name_postfix = e['name_postfix']
        ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
        actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[add_layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []

        # 3a. Generate and save completions on harmful evaluation datasets
        for dataset_name in cfg.evaluation_datasets:
            if cfg.evaluate_ablation:
                generate_and_save_completions_for_dataset(cfg, download_dir, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, f'ablation{name_postfix}', dataset_name)
            if cfg.evaluate_actadd:
                generate_and_save_completions_for_dataset(cfg, download_dir, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, f'actadd{name_postfix}', dataset_name)

        # 4a. Generate and save completions on harmless evaluation dataset
        if cfg.evaluate_harmless:
            print("Evaluating on harmless instructions")
            actadd_refusal_pre_hooks, actadd_refusal_hooks = [(model_base.model_block_modules[add_layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))], []
            generate_and_save_completions_for_dataset(cfg, download_dir, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, f'actadd{name_postfix}', 'harmless', dataset=harmless_test)

        if cfg.evaluate_loss:
            # 5. Evaluate loss on harmless datasets
            evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, f'ablation{name_postfix}')
            evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, f'actadd{name_postfix}')

    for e in eval_vectors:
        print("Evaluating completions")
        name_postfix = e['name_postfix']
        model_base.del_model()
        torch.cuda.empty_cache()

        for dataset_name in cfg.evaluation_datasets:
            if cfg.evaluate_ablation:
                evaluate_completions_and_save_results_for_dataset(cfg, download_dir, f'ablation{name_postfix}', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
            if cfg.evaluate_actadd:
                evaluate_completions_and_save_results_for_dataset(cfg, download_dir, f'actadd{name_postfix}', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

        if cfg.evaluate_harmless:
            evaluate_completions_and_save_results_for_dataset(cfg, download_dir, f'actadd{name_postfix}', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(wandb_run=args.wandb_run, sample_start_idx=args.sample_start_idx, sample_end_idx=args.sample_end_idx, save_dir=args.save_dir)

