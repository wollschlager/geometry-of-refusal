# %%
import argparse
import json
import os
import os.path
import sys
import dotenv
import torch
import torch.nn as nn
import wandb
import numpy as np
from nnsight import LanguageModel
from nnsight.envoy import Envoy
from torch.utils.data import DataLoader
from transformers import set_seed

from generate_utils import (projection_einops, 
                            generate_completions,
                            intervene_with_fn_vector_ablation,
                            intervene_with_fn_vector_addition)
from scoring import refusal_metric, get_bypass_scores 

dotenv.load_dotenv(override=True)
set_seed(42)

# Default configuration values
DEFAULT_CONFIG = {
    # Model settings
    'model': 'google/gemma-2-2b-it',  # Model identifier from HuggingFace
    'dtype': 'bfloat16',              # Floating point precision (bfloat16, float16, float32)
    
    # Training objectives
    'train_direction': False,         # Whether to train a single refusal direction
    'train_orthogonal_direction': False,  # Whether to train a direction orthogonal to the DIM direction
    'train_cone': False,              # Whether to train a refusal cone
    'train_independent_direction': False, # Whether to train a direction that is independent of the DIM direction
    
    # Optimization parameters
    'epochs': 1,                      # Number of training epochs
    'lr': 1e-2,                       # Learning rate for optimization
    'batch_size': 1,                  # Batch size for training
    'effective_batch_size': 16,       # Effective batch size (uses gradient accumulation)
    'patience': 5,                    # Patience for early stopping
    'n_lr_reduce': 2,                 # Number of learning rate reductions before stopping
    
    # Cone parameters
    'min_cone_dim': 2,                # Minimum dimension of the refusal cone (number of basis vectors)
    'max_cone_dim': 3,               # Maximum dimension of the refusal cone (number of basis vectors)
    'n_sample': 8,                    # Number of random samples to use during training
    'fixed_samples': 8,               # Number of fixed samples for evaluation
    'sampling_method': "hypersphere", # Method for sampling vectors ('hypersphere' or 'interpolation')
    'optimize_basis': True,           # Whether to optimize the basis vectors directly
    
    # Loss weights
    'ablation_lambda': 1,             # Weight for the ablation loss
    'addition_lambda': 0.2,           # Weight for the addition loss
    'retain_lambda': 1,               # Weight for the retain loss
    
    # Miscellaneous
    'target_generation_batch_size': 512,  # Batch size for generating targets
    'filter_data': True,              # Whether to filter data
    'filter_batch_size': 32,          # Batch size for filtering data
    'splits': "saladbench",           # Dataset split to use
}

def parse_args():
    """
    Parse command line arguments using the default configuration values.
    
    Returns:
        argparse.Namespace: Parsed arguments with default values if not specified.
    """
    # If running in interactive mode
    if not sys.argv[0].endswith('rdo.py'):
        return argparse.Namespace(**DEFAULT_CONFIG)
    
    # If running from command line
    parser = argparse.ArgumentParser(description='Refusal Direction Optimization (RDO)')
    
    # Model settings
    parser.add_argument('--model', type=str, default=DEFAULT_CONFIG['model'],
                    help='HuggingFace model identifier (e.g., google/gemma-2-2b-it, meta-llama/Llama-3-8B)')
    parser.add_argument('--dtype', type=str, default=DEFAULT_CONFIG['dtype'],
                    choices=['bfloat16', 'float16', 'float32'],
                    help='Floating point precision to use for model initialization')
    
    # Training objectives
    parser.add_argument('--train_direction', action='store_true', 
                    help='Train a standard refusal direction')
    parser.add_argument('--train_orthogonal_direction', action='store_true', 
                    help='Train a direction orthogonal to the DIM direction')
    parser.add_argument('--train_cone', action='store_true', 
                    help='Train a refusal cone (multiple basis vectors)')
    parser.add_argument('--train_independent_direction', action='store_true', 
                    help='Train a direction that is independent of the DIM direction')
    
    # Optimization parameters
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                    help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'],
                    help='Learning rate for optimization')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                    help='Batch size for training')
    parser.add_argument('--effective_batch_size', type=int, default=DEFAULT_CONFIG['effective_batch_size'],
                    help='Effective batch size (uses gradient accumulation)')
    parser.add_argument('--patience', type=int, default=DEFAULT_CONFIG['patience'],
                    help='Patience for early stopping')
    parser.add_argument('--n_lr_reduce', type=int, default=DEFAULT_CONFIG['n_lr_reduce'],
                    help='Number of learning rate reductions before stopping')
    
    # Cone parameters
    parser.add_argument('--min_cone_dim', type=int, default=DEFAULT_CONFIG['min_cone_dim'],
                    help='Minimum dimension of the refusal cone (number of basis vectors)')
    parser.add_argument('--max_cone_dim', type=int, default=DEFAULT_CONFIG['max_cone_dim'],
                    help='Maximum dimension of the refusal cone (number of basis vectors)')
    parser.add_argument('--n_sample', type=int, default=DEFAULT_CONFIG['n_sample'],
                    help='Number of random samples to use during training')
    parser.add_argument('--fixed_samples', type=int, default=DEFAULT_CONFIG['fixed_samples'],
                    help='Number of fixed samples for evaluation')
    parser.add_argument('--sampling_method', type=str, default=DEFAULT_CONFIG['sampling_method'],
                    choices=['hypersphere', 'interpolation'],
                    help='Method for sampling vectors (hypersphere or interpolation)')
    parser.add_argument('--optimize_basis', type=bool, default=DEFAULT_CONFIG['optimize_basis'],
                    help='Whether to optimize the basis vectors directly')
    
    # Loss weights
    parser.add_argument('--ablation_lambda', type=float, default=DEFAULT_CONFIG['ablation_lambda'],
                    help='Weight for the ablation loss')
    parser.add_argument('--addition_lambda', type=float, default=DEFAULT_CONFIG['addition_lambda'],
                    help='Weight for the addition loss')
    parser.add_argument('--retain_lambda', type=float, default=DEFAULT_CONFIG['retain_lambda'],
                    help='Weight for the retain loss')
    
    # Miscellaneous
    parser.add_argument('--target_generation_batch_size', type=int, default=DEFAULT_CONFIG['target_generation_batch_size'],
                    help='Batch size for generating targets')
    parser.add_argument('--filter_data', action='store_true',
                    help='Filter data')
    parser.add_argument('--filter_batch_size', type=int, default=DEFAULT_CONFIG['filter_batch_size'],
                    help='Batch size for filtering data')
    parser.add_argument('--splits', type=str, default=DEFAULT_CONFIG['splits'],
                    help='Dataset split to use')
    
    return parser.parse_args()

args = parse_args()
MODEL_PATH = args.model

# Apply configuration values
target_generation_batch_size = args.target_generation_batch_size
splits = args.splits

# %%
# Convert string dtype to torch dtype
if args.dtype == 'bfloat16':
    dtype = torch.bfloat16
elif args.dtype == 'float16':
    dtype = torch.float16
elif args.dtype == 'float32':
    dtype = torch.float32
else:
    raise ValueError(f"Unsupported dtype: {args.dtype}")

model = LanguageModel(MODEL_PATH, cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"), device_map='auto', torch_dtype=dtype)
model.requires_grad_(False)

# %%
# loading and testing model
with model.trace("Hello") as tracer:
    pass

# %%
model_id = MODEL_PATH.split("/")[-1]
dim_dir_path = f"{os.getenv('SAVE_DIR')}/{os.getenv('DIM_DIR')}/{model_id}"
direction_file = f"{dim_dir_path}/direction.pt"
metadata_file = f"{dim_dir_path}/direction_metadata.json"
mean_diffs_file = f"{dim_dir_path}/generate_directions/mean_diffs.pt"

# Check if DIM direction files exist
if not (os.path.exists(direction_file) and os.path.exists(metadata_file)):
    raise FileNotFoundError(
        "DIM direction files not found. Please compute the DIM directions first as described in the README."
    )

refusal_directions = torch.load(mean_diffs_file)
refusal_results = json.load(open(metadata_file))
best_layer = refusal_results["layer"]
best_token = refusal_results["pos"]
best_refusal_direction = torch.load(direction_file).to(model.dtype)

# %%
SAVE_DIR = f"{os.getenv('SAVE_DIR')}/rdo/{MODEL_PATH.split('/')[-1]}/"
os.makedirs(SAVE_DIR, exist_ok=True)

add_layer = best_layer
alpha = best_refusal_direction.norm().detach().clone()
print(f"add_layer: {add_layer}, alpha: {alpha}")

# %%
harmful_train = json.load(open(f'data/{splits}_splits/harmful_train.json'))
harmless_train = json.load(open(f'data/{splits}_splits/harmless_train.json'))

harmless_train = harmless_train[:len(harmful_train)]
print(len(harmful_train), len(harmless_train))

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
    else:
        raise ValueError(f"Model {MODEL_PATH} not supported, need to configure chat template")
    return prompts

# %%
print("Chat template preview:")
print(apply_chat_template(model.tokenizer, ["Hello"])[0])
# %%
# get first token completion
def generate_first_token(model, dataset, max_new_tokens=1):
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

example_completions, example_tokens = generate_first_token(model, harmful_train[:10])
print("Example first token completions:")
print(example_completions)
print("Example first tokens:")
print(example_tokens)

# %%
if "gemma" in MODEL_PATH.lower():
    refusal_tokens = [235285]
elif "qwen2.5" in MODEL_PATH.lower():
    refusal_tokens = [40, 2121]
elif "llama-3" in MODEL_PATH.lower():
    refusal_tokens = [40]
else:
    raise ValueError(f"Model {MODEL_PATH} not supported, need to configure refusal tokens")

# %%
harmful_train_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmful_train])
harmless_train_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmless_train])
# %%
module = model.model

# %%
def generate_harmful_targets(model, harmful_instructions, best_refusal_direction, targets_path, max_new_tokens):
    if os.path.exists(targets_path):
        return json.load(open(targets_path))

    print("Generating harmful targets")
        
    ablation_completions = intervene_with_fn_vector_ablation(model, harmful_instructions, best_refusal_direction.to(model.dtype), max_new_tokens=max_new_tokens, batch_size=target_generation_batch_size)
    
    # Create batch of dicts with new targets
    targets = []
    for i, instruction in enumerate(harmful_instructions):
        target_dict = {
            'prompt': instruction,
            'ablation': ablation_completions[i] if ablation_completions else "",
        }
        targets.append(target_dict)

    # Save all targets
    os.makedirs(os.path.dirname(targets_path), exist_ok=True)
    with open(targets_path, 'w') as f:
        json.dump(targets, f)
    
    return targets

def generate_harmless_targets(model, harmless_instructions, targets_path, max_new_tokens):
    if os.path.exists(targets_path):
        return json.load(open(targets_path))

    print("Generating harmless targets")

    addition_completions = intervene_with_fn_vector_addition(model, harmless_instructions, best_layer, best_refusal_direction.norm(), best_refusal_direction, max_new_tokens=max_new_tokens, batch_size=target_generation_batch_size)

    retain_completions = generate_completions(model, harmless_instructions, max_new_tokens=max_new_tokens-1, batch_size=target_generation_batch_size)
    
    targets = []
    for i, instruction in enumerate(harmless_instructions):
        target_dict = {
            'prompt': instruction,
            'addition': addition_completions[i].split(".")[0] if addition_completions else "",
            'retain': retain_completions[i] if retain_completions else ""
        }
        targets.append(target_dict)
    
    os.makedirs(os.path.dirname(targets_path), exist_ok=True)
    with open(targets_path, 'w') as f:
        json.dump(targets, f)
        
    return targets

num_target_tokens = 30

# Set up paths
harmful_targets_path = f"{os.getenv('SAVE_DIR')}/rdo/{model_id}/{splits}/targets/harmful_targets.json"
harmless_targets_path = f"{os.getenv('SAVE_DIR')}/rdo/{model_id}/{splits}/targets/harmless_targets.json"

# Generate all targets
harmful_targets = generate_harmful_targets(model, harmful_train_instructions, best_refusal_direction, harmful_targets_path, num_target_tokens)
harmless_targets = generate_harmless_targets(model, harmless_train_instructions, harmless_targets_path, num_target_tokens)

# %%
if args.filter_data:
    print("Filtering data")
    harmful_train_scores = get_bypass_scores(model, harmful_train_instructions, refusal_tokens, batch_size=args.filter_batch_size)
    harmless_train_scores = get_bypass_scores(model, harmless_train_instructions, refusal_tokens, batch_size=args.filter_batch_size)
    
    # Filter instructions based on scores
    filtered_harmful_indices = [i for i, score in enumerate(harmful_train_scores) if score > 0]
    filtered_harmless_indices = [i for i, score in enumerate(harmless_train_scores) if score < 0]
    
    # Filter instructions
    filtered_harmful_train_instructions = [harmful_train_instructions[i] for i in filtered_harmful_indices]
    filtered_harmless_train_instructions = [harmless_train_instructions[i] for i in filtered_harmless_indices]
    
    # Filter targets
    filtered_harmful_targets = [harmful_targets[i] for i in filtered_harmful_indices]
    filtered_harmless_targets = [harmless_targets[i] for i in filtered_harmless_indices]
    
    print(f"Remaining harmful train instances: {len(filtered_harmful_train_instructions)}")
    
    # Balance datasets
    max_instances = min(len(filtered_harmful_train_instructions), len(filtered_harmless_train_instructions))
    filtered_harmful_train_instructions = filtered_harmful_train_instructions[:max_instances]
    filtered_harmless_train_instructions = filtered_harmless_train_instructions[:max_instances]
    filtered_harmful_targets = filtered_harmful_targets[:max_instances]
    filtered_harmless_targets = filtered_harmless_targets[:max_instances]
    
    print(f"Remaining harmless train instances: {len(filtered_harmless_train_instructions)}")
    
    # Update variables with filtered data
    harmful_train_instructions = filtered_harmful_train_instructions
    harmless_train_instructions = filtered_harmless_train_instructions
    harmful_targets = filtered_harmful_targets
    harmless_targets = filtered_harmless_targets

# Extract targets from filtered data
ablation_train_targets = [t["ablation"] for t in harmful_targets]
addition_train_targets = [t["addition"] for t in harmless_targets]
retain_train_targets = [t["retain"] for t in harmless_targets]

# %%
def build_prompts_and_labels(model, harmful_instructions, harmless_instructions, ablation_targets, addition_targets, retain_targets):
    ablation_prompts = []
    addition_prompts = []
    ablation_labels = []
    addition_labels = []
    retain_prompts = []
    for harmful_instruction, harmless_instruction, ablation_target, addition_target, retain_target in zip(harmful_instructions, harmless_instructions, ablation_targets, addition_targets, retain_targets):
        ablation_text = harmful_instruction + ablation_target
        addition_text = harmless_instruction + addition_target
        retain_text = harmless_instruction + retain_target
        ablation_prompts.append(ablation_text)
        addition_prompts.append(addition_text)
        retain_prompts.append(retain_text)

        # Tokenize without padding
        ablation_tokens = model.tokenizer.encode(ablation_text, add_special_tokens=True, return_tensors='pt')[0]
        addition_tokens = model.tokenizer.encode(addition_text, add_special_tokens=True, return_tensors='pt')[0]
        
        ablation_label = ablation_tokens[1:].clone()
        addition_label = addition_tokens[1:].clone()
        
        # Get the length of the instruction
        harmful_instruction_length = len(model.tokenizer.encode(harmful_instruction, add_special_tokens=True)) - 1
        harmless_instruction_length = len(model.tokenizer.encode(harmless_instruction, add_special_tokens=True)) - 1
        
        # Set labels corresponding to the instruction tokens to -100
        ablation_label[:harmful_instruction_length] = -100
        addition_label[:harmless_instruction_length] = -100

        ablation_labels.append(ablation_label)
        addition_labels.append(addition_label)
    return ablation_prompts, addition_prompts, retain_prompts, ablation_labels, addition_labels

ablation_train_prompts, addition_train_prompts, retain_train_prompts, ablation_train_labels, addition_train_labels = build_prompts_and_labels(model, harmful_train_instructions, harmless_train_instructions, ablation_train_targets, addition_train_targets, retain_train_targets)

# %%
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, harmful_prompts, harmless_prompts, ablation_prompts, ablation_targets, ablation_labels, addition_prompts, addition_targets, addition_labels, retain_prompts, retain_targets):
        self.harmful_prompts = harmful_prompts
        self.harmless_prompts = harmless_prompts
        self.ablation_prompts = ablation_prompts
        self.ablation_targets = ablation_targets
        self.ablation_labels = ablation_labels
        self.addition_prompts = addition_prompts
        self.addition_targets = addition_targets
        self.addition_labels = addition_labels
        self.retain_prompts = retain_prompts
        self.retain_targets = retain_targets

    def __len__(self):
        return len(self.harmful_prompts)
    
    def __getitem__(self, idx):
        return {
            'harmful_prompt': self.harmful_prompts[idx],
            'harmless_prompt': self.harmless_prompts[idx],
            'ablation_prompt': self.ablation_prompts[idx],
            'ablation_target': self.ablation_targets[idx],
            'ablation_labels': self.ablation_labels[idx],
            'addition_prompt': self.addition_prompts[idx],
            'addition_target': self.addition_targets[idx],
            'addition_labels': self.addition_labels[idx],
            'retain_prompt': self.retain_prompts[idx],
            'retain_target': self.retain_targets[idx],
        }

train_dataset = CustomDataset(harmful_train_instructions, harmless_train_instructions, ablation_train_prompts, ablation_train_targets, ablation_train_labels, addition_train_prompts, addition_train_targets, addition_train_labels, retain_train_prompts, retain_train_targets)
print(len(train_dataset))
print("Example item:")
d = train_dataset[0]
for item in d.items():
    print(item)

# %%
print(f"Length of harmful_prompts: {len(train_dataset.harmful_prompts)}")
print(f"Length of harmless_prompts: {len(train_dataset.harmless_prompts)}")
print(f"Length of ablation_prompts: {len(train_dataset.ablation_prompts)}")
print(f"Length of ablation_targets: {len(train_dataset.ablation_targets)}")
print(f"Length of ablation_labels: {len(train_dataset.ablation_labels)}")
print(f"Length of addition_prompts: {len(train_dataset.addition_prompts)}")
print(f"Length of addition_targets: {len(train_dataset.addition_targets)}")
print(f"Length of addition_labels: {len(train_dataset.addition_labels)}")
print(f"Length of retain_prompts: {len(train_dataset.retain_prompts)}")
print(f"Length of retain_targets: {len(train_dataset.retain_targets)}")

lengths = [
    len(train_dataset.harmful_prompts),
    len(train_dataset.harmless_prompts),
    len(train_dataset.ablation_prompts),
    len(train_dataset.ablation_targets),
    len(train_dataset.ablation_labels),
    len(train_dataset.addition_prompts),
    len(train_dataset.addition_targets),
    len(train_dataset.addition_labels),
    len(train_dataset.retain_prompts),
    len(train_dataset.retain_targets),
]
assert len(set(lengths)) == 1, f"Dataset component lengths are not equal: {lengths}"
print(f"All dataset component lengths are equal: {lengths[0]}")

# %%
def custom_collate(batch):
    return {
        'harmful_prompt': [item['harmful_prompt'] for item in batch],
        'harmless_prompt': [item['harmless_prompt'] for item in batch],
        'ablation_prompt': [item['ablation_prompt'] for item in batch],
        'ablation_target': [item['ablation_target'] for item in batch],
        'ablation_labels': torch.stack([item['ablation_labels'] for item in batch]),
        'addition_prompt': [item['addition_prompt'] for item in batch],
        'addition_target': [item['addition_target'] for item in batch], 
        'addition_labels': torch.stack([item['addition_labels'] for item in batch]),
        'retain_prompt': [item['retain_prompt'] for item in batch],
        'retain_target': [item['retain_target'] for item in batch],
    }
# %%
def sample_hypersphere_gaussian(batch_size, dim):
    # Sample from standard normal distribution
    samples = torch.randn(batch_size, dim, dtype=torch.float32, device=model.device).abs()
    # Normalize to unit length
    samples = samples / torch.norm(samples, dim=1, keepdim=True)
    return samples

def sample_prob_vectors(batch_size, dim):
    samples = torch.exp(torch.randn(batch_size, dim, dtype=torch.float32, device=model.device))
    samples = samples / samples.sum(dim=1, keepdim=True)
    return samples

def compute_ce_loss(logits, labels):
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    # Always pad labels with ignore tokens (-100) to match logits shape
    padding = torch.full((logits.size(0),), -100, device=labels.device)
    padding[-labels.size(0):] = labels
    return torch.nn.functional.cross_entropy(logits, padding, ignore_index=-100)

# def compute_ce_loss(logits, labels):
#     logits = logits.view(-1, logits.size(-1))
#     labels = labels.view(-1)
#     return torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)

def kl_div_fn(logits_a, logits_b, reduction='batchmean'):
    # Compute log-probabilities for the first distribution
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)
    
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits_a, dim=-1), 
        torch.nn.functional.softmax(logits_b, dim=-1),
        reduction=reduction
    )

def get_cosine_sims_for_vector(model, dot_vector, last_token=True):
    """Calculate cosine similarities between activations and provided vector across layers.
    
    Args:
        model: The language model
        prompt: Input prompt to get activations for
        dot_vector: Vector to compute cosine similarity against
        
    Returns:
        torch.Tensor: Tensor of cosine similarities across layers
    """
    cosine_sims = []
    for layer in model.model.layers:
        if last_token:
            cosine_sim = torch.nn.functional.cosine_similarity(layer.input[0, -1], dot_vector, dim=-1).save()
        else:
            cosine_sim = torch.nn.functional.cosine_similarity(layer.input[0, :], dot_vector, dim=-1).save()
        cosine_sims.append(cosine_sim)
    return torch.stack(cosine_sims)


def clip_grad_norm(grad, max_norm):
    total_norm = grad.norm()
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    return grad * clip_coef_clamped


class RefusalCone(nn.Module):
    def __init__(self, module: Envoy, dim: int, n_vectors: int, init_vectors: torch.Tensor | None = None, orthogonal_vectors: torch.Tensor | None = None) -> None:
        super(RefusalCone, self).__init__()
        self.module = module
        self.n_vectors = n_vectors
        self.fn_vectors = [torch.nn.Parameter(torch.randn(dim, dtype=torch.float32).cuda(), requires_grad=True) for _ in range(n_vectors)]
        if init_vectors is not None:
            for i, init_vector in enumerate(init_vectors):
                init_vector = init_vector / init_vector.norm()
                self.fn_vectors[i].data = init_vector.detach().clone().cuda().to(torch.float32)
        self.orthogonal_vectors = [(o / o.norm()).to(torch.float32).cpu() for o in orthogonal_vectors]
        self.orthogonalize()
    
    def __call__(self, direction):
        normalized_direction = direction / direction.norm()
        normalized_direction = normalized_direction.to(model.dtype)
        for layer in self.module.layers:
            self.ablate_input(layer, normalized_direction)
            self.ablate_output(layer.self_attn, normalized_direction, 3)
            self.ablate_output(layer.mlp, normalized_direction, 1)
    
    def ablate_output(self, layer, direction, tuple_length=1):
        if tuple_length > 1:
            activation = layer.output[0][:]
        else:
            activation = layer.output
        projection = projection_einops(activation, direction)
        new_activation = activation - projection
        if tuple_length == 2:
            layer.output = (new_activation, layer.output[1])
        elif tuple_length == 3:
            layer.output = (new_activation, layer.output[1], layer.output[2])
        elif tuple_length == 1:
            layer.output = new_activation
    
    def ablate_input(self, layer, direction):
        projection = projection_einops(layer.input, direction)
        new_activation = layer.input - projection
        layer.input = new_activation

    def add(self, direction, alpha, layer_idx):
        direction = direction / direction.norm()
        direction = direction.to(model.dtype)
        self.module.layers[layer_idx].input += alpha * direction
    
    def transform(self, sample):
        fn_vectors = torch.stack(self.fn_vectors, dim=0)
        transformed_sample = torch.matmul(sample, fn_vectors).to(model.dtype)
        transformed_sample = transformed_sample / torch.norm(transformed_sample)
        return transformed_sample

    def parameters(self):
        return self.fn_vectors
    
    def orthogonalize(self):
        with torch.no_grad():
            for i in range(len(self.fn_vectors)):
                for j in range(i):
                    self.fn_vectors[i].data.sub_(projection_einops(self.fn_vectors[i].data, self.fn_vectors[j].data))
                self.fn_vectors[i].data.div_(self.fn_vectors[i].data.norm())
            
            if self.orthogonal_vectors:
                v = self.fn_vectors[0].data.clone().cpu()
                
                # Stack your vectors as rows in a matrix A
                A = torch.stack([vec.flatten().to(torch.float32) for vec in self.orthogonal_vectors])
                # Compute projection matrix P = A^T(AA^T)^-1A
                # The nullspace projector is then I - P
                AAT = A @ A.t()
                AAT_inv = torch.inverse(AAT)
                P = A.t() @ AAT_inv @ A
                I = torch.eye(P.shape[0], device=P.device)
                
                # Project onto nullspace (orthogonal complement)
                v_flat = v.flatten()
                v_ortho = (I - P) @ v_flat
                
                # Reshape back to original shape and normalize
                v_ortho = v_ortho.reshape(v.shape)
                v_ortho = v_ortho / torch.norm(v_ortho)

                self.fn_vectors[0].data = v_ortho.to(self.fn_vectors[0].dtype).to(self.fn_vectors[0].device)
    
    def normalize(self):
        for i in range(len(self.fn_vectors)):
            self.fn_vectors[i].data.div_(self.fn_vectors[i].data.norm())
                    

def refusal_cone_optimization(model, train_dataset, 
                              batch_size=DEFAULT_CONFIG['batch_size'], 
                              effective_batch_size=DEFAULT_CONFIG['effective_batch_size'], 
                              epochs=DEFAULT_CONFIG['epochs'], 
                              lr=DEFAULT_CONFIG['lr'], 
                              cone_dim=1,
                              n_sample=DEFAULT_CONFIG['n_sample'], 
                              fixed_samples=DEFAULT_CONFIG['fixed_samples'], 
                              sampling_method=DEFAULT_CONFIG['sampling_method'], 
                              optimize_basis=DEFAULT_CONFIG['optimize_basis'], 
                              fixed_basis_vectors=[], 
                              ablation_lambda=DEFAULT_CONFIG['ablation_lambda'], 
                              alpha=alpha,  # Keep alpha as it's defined globally
                              addition_lambda=DEFAULT_CONFIG['addition_lambda'], 
                              retain_lambda=DEFAULT_CONFIG['retain_lambda'], 
                              patience=DEFAULT_CONFIG['patience'], 
                              init_vectors=[], 
                              n_lr_reduce=DEFAULT_CONFIG['n_lr_reduce'], 
                              orthogonal_vectors=[]):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate)

    operation = RefusalCone(model.model, model.config.hidden_size, cone_dim, init_vectors=init_vectors, orthogonal_vectors=orthogonal_vectors)

    optimizer = torch.optim.AdamW(operation.parameters(), lr=lr, betas=(.9,.98), weight_decay=0.0, amsgrad=True)

    print("Cone dim", cone_dim)
    if cone_dim == 1:
        n_sample = 0

    accumulation_steps = effective_batch_size // batch_size
    print("Accumulation steps", accumulation_steps)
    vectors = []
    train_losses = []
    stopped = False
    lowest_training_loss = float('inf')
    bypass_scores = []
    patience_counter = 0
    lr_reduce_counter = 0

    print("Starting training")

    step_counter = 0
    batch_sample_ablation_loss = 0.0
    batch_sample_addition_loss = 0.0
    batch_sample_retain_loss = 0.0
    batch_basis_ablation_loss = 0.0
    batch_basis_addition_loss = 0.0
    batch_basis_retain_loss = 0.0

    batch_sample_bypass_scores = []
    batch_sample_induce_scores = []
    batch_basis_bypass_scores = []
    batch_basis_induce_scores = []

    add_layer = best_layer

    if n_sample > 0:
        if sampling_method == "hypersphere":
            fixed_sample_vectors = sample_hypersphere_gaussian(fixed_samples, cone_dim)
        elif sampling_method == "interpolation":
            fixed_sample_vectors = sample_prob_vectors(fixed_samples, cone_dim)
        fixed_sample_vectors = [fixed_sample_vectors[i] for i in range(fixed_samples)]

    for epoch in range(epochs):
        print('Epoch', epoch)
        for _, batch in enumerate(train_dataloader):
            ablation_prompt = batch['ablation_prompt']
            ablation_labels = batch['ablation_labels']
            addition_prompt = batch['addition_prompt']
            addition_labels = batch['addition_labels']
            retain_prompt = batch['retain_prompt']
            harmful_prompt = batch['harmful_prompt']
            harmless_prompt = batch['harmless_prompt']

            if n_sample > 0:
                if sampling_method == "hypersphere":
                    sample_vectors = sample_hypersphere_gaussian(n_sample, cone_dim)
                elif sampling_method == "interpolation":
                    sample_vectors = sample_prob_vectors(n_sample, cone_dim)

                for sample_vector in sample_vectors:
                    if ablation_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(ablation_prompt):
                                direction = operation.transform(sample_vector)
                                operation(direction)
                                logits = model.lm_head.output[:, :-1]
                                sample_ablation_loss = compute_ce_loss(logits, ablation_labels) / n_sample
                                log = sample_ablation_loss.detach().item().save()
                            (ablation_lambda * sample_ablation_loss).backward()
                    batch_sample_ablation_loss += log
                    if addition_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(addition_prompt):
                                direction = operation.transform(sample_vector)
                                operation.add(direction, alpha, add_layer)
                                logits = model.lm_head.output[:, :-1]
                                sample_addition_loss = compute_ce_loss(logits, addition_labels) / n_sample
                                log = sample_addition_loss.detach().item().save()
                            (addition_lambda * sample_addition_loss).backward()
                        batch_sample_addition_loss += log
                    if retain_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(retain_prompt):
                                baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                            with tracer.invoke(retain_prompt):
                                direction = operation.transform(sample_vector)
                                operation(direction)
                                sample_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                                sample_retain_loss = kl_div_fn(baseline_retain_logits, sample_retain_logits).mean() / n_sample
                                log = sample_retain_loss.detach().item().save()
                            (retain_lambda * sample_retain_loss).backward()
                        batch_sample_retain_loss += log
                
            if optimize_basis:
                for fn_vector in operation.fn_vectors:
                    if ablation_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(ablation_prompt):
                                operation(fn_vector)
                                logits = model.lm_head.output[:, :-1]
                                basis_ablation_loss = compute_ce_loss(logits, ablation_labels) / cone_dim
                                log = basis_ablation_loss.detach().item().save()
                            (ablation_lambda * basis_ablation_loss).backward()
                        batch_basis_ablation_loss += log

                    if addition_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(addition_prompt):
                                operation.add(fn_vector, alpha, add_layer)
                                logits = model.lm_head.output[:, :-1]
                                basis_addition_loss = compute_ce_loss(logits, addition_labels) / cone_dim
                                log = basis_addition_loss.detach().item().save()
                            (addition_lambda * basis_addition_loss).backward()
                        batch_basis_addition_loss += log

                    if retain_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(retain_prompt):
                                baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                            with tracer.invoke(retain_prompt):
                                operation(fn_vector)
                                retain_logits = model.lm_head.output[:, -num_target_tokens:]
                                basis_retain_loss = kl_div_fn(baseline_retain_logits, retain_logits).mean() / cone_dim
                                log = basis_retain_loss.detach().item().save()
                            (retain_lambda * basis_retain_loss).backward()
                        batch_basis_retain_loss += log
                
            with torch.no_grad():
                with model.trace() as tracer:
                    for fn_vector in operation.fn_vectors:
                        with tracer.invoke(harmful_prompt):
                            operation(fn_vector)
                            last_token_logits = model.lm_head.output[:, -1]
                            bypass_score = refusal_metric(last_token_logits, refusal_tokens).detach().item().save()
                        batch_basis_bypass_scores.append(bypass_score)

                with model.trace() as tracer:
                    for fn_vector in operation.fn_vectors:
                        with tracer.invoke(harmless_prompt):
                            operation.add(fn_vector, alpha, add_layer)
                            last_token_logits = model.lm_head.output[:, -1]
                            induce_score = refusal_metric(last_token_logits, refusal_tokens).detach().item().save()
                        batch_basis_induce_scores.append(induce_score)
                if n_sample > 0:
                    with model.trace() as tracer:
                        for fixed_sample_vector in fixed_sample_vectors:
                            with tracer.invoke(harmful_prompt):
                                direction = operation.transform(fixed_sample_vector)
                                operation(direction)
                                sample_last_token_logits = model.lm_head.output[:, -1]
                                sample_bypass_score = refusal_metric(sample_last_token_logits, refusal_tokens).detach().item().save()
                                batch_sample_bypass_scores.append(sample_bypass_score)
                    with model.trace() as tracer:
                        for fixed_sample_vector in fixed_sample_vectors:
                            with tracer.invoke(harmless_prompt):
                                direction = operation.transform(fixed_sample_vector)
                                operation.add(direction, alpha, add_layer)
                                sample_last_token_logits = model.lm_head.output[:, -1]
                                sample_induce_score = refusal_metric(sample_last_token_logits, refusal_tokens).detach().item().save()
                                batch_sample_induce_scores.append(sample_induce_score)

                step_counter += 1
                if step_counter % accumulation_steps == 0:
                    for fn_vector in operation.fn_vectors:
                        fn_vector.grad.sub_(projection_einops(fn_vector.grad, fn_vector.data))
                    for fn_vector in operation.fn_vectors:
                        fn_vector.grad.div_(accumulation_steps)
                    torch.nn.utils.clip_grad_norm_(operation.parameters(), 10.0)
                    grad_norm = operation.fn_vectors[-1].grad.norm().item()
                    optimizer.step()
                    optimizer.zero_grad()
                    if len(fixed_basis_vectors) > 0:
                        for i, fixed_basis_vector in enumerate(fixed_basis_vectors):
                            fixed_basis_vector = fixed_basis_vector / fixed_basis_vector.norm()
                            operation.fn_vectors[i].data.copy_(fixed_basis_vector.data)
                    operation.orthogonalize()

                    batch_sample_ablation_loss /= accumulation_steps
                    batch_sample_addition_loss /= accumulation_steps
                    batch_sample_retain_loss /= accumulation_steps
                    batch_basis_ablation_loss /= accumulation_steps
                    batch_basis_addition_loss /= accumulation_steps
                    batch_basis_retain_loss /= accumulation_steps

                    train_loss = batch_sample_ablation_loss + batch_sample_addition_loss + batch_sample_retain_loss + batch_basis_ablation_loss + batch_basis_addition_loss + batch_basis_retain_loss
                    train_losses.append(train_loss)

                    batch_basis_bypass_scores = [s.value for s in batch_basis_bypass_scores]
                    batch_basis_induce_scores = [s.value for s in batch_basis_induce_scores]
                    basis_bypass_scores = [torch.mean(torch.tensor(batch_basis_bypass_scores[i::cone_dim])).item() for i in range(cone_dim)]
                    basis_induce_scores = [torch.mean(torch.tensor(batch_basis_induce_scores[i::cone_dim])).item() for i in range(cone_dim)]

                    vectors.append(torch.stack(operation.fn_vectors, dim=0).detach().cpu().data.clone())
                    bypass_scores.append(basis_bypass_scores)

                    wandb_logs = {
                        "train/total_loss": train_loss,
                        "train/basis_ablation_loss": batch_basis_ablation_loss,
                        "train/basis_addition_loss": batch_basis_addition_loss,
                        "train/basis_retain_loss": batch_basis_retain_loss,
                        "train/basis_bypass_score": basis_bypass_scores,
                        "train/basis_induce_score": basis_induce_scores,
                        "train/grad_norm": grad_norm
                    }

                    if n_sample > 0:
                        batch_sample_bypass_scores = [s.value for s in batch_sample_bypass_scores]
                        batch_sample_induce_scores = [s.value for s in batch_sample_induce_scores]

                        sample_vector_bypass_scores = [torch.mean(torch.tensor(batch_sample_bypass_scores[i::fixed_samples])).item() for i in range(fixed_samples)]
                        min_sample_bypass_score = min(sample_vector_bypass_scores)
                        max_sample_bypass_score = max(sample_vector_bypass_scores)
                        mean_sample_bypass_score = torch.mean(torch.tensor(sample_vector_bypass_scores)).item()
                        std_sample_bypass_score = torch.std(torch.tensor(sample_vector_bypass_scores)).item()

                        sample_vector_induce_scores = [torch.mean(torch.tensor(batch_sample_induce_scores[i::fixed_samples])).item() for i in range(fixed_samples)]
                        min_sample_induce_score = min(sample_vector_induce_scores)
                        max_sample_induce_score = max(sample_vector_induce_scores)
                        mean_sample_induce_score = torch.mean(torch.tensor(sample_vector_induce_scores)).item()
                        std_sample_induce_score = torch.std(torch.tensor(sample_vector_induce_scores)).item()

                        wandb_logs.update({
                            "train/sample_ablation_loss": batch_sample_ablation_loss if n_sample > 0 else 0,
                            "train/sample_addition_loss": batch_sample_addition_loss if n_sample > 0 else 0,
                            "train/sample_retain_loss": batch_sample_retain_loss if n_sample > 0 else 0,
                            "train/min_sample_bypass_score": min_sample_bypass_score if n_sample > 0 else 0,
                            "train/max_sample_bypass_score": max_sample_bypass_score if n_sample > 0 else 0,
                            "train/mean_sample_bypass_score": mean_sample_bypass_score if n_sample > 0 else 0,
                            "train/std_sample_bypass_score": std_sample_bypass_score if n_sample > 0 else 0,
                            "train/min_sample_induce_score": min_sample_induce_score if n_sample > 0 else 0,
                            "train/max_sample_induce_score": max_sample_induce_score if n_sample > 0 else 0,
                            "train/mean_sample_induce_score": mean_sample_induce_score if n_sample > 0 else 0,
                            "train/std_sample_induce_score": std_sample_induce_score if n_sample > 0 else 0,
                        })

                    wandb.log(wandb_logs, step=step_counter)

                    print("Step", step_counter, "train/basis_vector_bypass_score", [round(s, 2) for s in basis_bypass_scores], "train/basis_vector_induce_score", [round(s, 2) for s in basis_induce_scores])
                    
                    if n_sample > 0:
                        print("train/mean_sample_bypass_scores", round(mean_sample_bypass_score, 2), "train/mean_sample_induce_scores", round(mean_sample_induce_score, 2))
                    if train_loss >= lowest_training_loss:
                        patience_counter += 1
                    else:
                        lowest_training_loss = train_loss
                        patience_counter = 0
                    if patience_counter >= patience:
                        if lr_reduce_counter >= n_lr_reduce:
                            print(f'Stopping')
                            stopped = True
                            break
                        lr_reduce_counter += 1
                        print("Reducing lr to", optimizer.param_groups[0]['lr'] / 10)
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                        patience_counter = 0

                    batch_sample_ablation_loss = 0.
                    batch_sample_addition_loss = 0.
                    batch_sample_retain_loss = 0.
                    batch_basis_ablation_loss = 0.
                    batch_basis_addition_loss = 0.
                    batch_basis_retain_loss = 0.

                    batch_sample_bypass_scores = []
                    batch_sample_induce_scores = []
                    batch_basis_bypass_scores = []
                    batch_basis_induce_scores = []

                    torch.cuda.empty_cache()

        if stopped:
            break

    save_vectors = vectors
    lowest_loss_index = torch.argmin(torch.tensor(train_losses)).item()
    lowest_loss_vector = save_vectors[lowest_loss_index]
    run_id = wandb.run.id
    artifact = wandb.Artifact(f'trained_vectors_run_{run_id}', type='vector')
    with artifact.new_file(f'vectors.pt', mode='wb') as f:
        torch.save(save_vectors, f)
    with artifact.new_file(f'lowest_loss_vector.pt', mode='wb') as f:
        torch.save(lowest_loss_vector, f)
    wandb.log_artifact(artifact)
    
    return {"vectors": vectors, "lowest_loss": lowest_training_loss, "refusal_scores": bypass_scores, "train_losses": train_losses, "lowest_loss_vector": lowest_loss_vector}

# %%
def train_refusal_vector(group_name=None, run_name=None, orthogonal_vectors=[], **kwargs):
    """
    Train a single refusal direction vector.

    Args:
        group_name: Name of the wandb group
        run_name: Name of the wandb run
        orthogonal_vectors: List of vectors to which the trained vector should be orthogonal
        **kwargs: Additional parameters to override the defaults

    Returns:
        Dictionary containing training results
    """
    # Combine specific args and potential overrides from kwargs
    train_kwargs = {
        "cone_dim": 1,  # Specific override for single direction training
        "orthogonal_vectors": orthogonal_vectors,
    }
    train_kwargs.update(kwargs) # Apply any user-provided overrides

    # Prepare wandb config starting with all parsed args
    wandb_config = vars(args).copy()
    # Add metadata not present in args
    wandb_config.update({
        "model_id": model_id,
        "add_layer": add_layer,
        "alpha": alpha,
    })
    # Update wandb config with specific overrides for this run (from train_kwargs)
    # This ensures cone_dim=1 and any passed kwargs are logged correctly
    wandb_config.update(train_kwargs)
    # Remove vectors from wandb config as they are large and not serializable
    wandb_config.pop('orthogonal_vectors', None)
    wandb_config.pop('train_cone', None)
    wandb_config.pop('train_direction', None)
    wandb_config.pop('train_orthogonal_direction', None)
    wandb_config.pop('train_independent_direction', None)

    # Initialize wandb run
    with wandb.init(project=os.getenv("WANDB_PROJECT"),
                   config=wandb_config,
                   group=f"{group_name}_{model_id}",
                   name=run_name,
                   mode="online"):

        # Run optimization, passing only necessary/overridden args
        results = refusal_cone_optimization(
            model=model,
            train_dataset=train_dataset,
            **train_kwargs # Pass cone_dim=1, orthogonal_vectors, and any other kwargs
        )

    wandb.finish()
    return results

# Conditional training based on command line arguments
if args.train_direction:
    print("Training standard refusal direction")
    group_name = "basic_rdo" 
    run_name = None
    train_refusal_vector(
        group_name=group_name,
        run_name=run_name,
    )

if args.train_orthogonal_direction:
    print("Training orthogonal refusal direction")
    group_name = "basic_rdo_orthogonal" 
    orthogonal_vectors = [best_refusal_direction]
    run_name = None
    train_refusal_vector(
        group_name=group_name,
        run_name=run_name,
        orthogonal_vectors=orthogonal_vectors,
    )

# %%
def train_refusal_cone(group_name, run_name, init_vectors, **kwargs):
    """
    Train a refusal cone with multiple basis vectors.

    Args:
        group_name: Name of the wandb group
        run_name: Name of the wandb run
        init_vectors: Initial vectors for the cone
        **kwargs: Additional parameters to override the defaults

    Returns:
        Dictionary containing training results
    """
    # Combine specific args and potential overrides from kwargs
    train_kwargs = {
        "init_vectors": init_vectors,
        # cone_dim will be added from kwargs if present, otherwise defaults from args will be used later
    }
    # Apply any user-provided overrides from kwargs
    # This might override cone_dim, which is desired.
    train_kwargs.update(kwargs)

    # Prepare wandb config starting with all parsed args
    wandb_config = vars(args).copy()
    # Add metadata not present in args
    wandb_config.update({
        "model_id": model_id,
        "add_layer": add_layer,
        "alpha": alpha,
    })
    # Update wandb config with specific overrides for this run (from train_kwargs)
    # This ensures the correct cone_dim (from kwargs or args) are logged
    wandb_config.update(train_kwargs)
    # Remove vectors from wandb config as they are large and not serializable
    wandb_config.pop('init_vectors', None)
    wandb_config.pop('train_cone', None)
    wandb_config.pop('train_direction', None)
    wandb_config.pop('train_orthogonal_direction', None)
    wandb_config.pop('train_independent_direction', None)

    # Initialize wandb run
    with wandb.init(project=os.getenv("WANDB_PROJECT"),
                   config=wandb_config,
                   group=f"{group_name}_{model_id}",
                   name=run_name,
                   mode="online"):

        # Run optimization, passing only necessary/overridden args
        results = refusal_cone_optimization(
            model=model,
            train_dataset=train_dataset,
            **train_kwargs # Pass init_vectors and any other kwargs (including cone_dim)
        )

    wandb.finish()
    return results

if args.train_cone:
    print(f"Training refusal cone with dimensions from {args.min_cone_dim} to {args.max_cone_dim}")
    subspace_dimensions = range(args.min_cone_dim, args.max_cone_dim + 1)
    group_name = "basic_rco"
    init_vectors = []

    for i in subspace_dimensions:
        run_name = f"dim_{i}"
        print(f"Training dimension {i}")
        results = train_refusal_cone(
            group_name=group_name, 
            run_name=run_name, 
            init_vectors=init_vectors,
            cone_dim=i
        )
        lowest_loss_vector = results['lowest_loss_vector']
        init_vectors = list(lowest_loss_vector)
# %%
import nnsight

class DirectionalAblation(nn.Module):
    def __init__(self, module, dim, orthogonal_vectors, add_layer_idx, alpha, init_vector = None):
        super(DirectionalAblation, self).__init__()
        self.module = module
        self.fn_vector = torch.nn.Parameter(
            init_vector.to(torch.float32) if init_vector is not None else torch.randn(dim, dtype=torch.float32),
            requires_grad=True
        ).save()
        self.alpha = torch.nn.Parameter(torch.tensor(float(alpha.item())).to(torch.float32), requires_grad=False).save()
        nnsight.log("alpha", self.alpha)
        self.orthogonal_vectors = [(o / o.norm()).to(torch.float32).cpu() for o in orthogonal_vectors]
        self.add_layer_idx = add_layer_idx

        self.orthogonalize()

    def __call__(self, direction):
        direction = direction / direction.norm()
        direction = direction.to(model.dtype)
        for layer in self.module.layers:
            self.ablate_input(layer, direction)
            self.ablate_output(layer.self_attn, direction, 3)
            self.ablate_output(layer.mlp, direction, 1)
    
    def ablate_output(self, layer, direction, tuple_length=1):
        if tuple_length > 1:
            activation = layer.output[0][:]
        else:
            activation = layer.output
        projection = projection_einops(activation, direction)
        new_activation = activation - projection
        if tuple_length == 2:
            layer.output = (new_activation, layer.output[1])
        elif tuple_length == 3:
            layer.output = (new_activation, layer.output[1], layer.output[2])
        elif tuple_length == 1:
            layer.output = new_activation
    
    def ablate_input(self, layer, direction):
        projection = projection_einops(layer.input, direction)
        new_activation = layer.input - projection
        layer.input = new_activation

    def add(self, direction):
        direction = direction / direction.norm()
        direction = direction.to(model.dtype)
        self.module.layers[self.add_layer_idx].input += self.alpha * direction

    def normalize(self):
        with torch.no_grad():
            self.fn_vector.data = self.fn_vector.data / self.fn_vector.data.norm()

    def orthogonalize(self):
        if self.orthogonal_vectors:
            self.fn_vector.data = nnsight.apply(self._orthogonalize, self.fn_vector)

    def _orthogonalize(self, vector):
        with torch.no_grad():
            v = vector.data.clone().cpu()
            
            # Stack your vectors as rows in a matrix A
            A = torch.stack([vec.flatten().to(torch.float32) for vec in self.orthogonal_vectors])
            # Compute projection matrix P = A^T(AA^T)^-1A
            # The nullspace projector is then I - P
            AAT = A @ A.t()
            AAT_inv = torch.inverse(AAT)
            P = A.t() @ AAT_inv @ A
            I = torch.eye(P.shape[0], device=P.device)
            
            # Project onto nullspace (orthogonal complement)
            v_flat = v.flatten()
            v_ortho = (I - P) @ v_flat
            
            # Reshape back to original shape and normalize
            v_ortho = v_ortho.reshape(v.shape)
            v_ortho = v_ortho / torch.norm(v_ortho)
                
            return v_ortho.to(vector.dtype).to(vector.device)

def repind_rdo(model,
               train_dataset,
               batch_size=DEFAULT_CONFIG['batch_size'],
               effective_batch_size=DEFAULT_CONFIG['effective_batch_size'],
               epochs=DEFAULT_CONFIG['epochs'],
               lr=DEFAULT_CONFIG['lr'],
               ablation_lambda=DEFAULT_CONFIG['ablation_lambda'],
               addition_lambda=DEFAULT_CONFIG['addition_lambda'],
               retain_lambda=DEFAULT_CONFIG['retain_lambda'],
               patience=DEFAULT_CONFIG['patience'],
               n_lr_reduce=DEFAULT_CONFIG['n_lr_reduce'],
               alpha=alpha,
               orthogonal_vectors=[],
               repind_layers=[],
               repind_lambda=1,
               independent_vectors=[],
               verbose=False,
               init_vector=None):

    with model.session() as session:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate)

        # Handle independent vectors safely
        processed_independent_vectors = []
        for ind in independent_vectors:
            if ind is not None:
                processed_independent_vectors.append(ind.detach().clone().cuda().to(model.dtype))
        independent_vectors = processed_independent_vectors
        
        # Handle init_vector safely
        if init_vector is not None:
            init_vector = init_vector.detach().clone().cuda().to(model.dtype)

        operation = DirectionalAblation(model.model, model.config.hidden_size, orthogonal_vectors, best_layer, alpha, init_vector=init_vector)

        parameters = [{"params": operation.fn_vector, "lr": lr}]
        optimizer = torch.optim.AdamW(parameters, betas=(.9,.98), weight_decay=0.0, amsgrad=True)

        # log optimizer parameters
        accumulation_steps = effective_batch_size // batch_size
        if verbose:
            print("Accumulation steps", accumulation_steps)

        vectors = nnsight.list().save()
        train_losses = nnsight.list().save()
        lowest_training_loss = nnsight.float(float('inf')).save()
        patience_counter = nnsight.int(0).save()
        lr_reduce_counter = nnsight.int(0).save()
        stopped = nnsight.bool(False).save()

        if verbose:
            print("Starting training")

        step_counter = nnsight.int(0)
        batch_ablation_loss = nnsight.float(0)
        batch_addition_loss = nnsight.float(0)
        batch_repind_loss = nnsight.float(0)
        batch_retain_loss = nnsight.float(0)
        batch_refusal_score = nnsight.float(0)
        batch_induce_score = nnsight.float(0)
        batch_gradients = nnsight.list()

        with session.iter(range(epochs), return_context=True) as (epoch, epoch_iterator):
            if verbose:
                epoch_iterator.log('Epoch', epoch)
            with session.iter(train_dataloader, return_context=True) as (batch, train_iterator):
                ablation_prompt = batch['ablation_prompt']
                ablation_labels = batch['ablation_labels']
                addition_prompt = batch['addition_prompt']
                addition_labels = batch['addition_labels']
                retain_prompt = batch['retain_prompt']
                harmful_prompt = batch['harmful_prompt']
                harmless_prompt = batch['harmless_prompt']

                if ablation_lambda > 0:
                    with model.trace() as tracer:
                        ablation_loss = torch.tensor(0.0, device=model.device, dtype=model.dtype)
                        with tracer.invoke(ablation_prompt):
                            operation(operation.fn_vector)
                            logits = model.lm_head.output[:, :-1]
                            ablation_loss += compute_ce_loss(logits, ablation_labels)
                        ablation_loss = ablation_loss / accumulation_steps
                        batch_ablation_loss.update(batch_ablation_loss + ablation_loss.detach().item())
                        (ablation_lambda * ablation_loss).backward()

                if repind_lambda > 0:
                    with model.trace() as tracer:
                        repind_losses = []
                        for independent_vector in independent_vectors:
                            with tracer.invoke(harmful_prompt):
                                target_independent_cosine_sims = get_cosine_sims_for_vector(model, independent_vector)
                                target_fn_vector_cosine_sims = get_cosine_sims_for_vector(model, operation.fn_vector)

                            with tracer.invoke(harmful_prompt):
                                operation(operation.fn_vector)
                                current_independent_cosine_sims = get_cosine_sims_for_vector(model, independent_vector)
                                repind_losses.append((current_independent_cosine_sims - target_independent_cosine_sims)[repind_layers].square().mean())

                            with tracer.invoke(harmful_prompt):
                                operation(independent_vector)
                                current_fn_vector_cosine_sims = get_cosine_sims_for_vector(model, operation.fn_vector)
                                repind_losses.append((current_fn_vector_cosine_sims - target_fn_vector_cosine_sims)[repind_layers].square().mean())

                        repind_loss = torch.stack(repind_losses).sum()
                        repind_loss = repind_loss / accumulation_steps
                        batch_repind_loss.update(batch_repind_loss + repind_loss.detach().item())
                        (repind_lambda * repind_loss).backward()

                if addition_lambda > 0:
                    addition_loss = torch.tensor(0.0, device=model.device, dtype=model.dtype)
                    with model.trace() as tracer:
                        with tracer.invoke(addition_prompt) as _:
                            operation.add(operation.fn_vector)
                            logits = model.lm_head.output[:, :-1]
                            addition_loss += compute_ce_loss(logits, addition_labels)
                    addition_loss = addition_loss / accumulation_steps
                    batch_addition_loss.update(batch_addition_loss + addition_loss.detach().item())
                    (addition_lambda * addition_loss).backward()

                if retain_lambda > 0:
                    retain_loss = torch.tensor(0.0, device=model.device, dtype=model.dtype)
                    with model.trace() as tracer:
                        with tracer.invoke(retain_prompt):
                            baseline_retain_logits = model.lm_head.output[:, -num_target_tokens:]
                        with tracer.invoke(retain_prompt):
                            operation(operation.fn_vector)
                            retain_logits = model.lm_head.output[:, -num_target_tokens:]
                            retain_loss += kl_div_fn(baseline_retain_logits, retain_logits).mean()
                        retain_loss = retain_loss / accumulation_steps
                        batch_retain_loss.update(batch_retain_loss + retain_loss.detach().item())
                        (retain_lambda * retain_loss).backward()

                with torch.no_grad():
                    with model.trace() as tracer:
                        with tracer.invoke(harmful_prompt) as _:
                            operation(operation.fn_vector)
                            last_token_logits = model.lm_head.output[:, -1]
                            refusal_score = refusal_metric(last_token_logits, refusal_tokens)
                            refusal_score = refusal_score / accumulation_steps
                            batch_refusal_score.update(batch_refusal_score + refusal_score.detach().item())
                    with model.trace() as tracer:
                        with tracer.invoke(harmless_prompt) as _:
                            operation.add(operation.fn_vector)
                            last_token_logits = model.lm_head.output[:, -1]
                            induce_score = refusal_metric(last_token_logits, refusal_tokens)
                            induce_score = induce_score / accumulation_steps
                            batch_induce_score.update(batch_induce_score + induce_score.detach().item())

                    step_counter.update(step_counter + 1)
                    batch_gradients.append(operation.fn_vector.grad.detach().cpu())
                    with train_iterator.cond(step_counter % accumulation_steps == 0):
                        grad_sum = nnsight.apply(sum, batch_gradients)
                        grad_sum = nnsight.apply(lambda x, y: x - projection_einops(x, y / y.norm()), grad_sum, operation.fn_vector) # project gradient to tangent of sphere
                        grad_sum = nnsight.apply(lambda x, y: x - projection_einops(x, y / y.norm()), grad_sum, operation.fn_vector) # project gradient to tangent of sphere
                        grad_norm = grad_sum.norm().item()
                        operation.fn_vector.grad = grad_sum
                        optimizer.step()
                        optimizer.zero_grad()

                        if orthogonal_vectors:
                            operation.orthogonalize()
                        operation.normalize()
                        train_loss = batch_ablation_loss + batch_addition_loss + batch_repind_loss + batch_retain_loss
                        train_losses.append(train_loss)

                        vectors.append(operation.fn_vector.detach().data.clone())

                        nnsight.apply(wandb.log, {
                            "train/total_loss": train_loss,
                            "train/ablation_loss": batch_ablation_loss,
                            "train/addition_loss": batch_addition_loss,
                            "train/repind_loss": batch_repind_loss,
                            "train/retain_loss": batch_retain_loss,
                            "train/refusal_score": batch_refusal_score,
                            "train/induce_score": batch_induce_score,
                            "train/grad_norm": grad_norm,
                        }, step=step_counter)
                        nnsight.log("Step", step_counter, "train/refusal_score", batch_refusal_score, "train/induce_score", batch_induce_score)

                        with train_iterator.cond(train_loss >= lowest_training_loss):
                            patience_counter.update(patience_counter + 1)
                        with train_iterator.cond(train_loss < lowest_training_loss):
                            lowest_training_loss.update(train_loss)
                            patience_counter.update(0)
                        with train_iterator.cond(patience_counter >= patience):
                            with train_iterator.cond(lr_reduce_counter >= n_lr_reduce):
                                if verbose:
                                    nnsight.log(f'Stopping')
                                stopped.update(True)
                                train_iterator.exit()
                            with train_iterator.cond(lr_reduce_counter < n_lr_reduce):
                                lr_reduce_counter.update(lr_reduce_counter + 1)
                                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                                nnsight.log("Reducing lr to", optimizer.param_groups[0]['lr'])
                                patience_counter.update(0)

                        batch_ablation_loss.update(0)
                        batch_addition_loss.update(0)
                        batch_repind_loss.update(0)
                        batch_retain_loss.update(0)
                        batch_refusal_score.update(0)
                        batch_induce_score.update(0)
                        batch_gradients.update([])

            with epoch_iterator.cond(stopped == True):
                epoch_iterator.exit()

    save_vectors = vectors.value
    run_id = wandb.run.id
    lowest_loss_index = torch.argmin(torch.tensor(train_losses.value)).item()
    lowest_loss_vector = vectors.value[lowest_loss_index]
    artifact = wandb.Artifact(f'trained_vectors_run_{run_id}', type='vector')
    with artifact.new_file('vector.pt', mode='wb') as f:
        torch.save(save_vectors, f)
    with artifact.new_file('lowest_loss_vector.pt', mode='wb') as f:
        torch.save(lowest_loss_vector, f)
    wandb.log_artifact(artifact)

    return {"vectors": save_vectors, "lowest_loss_vector": lowest_loss_vector}

# %%
def train_independent_vector(group_name=None, run_name=None, independent_vectors=None, **kwargs):
    train_kwargs = {
        'epochs': 2,
        'retain_lambda': 0.1,
        'repind_lambda': 200,
        'repind_layers': repind_layers,
        'independent_vectors': independent_vectors,
    }
    train_kwargs.update(kwargs) # Apply any user-provided overrides

    # Prepare wandb config starting with all parsed args
    wandb_config = vars(args).copy()
    # Add metadata not present in args
    wandb_config.update({
        "model_id": model_id,
        "add_layer": add_layer,
        "alpha": alpha,
    })

    wandb_config.update(train_kwargs)
    wandb_config.pop('independent_vectors', None) # Remove from wandb config as it's handled separately
    wandb_config.pop('train_cone', None)
    wandb_config.pop('train_direction', None)
    wandb_config.pop('train_orthogonal_direction', None)
    wandb_config.pop('train_independent_direction', None)

    with wandb.init(project=os.getenv("WANDB_PROJECT"),
                   config=wandb_config,
                   group=f"{group_name}_{model_id}",
                   name=run_name,
                   mode="online"):
        results = repind_rdo(
            model=model,
            train_dataset=train_dataset,
            verbose=True,
            **train_kwargs
    )
    wandb.finish()
    return results

# %%
if args.train_independent_direction:
    harmful_val = json.load(open(f'data/{splits}_splits/harmful_val.json'))
    harmless_val = json.load(open(f'data/{splits}_splits/harmless_val.json'))
    harmful_val_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmful_val])
    harmless_val_instructions = apply_chat_template(model.tokenizer, [d["instruction"] for d in harmless_val])
    harmful_val_scores = get_bypass_scores(model, harmful_val_instructions, refusal_tokens, batch_size=args.filter_batch_size)
    harmless_val_scores = get_bypass_scores(model, harmless_val_instructions, refusal_tokens, batch_size=args.filter_batch_size)
    filtered_harmful_val_instructions = [d for d, score in zip(harmful_val_instructions, harmful_val_scores) if score > 0]
    filtered_harmless_val_instructions = [d for d, score in zip(harmless_val_instructions, harmless_val_scores) if score < 0]
    harmful_val_instructions = filtered_harmful_val_instructions
    harmless_val_instructions = filtered_harmless_val_instructions[:len(filtered_harmful_val_instructions)]

    layer_cutoff = 0.9
    repind_layers = list(range(int(model.config.num_hidden_layers * layer_cutoff)))

    group_name = f"repind_iterative"

    best_vectors = []
    independent_vectors = [best_refusal_direction]
    n_idx = 1
    inits = 1
    for idx in range(1, n_idx + 1):
        mean_refusal_scores = []
        lowest_loss_vectors = []

        for i in range(1, inits + 1):
            run_name = f"repind_{idx}_run_{i}"
            results = train_independent_vector(group_name=group_name, run_name=run_name, independent_vectors=independent_vectors, repind_layers=repind_layers)

            lowest_loss_vector = results['lowest_loss_vector']
            lowest_loss_vectors.append(lowest_loss_vector)
            refusal_scores = get_bypass_scores(model, harmful_val_instructions, refusal_tokens, fn_vector=lowest_loss_vector, batch_size=args.filter_batch_size)
            mean_refusal_score = refusal_scores.mean().item()
            mean_refusal_scores.append(mean_refusal_score)
            print(f"mean_refusal_score: {mean_refusal_score}")

        best_idx = np.argmin(mean_refusal_scores)
        print(f"best_idx: {best_idx}, i.e. run name 'rep_ind{idx}_run_{best_idx+1}'")
        independent_vectors.append(lowest_loss_vectors[best_idx])
    print(mean_refusal_scores)
# %%
