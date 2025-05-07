import os
import json

dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmless', 'harmful']

# SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, '../../data/saladbench_splits/{harmtype}_{split}.json')
SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, '../../data/saladbench_splits/{harmtype}_{split}.json')

PROCESSED_DATASET_NAMES = ["advbench", "tdc2023", "maliciousinstruct", "harmbench_val", "harmbench_test", "jailbreakbench", "strongreject", "alpaca", "xstest", "sorrybench"]

def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False):
    assert harmtype in HARMTYPES
    assert split in SPLITS

    file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset

def load_strongreject():
    from strong_reject.load_datasets import load_strongreject
    forbidden_prompt_dataset = load_strongreject()
    forbidden_prompts = forbidden_prompt_dataset["forbidden_prompt"]
    categories = forbidden_prompt_dataset["category"]
    return [{"instruction": prompt, "category": category} for prompt, category in zip(forbidden_prompts, categories)]

def load_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in PROCESSED_DATASET_NAMES, f"Valid datasets: {PROCESSED_DATASET_NAMES}"

    if dataset_name == "strongreject":
        return load_strongreject()

    file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]
 
    return dataset
