# %%
import sys
sys.path.append('..')

# %%
import os
import json
import pandas as pd
import random
import requests

from datasets import load_dataset
# %%
def download_file(url, file_path):
    response = requests.get(url)
    response.raise_for_status()

    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(file_path, "wb") as file:
        file.write(response.content)

def dump_json(data, file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

# %%
current_dir = os.path.abspath("")
raw_data_dir = os.path.join(current_dir, 'raw')
processed_data_dir = os.path.join(current_dir, 'processed')

def download_advbench():
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    raw_file_path = os.path.join(raw_data_dir, 'advbench.csv')
    processed_file_path = os.path.join(processed_data_dir, 'advbench.json')

    download_file(url, raw_file_path)
    dataset = pd.read_csv(raw_file_path)

    instructions = dataset['goal'].to_list()
    dataset_json = [{'instruction': instruction.strip(), 'category': None} for instruction in instructions]
    dump_json(dataset_json, processed_file_path)

def download_malicious_instruct():
    url = 'https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt'
    raw_file_path = os.path.join(raw_data_dir, 'malicious_instruct.txt')
    processed_file_path = os.path.join(processed_data_dir, 'malicious_instruct.json')

    download_file(url, raw_file_path)
    with open(raw_file_path, 'r') as f:
        instructions = f.readlines()

    dataset_json = [{'instruction': instruction.strip(), 'category': None} for instruction in instructions]
    dump_json(dataset_json, processed_file_path)

def download_tdc2023():
    urls = [
        'https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/dev/behaviors.json',
        'https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/test/behaviors.json'
    ]
    raw_file_paths = [
        os.path.join(raw_data_dir, 'tdc2023_dev_behaviors.json'),
        os.path.join(raw_data_dir, 'tdc2023_test_behaviors.json')
    ]
    processed_file_path = os.path.join(processed_data_dir, 'tdc2023.json')

    for url, raw_file_path in zip(urls, raw_file_paths):
        download_file(url, raw_file_path)

    instructions = []
    for raw_file_path in raw_file_paths:
        with open(raw_file_path, 'r') as f:
            dataset = json.load(f)
        instructions.extend(dataset)

    dataset_json = [{'instruction': instruction.strip(), 'category': None} for instruction in instructions]
    dump_json(dataset_json, processed_file_path)

def download_jailbreakbench():
    url = 'https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/main/src/jailbreakbench/data/behaviors.csv'
    raw_file_path = os.path.join(raw_data_dir, 'jailbreakbench.csv')
    processed_file_path = os.path.join(processed_data_dir, 'jailbreakbench.json')

    download_file(url, raw_file_path)
    dataset = pd.read_csv(raw_file_path)

    instructions = dataset['Goal'].to_list()
    categories = dataset['Category'].to_list()

    dataset_json = [{'instruction': instruction.strip(), 'category': category} for instruction, category in zip(instructions, categories)]
    dump_json(dataset_json, processed_file_path)

def download_harmbench(split):
    assert split in ['val', 'test']

    url = f'https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_{split}.csv'
    raw_file_path = os.path.join(raw_data_dir, f'harmbench_{split}.csv')
    processed_file_path = os.path.join(processed_data_dir, f'harmbench_{split}.json')

    download_file(url, raw_file_path)
    dataset = pd.read_csv(raw_file_path)

    instructions = []
    categories = []
    # filter out instructions with Category=copyright or Tags=context
    filtered_dataset = dataset[
        ~dataset['FunctionalCategory'].str.contains('copyright', case=False, na=False) &
        ~dataset['Tags'].str.contains('context', case=False, na=False)
    ]

    instructions.extend(filtered_dataset['Behavior'].to_list())
    categories.extend(filtered_dataset['SemanticCategory'].to_list())

    dataset_json = [{'instruction': instruction.strip(), 'category': category} for instruction, category in zip(instructions, categories)]
    dump_json(dataset_json, processed_file_path)

def download_strongreject():
    url = 'https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv'
    raw_file_path = os.path.join(raw_data_dir, 'strongreject.csv')
    processed_file_path = os.path.join(processed_data_dir, 'strongreject.json')

    download_file(url, raw_file_path)
    dataset = pd.read_csv(raw_file_path)

    instructions = dataset['forbidden_prompt'].to_list()
    categories = dataset['category'].to_list()

    dataset_json = [{'instruction': instruction.strip(), 'category': category} for instruction, category in zip(instructions, categories)]
    dump_json(dataset_json, processed_file_path)

def download_alert():
    import jsonlines
    url = 'https://raw.githubusercontent.com/Babelscape/ALERT/refs/heads/master/data/alert.jsonl'
    raw_file_path = os.path.join(raw_data_dir, 'alert.jsonl')
    processed_file_path = os.path.join(processed_data_dir, 'alert.json')

    download_file(url, raw_file_path)
    data = []
    with open(raw_file_path, "r") as json_file:
        for line in json_file:
            data.append(json.loads(line))
    
    start = "### Instruction:\n"
    end = "\n### Response:"
    dataset_json = [{'instruction': d["prompt"].strip().replace(start, "").replace(end, ""), 'category': d["category"]} for d in data]
    dump_json(dataset_json, processed_file_path)

def download_alpaca():
    hf_path = 'tatsu-lab/alpaca'
    processed_file_path = os.path.join(processed_data_dir, 'alpaca.json')

    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])

    dataset_json = [{'instruction': instruction.strip(), 'category': None} for instruction in instructions]
    dump_json(dataset_json, processed_file_path)

def download_circuit_breakers():
    url = "https://raw.githubusercontent.com/GraySwanAI/circuit-breakers/refs/heads/main/data/circuit_breakers_train.json"
    raw_file_path = os.path.join(raw_data_dir, 'circuit_breakers.json')
    processed_file_path = os.path.join(processed_data_dir, 'circuit_breakers.json')

    download_file(url, raw_file_path)
    dataset = json.load(open(raw_file_path))
    dataset_json = [{'instruction': d["prompt"].strip(), 'target': d["output"].strip(), 'category': d["category"]} for d in dataset]
    dump_json(dataset_json, processed_file_path)

def download_salad_bench():
    from datasets import load_dataset
    dataset = load_dataset("OpenSafetyLab/Salad-Data", name='base_set', split='train')
    raw_file_path = os.path.join(raw_data_dir, 'salad_bench.json')
    processed_file_path = os.path.join(processed_data_dir, 'salad_bench.json')

    # save dataset to json using huggingface function
    dataset.to_json(raw_file_path)

    # remove source "Multilingual" and "ToxicChat"
    dataset = dataset.filter(lambda x: x['source'] not in ["Multilingual", "ToxicChat"])
    max_examples_per_source = 256
    # reduce each source to max 256 examples
    source_groups = dataset.to_pandas().groupby('source')
    filtered_data = []
    for source, group in source_groups:
        if len(group) > max_examples_per_source:
            filtered_data.append(group.sample(n=max_examples_per_source))
        else:
            filtered_data.append(group)
    dataset = pd.concat(filtered_data)

    print(len(dataset))
    # Convert DataFrame back to a list of dictionaries
    dataset_json = [{'instruction': row["question"].strip(), 'source': row["source"]} for index, row in dataset.iterrows()]
    dump_json(dataset_json, processed_file_path)

def download_xstest():
    url = 'https://raw.githubusercontent.com/paul-rottger/xstest/refs/heads/main/xstest_prompts.csv'
    raw_file_path = os.path.join(raw_data_dir, 'xstest.csv')
    processed_file_path = os.path.join(processed_data_dir, 'xstest.json')

    download_file(url, raw_file_path)
    dataset = pd.read_csv(raw_file_path)
    # Filter to only include examples where label is safe
    dataset = dataset[dataset["label"] == "safe"]
    dataset_json = [{'instruction': row["prompt"].strip(), 'category': None} for index, row in dataset.iterrows()]
    print(len(dataset_json))
    dump_json(dataset_json, processed_file_path)

def download_sorrybench():
    hf_path = 'sorry-bench/sorry-bench-202503'
    processed_file_path = os.path.join(processed_data_dir, 'sorrybench.json')

    dataset = load_dataset(hf_path)
    
    # Check if any example has more than 1 turn
    has_multiple_turns = any(len(d["turns"]) > 1 for d in dataset['train'])
    if has_multiple_turns:
        print("Warning: Some examples in SorryBench have more than 1 turn. Only using the first turn.")
    
    # Filter to only include examples with prompt_style=base
    dataset_json = [{'instruction': d["turns"][0], 'category': d["category"]} 
                    for d in dataset['train'] 
                    if d["turns"][0] is not None and d["prompt_style"] == "base"]
    print(len(dataset_json))
    dump_json(dataset_json, processed_file_path)

def download_orbench_hard():
    hf_path = "bench-llm/or-bench"
    processed_file_path = os.path.join(processed_data_dir, 'orbench_hard.json')

    dataset = load_dataset(hf_path, name="or-bench-hard-1k")
    dataset_json = [{'instruction': d["prompt"].strip(), 'category': d["category"]} for d in dataset['train']]
    dump_json(dataset_json, processed_file_path)

# %%
# download_advbench()
# download_malicious_instruct()
# download_tdc2023()

download_jailbreakbench()
download_harmbench(split='val')
download_harmbench(split='test')
download_strongreject()
download_sorrybench()

download_alpaca()

# download_circuit_breakers()

download_salad_bench()
download_xstest()
# download_orbench_hard()
# %%
current_dir = os.path.abspath("")
splits_data_dir = os.path.join(current_dir, 'saladbench_splits')

max_train_subset_size = 128 # limits the number of examples from a single dataset

def construct_harmful_dataset_splits():
    harmful_train_path = os.path.join(splits_data_dir, 'harmful_train.json')
    harmful_val_path = os.path.join(splits_data_dir, 'harmful_val.json')
    harmful_test_path = os.path.join(splits_data_dir, 'harmful_test.json')

    harmful_instructions = []
    for file in ['salad_bench.json']:
        with open(os.path.join(processed_data_dir, file), 'r') as f:
            harmful_instructions.extend(json.load(f))
    
    harmful_test_instructions = []
    for file in ['jailbreakbench.json', 'harmbench_test.json', 'strongreject.json']:
        with open(os.path.join(processed_data_dir, file), 'r') as f:
            harmful_test_instructions.extend(json.load(f))

    # now ensure that there are no duplicates across datasets
    filtered_out = 0
    filtered_harmful_instructions = []
    for instruction in harmful_instructions:
        if instruction['instruction'] not in [x['instruction'] for x in harmful_test_instructions]:
            filtered_harmful_instructions.append(instruction)
        else:
            filtered_out += 1

    print(f"filtered out {filtered_out} duplicate instructions from dataset")
    harmful_instructions = filtered_harmful_instructions

    # Now sample from filtered train set for train and val splits
    harmful_train_instructions = []
    harmful_val_instructions = []
    random.seed(42)
    random.shuffle(harmful_instructions)
    harmful_train_instructions = harmful_instructions[:1024]  # Sample 1024 for train
    harmful_val_instructions = harmful_instructions[1024:1024+128]  # Sample 128 for val

    # print length of the train, val and test sets
    print(f"train set length: {len(harmful_train_instructions)}")
    print(f"val set length: {len(harmful_val_instructions)}")
    print(f"test set length: {len(harmful_test_instructions)}")

    dump_json(harmful_train_instructions, harmful_train_path)
    dump_json(harmful_val_instructions, harmful_val_path)
    dump_json(harmful_test_instructions, harmful_test_path)

def construct_harmless_dataset_splits():
    harmless_train_path = os.path.join(splits_data_dir, 'harmless_train.json')
    harmless_val_path = os.path.join(splits_data_dir, 'harmless_val.json')
    harmless_test_path = os.path.join(splits_data_dir, 'harmless_test.json')

    train_p, val_p, test_p = 0.8, 0.10, 0.10

    harmless_instructions = []
    for file in ['alpaca.json']:
        with open(os.path.join(processed_data_dir, file), 'r') as f:
            harmless_instructions.extend(json.load(f))

    random.seed(42)
    random.shuffle(harmless_instructions)

    total_size = len(harmless_instructions)
    train_size = int(train_p * total_size)
    val_size = int(val_p * total_size)

    harmless_train_instructions = harmless_instructions[:train_size]
    harmless_val_instructions = harmless_instructions[train_size:train_size+val_size]
    harmless_test_instructions = harmless_instructions[train_size+val_size:]

    dump_json(harmless_train_instructions, harmless_train_path)
    dump_json(harmless_val_instructions, harmless_val_path)
    dump_json(harmless_test_instructions, harmless_test_path)

# %%
construct_harmful_dataset_splits()
construct_harmless_dataset_splits()