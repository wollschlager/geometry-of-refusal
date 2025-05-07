from nnsight import LanguageModel
from tqdm import tqdm
import einops
from torch import Tensor
from jaxtyping import Float

def projection_einops(activation, direction):
    proj = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction
    )
    return proj

def generate_completions(
    model: LanguageModel,
    dataset: list[str],
    max_new_tokens: int = 150,
    batch_size: int = 16,
):
    all_completions = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        instructions = dataset[i:i+batch_size]
        start_token = len(model.tokenizer(instructions, add_special_tokens=True, padding=True, truncation=False)["input_ids"][0])
        
        with model.generate(instructions, max_new_tokens=max_new_tokens, do_sample=False) as generator:
            tokens = model.generator.output.save()

        completion = model.tokenizer.batch_decode(tokens.value[:, start_token:], skip_special_tokens=True)
        all_completions.extend(completion)
        
    return all_completions

def intervene_with_fn_vector_ablation(
    model: LanguageModel,
    dataset: list[str],
    fn_vector: Float[Tensor, "d_model"],
    max_new_tokens: int = 150,
    batch_size: int = 16,
    do_sample=False,
    temperature=0,
):
    fn_vector = fn_vector / fn_vector.norm()
    all_completions = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        instructions = dataset[i:i+batch_size]
        start_token = len(model.tokenizer(instructions, add_special_tokens=True, padding=True, truncation=False)["input_ids"][0])
        
        with model.generate(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature) as generator:
            with generator.invoke(instructions) as invoker:
                tokens_intervention = model.generator.output.save()
                for n in range(max_new_tokens - 1):
                    for layer in model.model.layers:
                        layer.input[:] -= projection_einops(layer.input[:], fn_vector)
                        layer.self_attn.output[0][:] -= projection_einops(layer.self_attn.output[0][:], fn_vector)
                        layer.mlp.output[:] -= projection_einops(layer.mlp.output[:], fn_vector)
                    generator.next()

        completion = model.tokenizer.batch_decode(tokens_intervention.value[:, start_token:], skip_special_tokens=True)
        all_completions.extend(completion)
        
    return all_completions


def intervene_with_fn_vector_addition(
    model: LanguageModel,
    dataset: list[str],
    layer: int,
    alpha: float,
    fn_vector: Float[Tensor, "d_model"],
    max_new_tokens: int = 150,
    post_tokens: int = -1,
    batch_size: int = 16,
):
    fn_vector = alpha * fn_vector / fn_vector.norm()
    all_completions = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        instructions = dataset[i:i+batch_size]
        start_token = len(model.tokenizer(instructions, add_special_tokens=True, padding=True, truncation=False)["input_ids"][0])
        
        with model.generate(max_new_tokens=max_new_tokens, do_sample=False) as generator:
            with generator.invoke(instructions) as invoker:
                tokens_intervention = model.generator.output.save()
                for n in range(max_new_tokens - 1):
                    if post_tokens == -1 or n <= post_tokens:
                        model.model.layers[layer].input += fn_vector
                    generator.next()
        completion = model.tokenizer.batch_decode(tokens_intervention.value[:, start_token:], skip_special_tokens=True)
        all_completions.extend(completion)
        
    return all_completions