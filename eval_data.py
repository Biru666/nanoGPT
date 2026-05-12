"""
Create prompt-response eval pairs from a Hugging Face training split.

Writes a JSON file that eval.py can read:
[
  {"prompt": "...", "response": "..."},
  ...
]
"""
import json
import random

import tiktoken
from datasets import load_dataset

# -----------------------------------------------------------------------------
dataset_name = "SparkleDark/Everything_about_dogs"
dataset_split = "train"
text_column = "text" 
output_file = "eval_data.json"
num_pairs = 20
prompt_tokens = 10
response_tokens = 10
seed = 1337
num_proc_load_dataset = 8
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------


def get_text(example, column):
    value = example[column]
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def move_to_token_boundary(ids, start, max_start, decode):
    while start < max_start:
        token_text = decode([ids[start]])
        if token_text.startswith((" ", "\n", "\t")) or token_text[:1].isalnum():
            return start
        start += 1
    return start


def make_eval_pairs(dataset, encode, decode):
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    pairs = []
    needed_tokens = prompt_tokens + response_tokens
    column = text_column

    for idx in indices:
        text = get_text(dataset[idx], column).strip()
        if not text:
            continue

        ids = encode(text)
        if len(ids) < needed_tokens:
            continue

        max_start = len(ids) - needed_tokens
        rand_start = rng.randint(0, max_start) if max_start > 0 else 0
        start = move_to_token_boundary(ids, rand_start, max_start, decode)
        prompt_ids = ids[start:start + prompt_tokens]
        response_ids = ids[start + prompt_tokens:start + needed_tokens]

        pairs.append({
            "prompt": decode(prompt_ids),
            "response": decode(response_ids),
        })

        if len(pairs) >= num_pairs:
            break

    if len(pairs) < num_pairs:
        raise ValueError(
            f"Only created {len(pairs)} pairs. Try lowering prompt_tokens/response_tokens "
            f"or use a larger dataset split."
        )

    return pairs


if __name__ == "__main__":
    dataset = load_dataset(
        dataset_name,
        split=dataset_split,
        num_proc=num_proc_load_dataset,
    )

    enc = tiktoken.get_encoding("gpt2")
    pairs = make_eval_pairs(dataset, enc.encode_ordinary, enc.decode)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {len(pairs)} prompt-response pairs to {output_file}")
