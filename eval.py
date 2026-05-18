"""
Evaluate a trained model on prompt-response pairs.

Accepted data file formats:
- JSONL: {"prompt": "...", "response": "..."}
- TSV: prompt<TAB>response
- Delimited text: prompt|||response
"""
import json
import os
import pickle
from contextlib import nullcontext

import torch
import tiktoken
from torch.nn import functional as F

from model1_3 import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # kept for argument compatibility with sample1_3.py
num_samples = 10 # kept for argument compatibility with sample1_3.py
max_new_tokens = 128 # kept for argument compatibility with sample1_3.py
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
data_file = 'eval_data.json' # file containing prompt-response pairs
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------


def load_data(path):
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        prompt, response = item['prompt'], item['response']
        pairs.append((prompt, response))
    return pairs



def score_response(model, prompt_ids, fixed_response_ids, temperature=1.0, top_k=None, device='cpu'):
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
    fixed_response = torch.tensor(fixed_response_ids, dtype=torch.long, device=device)[None, ...]

    full_idx = model.generate(
        x,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        fixed_response=fixed_response,
    )

    if max_new_tokens == 0:
        return full_idx, 0.0

    log_prob = torch.zeros(1, device=device, dtype=torch.float64)
    response_start = len(prompt_ids)
    full_ids = full_idx[0].tolist()

    for target_pos in range(response_start, len(full_ids)):
        if target_pos == 0:
            raise ValueError("Cannot score the first response token without any prompt/context token.")

        context_start = max(0, target_pos - model.config.block_size)
        context_ids = full_ids[context_start:target_pos]
        context = torch.tensor(context_ids, dtype=torch.long, device=device)[None, ...]

        logits, _ = model(context)
        logits = logits[:, -1, :].double() / temperature

        # if top_k is not None:
        #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        #     logits[logits < v[:, [-1]]] = -float('Inf')

        target = torch.tensor([full_ids[target_pos]], dtype=torch.long, device=device)
        token_log_prob = F.log_softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
        log_prob += token_log_prob

    return full_idx, torch.exp(log_prob).item()




def eval(data_file):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        checkpoint = None
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    else:
        raise ValueError(f"Unknown init_from: {init_from}")

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)

    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={""})
        decode = lambda l: enc.decode(l)

    pairs = load_data(data_file)

    with torch.no_grad():
        with ctx:
            for i, (prompt, response) in enumerate(pairs, start=1):
                prompt_ids = encode(prompt)
                response_ids = encode(response)
                full_idx, summed_probability = score_response(
                    model,
                    prompt_ids,
                    response_ids,
                    temperature=temperature,
                    top_k=top_k,
                    device=device,
                )
                print(f"Prompt: {prompt}")
                print(f"Response: {decode(full_idx[0, len(prompt_ids):].tolist())}")
                # print(f"Model output: {decode(full_idx[0].tolist())}")
                print(f"Response Probability: {summed_probability:.8e}")
                print('---------------')


if __name__ == '__main__':
    eval(data_file)
