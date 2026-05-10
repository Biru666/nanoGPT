"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import matplotlib.pyplot as plt
import numpy as np
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
show_probs = False # whether to show probability distribution charts for generated tokens
num_probs_to_show = 10 # number of initial tokens to visualize (for speed optimization)
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
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
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

def draw_chart(top_probs, top_indices, selected_token_idx, decode_fn, sample_idx=0, token_idx=0):
    """
    Draw a bar chart of top 10 token probabilities.
    Highlights the selected token in red, others in blue.
    
    Args:
        top_probs: top 10 probabilities
        top_indices: top 10 token indices
        selected_token_idx: the token index that was selected
        decode_fn: function to decode token indices to strings
        sample_idx: sample number (for filename)
        token_idx: token position (for filename)
    """
    top_tokens = [decode_fn([idx]) for idx in top_indices]
    
    # Color the selected token differently if it's in top 10
    colors = ['red' if idx == selected_token_idx else 'steelblue' for idx in top_indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(top_tokens)), top_probs, color=colors, alpha=0.8)
    
    ax.set_xlabel('Token', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'Top 10 Token Probabilities - Sample {sample_idx+1}, Token {token_idx+1}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(top_tokens)))
    ax.set_xticklabels(top_tokens, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, prob in zip(bars, top_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{prob:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save the chart
    filename = f'probs_sample{sample_idx+1}_token{token_idx+1}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Chart saved as '{filename}'")
    
    plt.show()
    plt.close()

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            if show_probs:
                # Hybrid approach: visualize first num_probs_to_show tokens, then fast generation
                x_gen = x.clone()
                all_selected_tokens = []
                
                for token_idx in range(max_new_tokens):
                    # Forward pass
                    logits, _ = model(x_gen)
                    logits = logits[0, -1, :] / temperature
                    
                    # Convert to probabilities
                    probs = torch.softmax(logits, dim=-1)

                    # Apply top_k filtering
                    if top_k is not None:
                        v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                        probs[probs < v[-1]] = 0
                        probs = probs / probs.sum()
                    
                    # Get top 10 tokens using torch.topk
                    top_probs, top_indices = torch.topk(probs, k=10)
                    
                    # Sample next token using torch.multinomial
                    selected_token_idx = torch.multinomial(probs, num_samples=1).item()
                    all_selected_tokens.append(selected_token_idx)
                    
                    # Draw chart for the current token
                    draw_chart(top_probs.cpu().numpy(), top_indices.cpu().numpy(), selected_token_idx, decode, k, token_idx)
                    
                    # Append to sequence
                    x_gen = torch.cat([x_gen, torch.tensor([[selected_token_idx]], device=device)], dim=1)
                
                # Fast generation for remaining tokens
                if max_new_tokens > num_probs_to_show:
                    remaining_tokens = max_new_tokens - num_probs_to_show
                    y_remaining, y_remaining_probs = model.generate(x_gen, remaining_tokens, temperature=temperature, top_k=top_k)
                    all_selected_tokens.extend(y_remaining[0, x_gen.shape[1]:].tolist())
                
                print(decode(all_selected_tokens))
            else:
                y, y_probs = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print(f"Response probability: {y_probs.item():.12e}")
                print('---------------')
