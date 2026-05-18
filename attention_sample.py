"""
Trace and visualize causal self-attention for a prompt.

Examples:
python attention_trace.py
python attention_trace.py --prompt="A puppy should eat" --layer=0 --head=0
python attention_trace.py --out_dir=out-dogs --prompt="A dog should eat" --device=cpu
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import tiktoken
from torch.nn import functional as F

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = "scratch" # "scratch", "resume", or "gpt2"
out_dir = "out-dogs"
prompt = "A dog should eat healthy food"
device = "cpu"
layer = 1
head = 1
output_file = "attention_weights.png"

# tiny scratch-model settings, used only when init_from == "scratch"
block_size = 32
vocab_size = 50257
n_layer = 2
n_head = 2
n_embd = 32
dropout = 0.0
bias = True
exec(open("configurator.py").read())
# -----------------------------------------------------------------------------


def load_model():  
    if init_from == "resume":
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith("gpt2"):
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    elif init_from == "scratch":
        gptconf = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
        )
        model = GPT(gptconf)
    else:
        raise ValueError(f"Unknown init_from: {init_from}")

    model.eval()
    model.to(device)
    return model


def clean_token(token):
    token = token.replace("\n", "\\n").replace("\t", "\\t")
    return token if token else "<empty>"


@torch.no_grad()
def trace_attention(model, idx):
    assert 0 <= layer < len(model.transformer.h), f"layer must be in [0, {len(model.transformer.h) - 1}]"
    first_attn = model.transformer.h[0].attn
    assert 0 <= head < first_attn.n_head, f"head must be in [0, {first_attn.n_head - 1}]"

    B, T = idx.size()
    C = model.config.n_embd
    hs = C // first_attn.n_head

    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)

    print("TRACE: causal self-attention")
    print(f"input token ids idx: {tuple(idx.shape)}")
    print(f"token embeddings tok_emb: {tuple(tok_emb.shape)}")
    print(f"position embeddings pos_emb: {tuple(pos_emb.shape)}")
    print()

    last_weights = None
    last_layer = len(model.transformer.h) - 1
    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=idx.device)).view(1, 1, T, T)

    for layer_idx, block in enumerate(model.transformer.h):
        attn = block.attn
        assert head < attn.n_head, f"head must be in [0, {attn.n_head - 1}] for layer {layer_idx}"

        ln_x = block.ln_1(x)
        qkv = attn.c_attn(ln_x)
        q, k, v = qkv.split(attn.n_embd, dim=2)
        q = q.view(B, T, attn.n_head, hs).transpose(1, 2)
        k = k.view(B, T, attn.n_head, hs).transpose(1, 2)
        v = v.view(B, T, attn.n_head, hs).transpose(1, 2)

        raw_scores = (q @ k.transpose(-2, -1)) * (1.0 / (hs ** 0.5))
        masked_scores = raw_scores.masked_fill(~causal_mask, float("-inf"))
        weights = F.softmax(masked_scores, dim=-1)
        context = weights @ v
        merged_context = context.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = attn.c_proj(merged_context)

        print(f"layer {layer_idx}:")
        print(f"  block input x: {tuple(x.shape)}")
        print(f"  after ln_1: {tuple(ln_x.shape)}")
        print(f"  combined qkv projection: {tuple(qkv.shape)}")
        print(f"  q: {tuple(q.shape)}")
        print(f"  k: {tuple(k.shape)}")
        print(f"  v: {tuple(v.shape)}")
        print(f"  raw attention scores q @ k.T / sqrt(head_size): {tuple(raw_scores.shape)}")
        print(f"  causal mask: {tuple(causal_mask.shape)}")
        print(f"  attention weights after softmax: {tuple(weights.shape)}")
        print(f"  context weights @ v: {tuple(context.shape)}")
        print(f"  merged heads: {tuple(merged_context.shape)}")
        print(f"  attention output projection: {tuple(attn_output.shape)}")
        print(f"  attention weight matrix for layer {layer_idx}, head {head}:")
        print(weights[0, head].cpu())
        print()

        last_weights = weights[0, head].cpu()
        x = x + attn_output
        x = x + block.mlp(block.ln_2(x))

    return last_weights, last_layer, head


def plot_attention(weights, tokens, layer, head):
    fig, ax = plt.subplots(figsize=(max(7, len(tokens) * 0.75), max(6, len(tokens) * 0.65)))
    im = ax.imshow(weights.numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"Causal attention weights, layer {layer}, head {head}")
    ax.set_xlabel("key token attended to")
    ax.set_ylabel("query token")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)

    for i in range(weights.size(0)):
        for j in range(weights.size(1)):
            if j <= i:
                ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)
    print(f"saved attention heatmap to {output_file}")


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    model = load_model()
    token_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    token_ids = token_ids[:model.config.block_size]
    idx = torch.tensor(token_ids, dtype=torch.long, device=device)[None, ...]
    tokens = [clean_token(enc.decode([token_id])) for token_id in token_ids]

    print(f"prompt: {prompt!r}")
    print(f"tokens: {tokens}")
    print()
    weights, traced_layer, traced_head = trace_attention(model, idx)
    plot_attention(weights, tokens, traced_layer, traced_head)
