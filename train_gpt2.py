from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask, or 'bias' following the OpenAI/HF naming
        self.register_buffer("bias", torch.tril(torch.ones(config.context_size, config.context_size))
                             .view(1, 1, config.context_size, config.context_size))
    
    def forward(self, x):
        B, T, C = x.size() # B: batch size, T: sequence length, C: embedding size
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # in GPT-2 (124M), n_head = 12, hs = 64, n_head * hs = n_embd = 768
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, hs)
        # attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, n_head, T, hs) x (B, n_head, hs, T) -> (B, n_head, T, T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, n_head, T, T) x (B, n_head, T, hs) -> (B, n_head, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, n_head, T, hs) -> (B, T, n_head * hs)
        # output projection
        y = self.c_proj(y)
        return y
        

class MLP(nn.Module):

    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPT2Config:
    context_size: int = 1024 # context size
    vocab_size: int = 50257 # size of vocabulary: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2, self).__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.context_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx, targets=None):
        B, T = idx.size() # B: batch size, T: sequence length
        assert T <= self.config.context_size, "Cannot forward sequence of length %d, context size is only %d" % (T, self.config.context_size)
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T,)
        pos_emb = self.transformer.wpe(pos) # positional embedding, (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embedding, (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Load a pre-trained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, "Invalid model type"
        from transformers import GPT2LMHeadModel
        print(f"Loading pre-trained model: {model_type}")

        # n_layer, n_head and N-embd are set according to the model type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50,257 for GPT model checkpoints
        config_args['context_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard mask

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all params re aligned and match innames and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # discard mask
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # discard mask
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # do necessary transpositions
        assert len(sd_keys_hf) == len(sd_keys), f"Mismatch in number of keys: {len(sd_keys_hf)} vs {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        # load the state dict
        model.load_state_dict(sd)
        print(f"Loaded pre-trained model: {model_type} with {sum(p.numel() for p in model.parameters())} parameters")
        return model

# ----------------------------------------
batch_size = 5
max_length = 30

model = GPT2.from_pretrained('gpt2')
model.eval()
model.to('cpu')
print("model loaded successfully!")

# prefill tokens
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(batch_size, 1) # (5, 8)
x = tokens.to('cpu')

# generate output. x is (B, T) where B = 5, T = 8
torch.manual_seed(42)
# This process calculate the entire T for each step, since we don't cache any KV.
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # get the last token logits
        logits_last = logits[:, -1, :] # (B, vocab_size)
        # get the next token probabilities
        probs = F.softmax(logits_last, dim=-1) # (B, vocab_size)
        # use top-k sampling of 50 (huggingface default)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1) # (B, 50)
        # sample from the top-k probabilities
        idx = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
        next_token = torch.gather(topk_indices, -1, idx) # (B, 1)
        x = torch.cat((x, next_token), dim=1) # (B, T+1)

# decode the output tokens
for i in range(batch_size):
    output_tokens = x[i, :max_length].tolist()
    decoded = enc.decode(output_tokens)
    print(f"Output {i+1}: {decoded}")