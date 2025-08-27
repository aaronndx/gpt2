from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import inspect

# ----------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT2_SCALE_INIT = 1
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
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, n_head, T, hs) x (B, n_head, hs, T) -> (B, n_head, T, T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, n_head, T, T) x (B, n_head, T, hs) -> (B, n_head, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Flash Attention

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
        self.c_proj.GPT2_SCALE_INIT = 1
    
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

        # sharing token <-> embedding weights
        self.transformer.wte.weight = self.lm_head.weight

        # initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT2_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # apply 1 / sqrt(num_layer) to stablize residual connections (additions)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # layer norm is already initialized by default in PyTorch as scale = 1.0, offset = 0.0
    
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
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

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

    def configure_optimizers(self, device, weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95), eps=1e-8):
        # Starts with all candidate parameters that requires gradients
        params_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # Create optim groups, and all 2D weights will be weight decayed.
        # i.e. all weights in matmuls, embeddings have decay, but biases and layer norms do not.
        decay_params = [p for pn, p in params_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn, p in params_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Number of decayed tensors: {len(decay_params)} with {num_decay_params:,} parameters")
        print(f"Number of non-decayed tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        extra_args = dict(fused=True) if use_fused else {}
        print(f"Using {'fused' if use_fused else 'non-fused'} AdamW optimizer")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, **extra_args)
        return optimizer


class DataLoaderLite:
    def __init__(self, file, B, T):
        self.B = B # batch size
        self.T = T # sequence length
        self.file = file

        with open(file, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens from {file}")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) # Input tokens
        y = buf[1:].view(B, T) # Labels
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0 # reset for next epoch
        return x, y

def simple_eval(device, input=None, model=None, batch_size=5, max_length=30):
    if input is None:
        input = "Hello, I'm a language model,"

    if model is None:
        model = GPT2.from_pretrained('gpt2')
    model.eval()
    model.to(device)
    print("model loaded successfully!")
    # prefill tokens
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(input)
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(batch_size, 1) # (5, 8)
    x = tokens.to(device)

    set_seed(1337)

    # generate output. x is (B, T) where B = 5, T = 8
    # This process calculate the entire T for each step, since we don't cache any KV.
    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits, _ = model(x) # (B, T, vocab_size)
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

def auto_pick_device():
    """Auto-detect the device to use for training."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def synchronize(device):
    """Synchronize the device to ensure all operations are complete."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    else:
        pass  # No synchronization needed for CPU

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_training_precision(device):
    """
    Gets the appropriate training precision strategy according to device.
    Returns:
        pair: (bool, torch.dtype) for AMP flag, and AMP dtype.
    """
    use_amp = False
    amp_dtype = torch.float32

    if device == "cuda" and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("CUDA device is Ampere or newer. Using TF32.")
            torch.set_float32_matmul_precision('high')
        else:
            print("CUDA device is older. Using AMP with FP16.")
            use_amp = True
            amp_dtype = torch.float16
    elif device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Apple MPS device found. Using AMP with FP16.")
        use_amp = True
        amp_dtype = torch.float16
    else:
        print("No GPU found. Running on CPU with AMP using BFloat16.")
        use_amp = True
        amp_dtype = torch.bfloat16
        
    return use_amp, amp_dtype

def simple_train(device, data=None, steps=50, B=4, T=32):
    if data is None:
        data = "input.txt"
    train_loader = DataLoaderLite(data, B, T)

    # get logits
    model = GPT2(GPT2Config())
    model.to(device)

    # optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(steps):
        x, y = train_loader.next_batch() # get next batch
        x, y = x.to(device), y.to(device) # move to device
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"Step {i+1}/{steps}, Loss: {loss.item()}")
    return model

def efficient_train(device, data=None, B=16, T=1024, steps=50, total_batch_size=0):
    set_seed(1337)

    import time
    if data is None:
        data = "input.txt"
    train_loader = DataLoaderLite(data, B, T)

    use_amp, amp_dtype = get_training_precision(device)
    if use_amp:
        print(f"Using AMP with dtype: {amp_dtype}")
    
    # --- Initialize GradScaler based on the configuration ---
    scaler = torch.amp.GradScaler(enabled=(use_amp and device == 'cuda'))

    # get logits
    model = GPT2(GPT2Config(vocab_size=50304)) # Use nice number with power of 2 (128)
    model.to(device)
    # compile the model for better performance with optimized python code and kernel fusion
    model = torch.compile(model)

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    def get_lr(step):
        """Calculate learning rate based on step."""
        # 1. linear warmup
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        # 2. min rate for above max steps
        elif step > steps:
            return min_lr
        # 3. cosine decay
        else:
            decay_ratio = (step - warmup_steps) / (steps - warmup_steps)
            assert 0 <= decay_ratio <= 1, "Decay ratio must be in [0, 1]"
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # cosine decay coefficient, 1 -> 0
            # return the learning rate based on cosine decay
            return min_lr + coeff * (max_lr - min_lr)
    
    gradient_accum_steps = 1
    if total_batch_size > 0:
        assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by batch_size * max_length"
        gradient_accum_steps = total_batch_size // (B * T)
        print(f"total desired batch size: {total_batch_size}")
        print(f"-> calculated gradient accumulation iterations: {gradient_accum_steps}")

    # optimization
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    optimizer = model.configure_optimizers(device=device, learning_rate=max_lr, betas=(0.9, 0.95), eps=1e-8)
    for step in range(steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(gradient_accum_steps):
            x, y = train_loader.next_batch() # get next batch
            x, y = x.to(device), y.to(device) # move to device
            # forward pass with automatic mixed precision (AMP) if enabled
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            loss = loss / gradient_accum_steps # normalize the loss for mean-loss calculation
            loss_accum += loss.detach()
            scaler.scale(loss).backward()  # scale the loss for AMP
        scaler.unscale_(optimizer)  # unscale the gradients for clipping
        # gradient clipping to avoid model shock by too big gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # detemine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        scaler.step(optimizer)
        scaler.update()  # update the scaler

        # print timing and performance metrics
        synchronize(device)  # ensure all accelerator operations are complete before timing
        t1 = time.time()
        dt = (t1 - t0) * 1000  # convert to milliseconds
        tokens_processed = train_loader.B * train_loader.T * gradient_accum_steps
        tokens_per_sec = tokens_processed / (t1 - t0)  # tokens per second
        print(f"Step {step:4d}, Loss: {loss_accum.item()}, lr: {lr:.4e}, Norm: {norm:.4f}, Time: {dt:.2f} ms, Tokens/sec: {tokens_per_sec:.2f}")
    return model

if __name__ == "__main__":
    # ----------------------------------------
    # auto detect device
    device = auto_pick_device()
    # device = "cpu" # override to cpu until cuda is available. MPS does not work well with pytorch, especially for training.
    print(f"Using device: {device}")

    model = efficient_train(device, data="input.txt", steps=50, B=4, T=256, total_batch_size=4*256*2) # total_batch_size = 2**19, ~0.5M tokens for GPT3 training
    simple_eval(device, max_length=100, input="They feast on wine and swan while our own tables see naught but the shadow of a crust", model=model)