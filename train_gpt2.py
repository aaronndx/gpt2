from dataclasses import dataclass
import math
import random
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
import tiktoken
import inspect
import numpy as np
import os
import tempfile
from hellaswag import HellaSwagEval
from data_loader_lite import DataLoaderDisk, DataLoaderHuggingFace

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
        # self.register_buffer("bias", torch.tril(torch.ones(config.context_size, config.context_size))
        #                      .view(1, 1, config.context_size, config.context_size))
    
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

def tag_from_config(config: GPT2Config):
    n_params = {
        # n_params from d_model(n_embd)
        768: '125M',
        1024: '350M',
        1536: '760M',
        2048: '1300M'
    }[config.n_embd]
    return f"{n_params}_ctx{config.context_size}"

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
    """
    Sets the seed for all major sources of randomness to ensure reproducibility.
    This includes Python's random module, NumPy, and PyTorch for CPU, CUDA, and MPS.
    Note that this sets rng to starting point, so should not be called if restoring rng state is needed.
    
    Args:
        seed (int): The integer value to use as the seed.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch on CPU
    torch.manual_seed(seed)
    
    # Set the seed for PyTorch on CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        # The following two lines are often used to ensure deterministic behavior
        # but can impact performance. Use them if you need strict reproducibility.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        
    # Set the seed for PyTorch on MPS (Apple Silicon GPU, if available)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def save_rng_state(device_type):
    """
    Saves the current state of all random number generators.
    
    Args:
        device_type (str): The device type ('cuda', 'cpu', 'mps').
    
    Returns:
        dict: A dictionary containing the RNG states.
    """
    rng_state = {
        'py_rng_state': random.getstate(),
        'np_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
    }
    if device_type == 'cuda':
        rng_state['cuda_rng_state'] = torch.cuda.get_rng_state()
    # Note: torch.mps.get_rng_state() is not yet implemented as of PyTorch 2.x
    return rng_state

def restore_rng_state(rng_state, device_type):
    """
    Restores the state of all random number generators from a saved state.
    
    Args:
        rng_state (dict): A dictionary containing the RNG states.
        device_type (str): The device type ('cuda', 'cpu', 'mps').
    """
    random.setstate(rng_state['py_rng_state'])
    np.random.set_state(rng_state['np_rng_state'])
    torch.set_rng_state(rng_state['torch_rng_state'])
    if device_type == 'cuda' and 'cuda_rng_state' in rng_state:
        torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
    # Note: torch.mps.set_rng_state() is not yet implemented

def restore_checkpoint(filename, model, optimizer, scaler, device, repo_id=None, ckpt_dir=None, rank=0, master_process=True, strip_ddp_prefix=False):
    """
    Loads a training checkpoint from either the Hugging Face Hub or a local directory.

    Args:
        model, optimizer, scaler, device: Standard training objects.
        filename (str): The name of the checkpoint file.
        repo_id (str, optional): The ID of the Hugging Face repo. Defaults to None.
        ckpt_dir (str, optional): The path to a local directory. Defaults to None.
    """
    from huggingface_hub import hf_hub_download

    if not repo_id and not ckpt_dir:
        raise ValueError("Must provide either a repo_id or a ckpt_dir.")

    is_ddp = dist.is_initialized()

    if repo_id:
        if master_process:
            print(f"Downloading checkpoint '{filename}' from '{repo_id}'...")
        ckpt_path_obj = [None] # Use a list to broadcast
        # Download the checkpoint file from the Hub, which caches it locally
        if rank == 0:
            try:
                ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
                ckpt_path_obj[0] = ckpt_path
            except Exception as e:
                print(f"Error downloading checkpoint: {e}")
                return None
        if master_process:
            print(f"Checkpoint downloaded to: {ckpt_path}")
        if is_ddp:
            dist.broadcast_object_list(ckpt_path_obj, src=0)
        ckpt_path = ckpt_path_obj[0]
    else:
        ckpt_path = os.path.join(ckpt_dir, filename)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")

    # Load the checkpoint onto the CPU first to avoid GPU memory issues
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False) # To load self-created checkpoints
    
    # Restore model state
    # The '.get()' method is used for safe key access in case a key is missing
    if 'model' in checkpoint:
        print("Restoring model state...")
        state_dict = checkpoint['model']
        if strip_ddp_prefix:
            # This handles cases where the model was saved with DDP wrapping, but wants to load without DDP
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    
    # Restore optimizer state
    if 'optimizer' in checkpoint and optimizer is not None:
        print("Restoring optimizer state...")
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    # Restore GradScaler state
    if 'scaler' in checkpoint and scaler is not None:
        print("Restoring GradScaler state...")
        scaler.load_state_dict(checkpoint['scaler'])
        
    # Restore RNG state
    if 'rng_state' in checkpoint:
        print("Restoring RNG state...")
        restore_rng_state(checkpoint['rng_state'], device_type=device)
    
    step = checkpoint.get('step', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    print(f"✅ Checkpoint loaded. Resuming from step {step} with val_loss {val_loss:.4f}")
    
    return {'step': step, 'val_loss': val_loss}

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
    train_loader = DataLoaderDisk(data, B, T)

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

def efficient_train(device, data_dir=None, data_repo_id=None, B=16, T=1024, context_size=1024, steps=50, total_batch_size=None, eval_every=10, save_ckpt_dir=None, save_repo_id=None, restore_ckpt_dir=None, restore_repo_id=None, restore_from_ckpt_filename=None, save_ckpt_every=100, log_dir=None, eval_with_hellaswag=False, compile=False, fast_learning=False):
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    from huggingface_hub import HfApi

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run_{timestamp}"

    # if compile and eval_with_hellaswag:
    #     print(f"Hellaswag eval is unavailable at compiled mode. Turning off")
    #     eval_with_hellaswag = False

    # set up DDP (Distributed Data Parallel)
    # torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_RANK
    ddp = dist.is_initialized()
    if ddp:
        # Use ddp atm demands CUDA
        assert torch.cuda.is_available(), "DDP only supported for CUDA for now"
        init_process_group(backend='nccl')
        ddp_rank = dist.get_rank() # Unique rank for this process
        ddp_local_rank = int(os.environ['LOCAL_RANK']) # Rank of GPU on a single node
        ddp_world_size = dist.get_world_size() # Total number of processes running
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # Does logging, checkpointing, etc.
    else:
        # non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

    set_seed(1337)

    if eval_with_hellaswag:
        hellaswag_eval = HellaSwagEval()

    use_amp, amp_dtype = get_training_precision(device)
    if use_amp:
        print(f"Using AMP with dtype: {amp_dtype}")
    
    # --- Initialize GradScaler based on the configuration ---
    scaler = torch.amp.GradScaler(enabled=(use_amp and device == 'cuda'))

    # create model
    config = GPT2Config(vocab_size=50304, context_size=context_size) # For vocab_size, use nice number with power of 2 (128)
    model = GPT2(config)
    model.to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)
    # compile the model for better performance with optimized python code and kernel fusion
    if compile:
        model = torch.compile(model)
    raw_model = model.module if ddp else model # For configure_optimizers

    if master_process:
        if log_dir is not None:
            log_dir = os.path.join(log_dir, run_name)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"log.txt")
            with open(log_file, "w") as f: # clear the file
                pass
        else:
            print("No log directory provided. Logs and checkpoints will not be saved.")
        if save_ckpt_dir and save_repo_id:
            print(f"Warning: both save_ckpt_dir and save_repo_id is provided. Pick huggingface repo.")
        if restore_ckpt_dir and restore_repo_id:
            print(f"Warning: both restore_ckpt_dir and restore_repo_id is provided. Pick huggingface repo.")
        if not restore_ckpt_dir and not restore_repo_id and restore_from_ckpt_filename:
            raise ValueError("Restore-from filename must be provided with a restore-from location (restore_ckpt_dir / restore_repo_id).")

    max_lr = 6e-4 * (3 if fast_learning else 1)
    min_lr = max_lr * 0.1
    warmup_steps = 375e6 / total_batch_size # GPT-3 warm-up is 375M tokens
    if master_process:
        param_log = f"max_lr: {max_lr}, min_lr: {min_lr}, steps: {steps}, warmup_steps: {warmup_steps}"
        print(param_log)
        if log_dir is not None:
            with open(log_file, "a") as f:
                f.write(param_log)

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
    if total_batch_size is not None:
        assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by batch_size * max_length * process_count"
        gradient_accum_steps = total_batch_size // (B * T * ddp_world_size)
        if master_process:
            print(f"total desired batch size: {total_batch_size}")
            print(f"-> calculated gradient accumulation iterations: {gradient_accum_steps}")

    # optimization
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    optimizer = raw_model.configure_optimizers(device=device, learning_rate=max_lr, betas=(0.9, 0.95), eps=1e-8)

    start_step = 0
    if (restore_ckpt_dir or restore_repo_id) and restore_from_ckpt_filename:
        ckpt_meta = restore_checkpoint(restore_from_ckpt_filename, raw_model, optimizer, scaler, device, repo_id=restore_repo_id, ckpt_dir=restore_ckpt_dir, rank=ddp_rank, master_process=master_process)
        start_step = ckpt_meta['step']
    
    if data_dir is not None and data_repo_id is not None:
        raise ValueError("Please provide either data_dir or repo_id, not both.")

    if data_dir is not None:
        # Load from local disk
        if master_process: print(f"Loading data from local directory: {data_dir}")
        train_loader = DataLoaderDisk(data_dir, B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', master_process=master_process, start_step=start_step)
        eval_loader = DataLoaderDisk(data_dir, B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val', master_process=master_process)
    elif data_repo_id is not None:
        # Load from Hugging Face Hub
        if master_process: print(f"Loading data from Hugging Face repo: {data_repo_id}")
        train_loader = DataLoaderHuggingFace(data_repo_id, B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', master_process=master_process, start_step=start_step)
        eval_loader = DataLoaderHuggingFace(data_repo_id, B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val', master_process=master_process)
    else:
        raise ValueError("Please provide a data source via data_dir or repo_id.")

    for step in range(start_step, steps):
        t0 = time.time()
        step_count = step + 1 # This is the number of steps until now, used for {eval|save}_every calculation
        last_step = step_count == steps
        eval_step = step_count % eval_every == 0 or last_step

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(gradient_accum_steps):
            x, y = train_loader.next_batch() # get next batch
            x, y = x.to(device), y.to(device) # move to device
            if ddp:
                # only sync at last step
                # It's the same as no_sync()
                model.require_backward_grad_sync = (micro_step == gradient_accum_steps - 1)
            # forward pass with automatic mixed precision (AMP) if enabled
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            loss = loss / gradient_accum_steps # normalize the loss for mean-loss calculation
            loss_accum += loss.detach()
            scaler.scale(loss).backward()  # scale the loss for AMP
        if ddp:
            # Average loss_accum across all processes
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
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
        tokens_processed = train_loader.B * train_loader.T * gradient_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / (t1 - t0)  # tokens per second
        if master_process:
            print(f"Step {step:4d}, Loss: {loss_accum.item()}, lr: {lr:.4e}, Norm: {norm:.4f}, Time: {dt:.2f} ms, Tokens/sec: {tokens_per_sec:.2f}")
            if log_dir is not None:
                with open(log_file, "a") as f:
                    f.write(f"{step} train {loss_accum.item():.4f}\n")

        # eval using evaluation dataset
        if eval_step:
            model.eval()
            eval_loader.reset()
            with torch.no_grad():
                eval_loss_accum = 0.0
                eval_loss_steps = 20
                for _ in range(eval_loss_steps):
                    x, y = eval_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                        _, loss = model(x, y)
                    loss = loss / eval_loss_steps
                    eval_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(eval_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"Validation loss: {eval_loss_accum.item():.4f}")
                if log_dir is not None:
                    with open(log_file, "a") as f:
                        f.write(f"{step} eval {eval_loss_accum.item():.4f}\n")
                step_to_save = step_count % save_ckpt_every == 0 or last_step
                if (save_ckpt_dir or save_repo_id) and step_to_save:
                    ckpt_name = f"{run_name}_ckpt_{tag_from_config(config)}_step{step_count:05d}.pt"
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': raw_model.config,
                        'step': step_count,
                        'val_loss': eval_loss_accum.item(),
                        'scaler': scaler.state_dict(),
                        'rng_state': save_rng_state(device)
                    }
                    if save_repo_id:
                        api = HfApi()
                        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmpfile:
                            torch.save(checkpoint, tmpfile.name)
                            # Upload the temporary file
                            api.upload_file(
                                path_or_fileobj=tmpfile.name,
                                path_in_repo=ckpt_name,
                                repo_id=save_repo_id,
                                repo_type="model"
                            )
                        print(f"Saved checkpoint {ckpt_name} to HuggingFace repo {save_repo_id}")
                        os.remove(tmpfile.name) # Clean up the temporary file    
                    else:
                        os.makedirs(save_ckpt_dir, exist_ok=True)
                        ckpt_path = os.path.join(log_dir, ckpt_name)
                        torch.save(checkpoint, ckpt_path)
                        print(f"Saved checkpoint {ckpt_name} to {ckpt_path}")
        
        # eval using hellaswag
        if eval_with_hellaswag and eval_step:
            acc_norm, num_correct, num_total = hellaswag_eval.evaluate_ddp(model=model, device=device, rank=ddp_rank, world_size=ddp_world_size, compile=False, print_first=0)
            if master_process:
                print(f"HellaSwag accuracy: {num_correct}/{num_total}={acc_norm:.4f}")
                if log_dir is not None:
                    with open(log_file, "a") as f:
                        f.write(f"{step} hella {acc_norm:.4f}\n")

    if ddp:
        destroy_process_group()
    return model

if __name__ == "__main__":
    # ----------------------------------------
    # auto detect device
    device = auto_pick_device()
    # device = "cpu" # override to cpu until cuda is available. MPS does not work well with pytorch, especially for training.
    print(f"Using device: {device}")

    model = efficient_train(device, data_dir="edu_fineweb10B", steps=123, B=4, T=256, total_batch_size=4*256*2, log_dir="log", compile=True,
                            eval_with_hellaswag=False, restore_ckpt_dir='log/run_20250831_134704/', save_ckpt_dir='log',
                            restore_from_ckpt_filename='run_20250831_134704_ckpt_125M_ctx1024_00110.pt', save_ckpt_every=1000) # total_batch_size = 2**19, ~0.5M tokens for GPT3 training
    simple_eval(device, max_length=100, input="They feast on wine and swan while our own tables see naught but the shadow of a crust", model=model)