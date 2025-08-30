import numpy as np
import torch
import os
from huggingface_hub import hf_hub_download

class DataLoaderLite:
    """
    Base class for loading data shards. It handles batching, process distribution,
    and shard rollover. Subclasses must implement the _load_tokens method.
    """
    def __init__(self, B, T, process_rank=0, num_processes=1, split='val', master_process=True):
        self.B = B  # Batch size
        self.T = T  # Sequence length
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        assert split in {'train', 'val'}
        self.master_process = master_process

        self.shards = []
        self.current_shard = 0
        self.tokens = None
        self.current_position = 0

    def _load_tokens(self, source):
        """
        Loads tokens from a given source. This method must be implemented by subclasses.
        Source could be a file path, a URL, etc.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def reset(self):
        """
        Resets the data loader to the beginning of the first shard.
        """
        self.current_shard = 0
        self.tokens = self._load_tokens(self.shards[self.current_shard])
        # Scatter data across processes
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """
        Returns the next batch of data. Handles moving to the next shard when the
        current one is exhausted.
        """
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # Input tokens
        y = buf[1:].view(B, T)   # Target tokens
        
        # Advance the position in the current shard
        self.current_position += B * T * self.num_processes
        
        # If the current shard is exhausted, move to the next one
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self._load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        
        return x, y

# ---

class DataLoaderDisk(DataLoaderLite):
    """
    Loads tokenized data shards from a local disk directory.
    """
    def __init__(self, dir, B, T, process_rank=0, num_processes=1, split='val', master_process=True):
        super().__init__(B, T, process_rank, num_processes, split, master_process)
        
        # List and sort the shard files from the directory
        shards = os.listdir(dir)
        shards = [s for s in shards if self.split in s]
        shards = sorted(shards)
        self.shards = [os.path.join(dir, s) for s in shards]
        
        assert len(self.shards) > 0, f"No shards found for split '{split}' in directory '{dir}'"
        if self.master_process:
            print(f"Found {len(self.shards)} shards for split '{split}' on disk.")
        
        # Initialize the loader by loading the first shard
        self.reset()

    def _load_tokens(self, filename):
        """
        Loads a .npy file from disk and converts it to a PyTorch tensor.
        """
        if self.master_process:
            print(f"Loading tokens from disk: {filename}")
        npt = np.load(filename)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

# ---

class DataLoaderHuggingFace(DataLoaderLite):
    """
    Streams tokenized data shards from a Hugging Face Hub repository.
    To use this, needs to first login to hugging face by `huggingface-cli login`
    """
    def __init__(self, repo_id, B, T, process_rank=0, num_processes=1, split='val', master_process=True):
        super().__init__(B, T, process_rank, num_processes, split, master_process)
        self.repo_id = repo_id
        
        # Programmatically generate the shard filenames
        if self.split == 'train':
            # Assumes 99 training shards named edufineweb_train_000001.npy to edufineweb_train_000099.npy
            self.shards = [f"edufineweb_train_{i:06d}.npy" for i in range(1, 100)]
        else:  # 'val'
            self.shards = ["edufineweb_val_000000.npy"]

        assert len(self.shards) > 0, "Failed to generate shard list."
        if self.master_process:
            print(f"Found {len(self.shards)} shards for split '{split}' on Hugging Face Hub.")

        # Initialize the loader by loading the first shard
        self.reset()

    def _load_tokens(self, filename):
        """
        Downloads a shard from the Hugging Face Hub, caches it locally,
        and then loads it into a PyTorch tensor.
        """
        if self.master_process:
            print(f"Streaming tokens from Hugging Face Hub: {self.repo_id}/{filename}")
            
        # hf_hub_download handles caching, so files are only downloaded once
        cached_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type='dataset' # Specify that it's a dataset repository
        )
        
        npt = np.load(cached_path)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt