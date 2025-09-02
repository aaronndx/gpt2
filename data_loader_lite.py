import numpy as np
import torch
import os
from huggingface_hub import hf_hub_download

class DataLoaderLite:
    """
    Base class for loading data shards. It handles batching, process distribution,
    and shard rollover. Subclasses must implement the _load_tokens method.
    """
    def __init__(self, B, T, process_rank, num_processes, split, master_process, start_step, batch_size_per_step):
        self.B = B  # Batch size
        self.T = T  # Sequence length
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        assert split in {'train', 'val'}
        self.tokens_per_shard = 100_000_000 # 100M tokens for each shard, expect the final one.
        self.start_step = start_step
        if self.start_step > 0:
            assert split == 'train', "Evaluation data loading doesn't support restoring from step."
            self.total_token_size = self.tokens_per_shard * 98 + 53989344 # The real total token count for training data.
        if batch_size_per_step:
            # User can override tokens per step in cases like gradient accumulation,
            # in which case a step consists of multiple micro steps each with size B * T * num_processes
            self.tokens_per_step = batch_size_per_step
        else:
            self.tokens_per_step = self.B * self.T * self.num_processes
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
        Resets the data loader to the beginning.
        If no start_step, goes to beginning of first shard. Otherwise restore to start_step's state.
        This won't restore state to exactly the position of the last training, since we skip some data
        at end of shards, and those are not calculated here. Though that should not impact result too much.
        """
        if self.start_step > 0:
            tokens_to_skip = self.start_step * self.tokens_per_step
            tokens_to_skip_cur_epoch = tokens_to_skip % self.total_token_size
            self.current_shard = tokens_to_skip_cur_epoch // self.tokens_per_shard
            self.tokens = self._load_tokens(self.shards[self.current_shard])
            pos_in_shard = tokens_to_skip_cur_epoch % self.tokens_per_shard
            self.current_position = pos_in_shard + self.B * self.T * self.process_rank
        else:
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
    def __init__(self, dir, B, T, process_rank=0, num_processes=1, split='val', master_process=True, start_step=0, batch_size_per_step=None):
        super().__init__(B, T, process_rank, num_processes, split, master_process, start_step, batch_size_per_step)
        
        # List and sort the shard files from the directory
        shards = os.listdir(dir)
        shards = [s for s in shards if self.split in s]
        shards = sorted(shards)
        self.shards = [os.path.join(dir, s) for s in shards]
        
        assert len(self.shards) > 0, f"No shards found for split '{split}' in directory '{dir}'"
        if self.master_process:
            print(f"Found {len(self.shards)} shards for split '{split}' on disk.")
        
        # Initialize the loader by loading the first shard, or restoring from start_step
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
    def __init__(self, repo_id, B, T, process_rank=0, num_processes=1, split='val', master_process=True, start_step=0, batch_size_per_step=None):
        super().__init__(B, T, process_rank, num_processes, split, master_process, start_step, batch_size_per_step)
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

        # Initialize the loader by loading the first shard, or restoring from start_step
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