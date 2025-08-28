"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# initialize the tokenizer
# Needs to put it in global for multiprocessing
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # 50256
def tokenize_as_uint16(doc):
    # tokenize a single document and return a numpy array of uint16
    tokens = [eot] # <|endoftext|> is delimiter between documents. Adds one as the start too.
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (tokens_np >= 0).all() and (tokens_np < 2**16).all(), "Token out of range for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def prepare_and_save_fineweb_dataset(local_dir="edu_fineweb10B", target_subset="sample-10BT", shard_size=int(1e8)):
    """
    Load and tokenize the fineweb dataset, and split into shards of given size.
    Writes the tokenized shards to the local directory provided.
    Paramters:
    - local_dir: local directory to save the shards
    - target_subset: which subset of the dataset to load
    - shard_size: size of each shard in number of tokens. Default to 100M tokens.
    """
    # create the cache the local directory if it doesn't exist yet
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=target_subset, split="train")
    
    def write_datafile(filename, tokens_np):
        # write a single shard to disk
        np.save(filename, tokens_np)
    
    # tokenize all documents and write in shards, with multiprocessing
    n_procs = max(1, os.cpu_count() // 2)
    with mp.Pool(n_procs) as pool:
        shard_index = 0
        # preallocate buffer to build current shard.
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize_as_uint16, ds, chunksize=16):
            # Check if there is enough space in the current shard for the new tokens
            if token_count + len(tokens) < shard_size:
                # enough space, add to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # not enough space, fill up the current shard and write to disk
                space_left = shard_size - token_count
                all_tokens_np[token_count:token_count+space_left] = tokens[:space_left]
                # write the current shard to disk
                if progress_bar is not None:
                    progress_bar.update(space_left)
                    progress_bar.close()
                    progress_bar = None
                split = "val" if shard_index == 0 else "train"
                shard_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                write_datafile(shard_filename, all_tokens_np)
                print(f"Wrote shard {shard_index} with {shard_size} tokens to {shard_filename}")
                shard_index += 1
                # start a new shard with the remaining tokens
                remaining_tokens = tokens[space_left:]
                all_tokens_np[:len(remaining_tokens)] = remaining_tokens
                token_count = len(remaining_tokens)
                # update progress bar for the new shard
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(remaining_tokens))
        # write any remaining tokens as the last shard
        if token_count > 0:
            if progress_bar is not None:
                progress_bar.update(token_count)
                progress_bar.close()
                progress_bar = None
            split = "val" if shard_index == 0 else "train"
            shard_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(shard_filename, all_tokens_np[:token_count])
            print(f"Wrote final shard {shard_index} with {token_count} tokens to {shard_filename}")

if __name__ == "__main__":
    prepare_and_save_fineweb_dataset()