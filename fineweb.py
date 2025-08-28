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

if __name__ == "__main__":
    prepare_and_save_fineweb_dataset()