import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import distributed as dist
from transformers import GPT2LMHeadModel
import itertools
import sys
from IPython.display import display, clear_output

class HellaSwagEval:
    """
    Downloads and evaluates HellaSwag in Python.
    https://github.com/rowanz/hellaswag

    Example HellaSwag json item:

    {"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

    ind: dataset ID
    activity_label: The ActivityNet or WikiHow label for this example
    context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
    endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
    split: train, val, or test.
    split_type: indomain if the activity label is seen during training, else zeroshot
    source_id: Which video or WikiHow article this example came from

    gpt2 (124M)
    - eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
    - this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

    gpt2-xl (1558M)
    - eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
    - this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

    The validation set of HellaSwag has a total of 10,042 examples.
    """

    class _LiveLogger:
        """
        A universal logger that updates text in-place.
        It automatically detects if it's in a notebook (Colab, Jupyter)
        or a standard terminal and uses the appropriate method.
        """
        def __init__(self):
            self.last_lines_printed = 0
            
            # Check if we are in a notebook environment
            self.is_notebook = 'ipykernel' in sys.modules
            
            # In a notebook, we might need a handle for displaying things
            if self.is_notebook:
                self.display_handle = display("Initializing HellaSwag Logger...", display_id=True)

        def reset(self):
            self.last_lines_printed = 0

        def log(self, text_to_print):
            if self.is_notebook and self.display_handle:
                # --- Notebook Method ---
                self.display_handle.update(text_to_print)
            else:
                # --- Terminal Method ---
                # Move cursor up to overwrite previous lines
                for _ in range(self.last_lines_printed):
                    print('\033[A', end='')

                # Print new lines, clearing each one
                lines = text_to_print.split('\n')
                for line in lines:
                    print(f"\r\033[K{line}")
                
                self.last_lines_printed = len(lines)

    def __init__(self):
        try:
            self.data_cache_dir = os.path.join(os.path.dirname(__file__), "hellaswag")
        except NameError:
            self.data_cache_dir = "hellaswag"
        print(f"Setting hellaswag cache dir to {self.data_cache_dir}")
        self.hellaswags_urls = {
            "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
            "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
            "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
        }
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.logger = self._LiveLogger()
    
    def _download(self, split):
        """Downloads HellaSwag to cache"""
        def download_file(url: str, fname: str, chunk_size=1024):
            """Helper to download a file from a given url"""
            resp = requests.get(url, stream=True)
            total = int(resp.headers.get("content-length", 0))
            with open(fname, "wb") as file, tqdm(
                desc=fname,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in resp.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)

        os.makedirs(self.data_cache_dir, exist_ok=True)
        data_url = self.hellaswags_urls[split]
        data_filename = os.path.join(self.data_cache_dir, f"hellaswag_{split}.jsonl")
        if not os.path.exists(data_filename):
            print(f"Downloading {data_url} to {data_filename}...")
            download_file(data_url, data_filename)
    
    def _render_example(self, example):
        """
        Given the example as a dictionary, render it as three torch tensors:
        - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
        - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
        - label (the index of the correct completion, which we hope has the highest likelihood)
        """
        ctx = example["ctx"]
        label = example["label"]
        endings = example["endings"]

        # data needed to reproduce this eval on the C size
        data = {
            "label": label,
            "ctx_tokens": None,
            "ending_tokens": [],
        }

        # gather up all the tokens
        ctx_tokens = self.tokenizer.encode(ctx)
        data["ctx_tokens"] = ctx_tokens
        tok_rows = []
        mask_rows = []
        for end in endings:
            end_tokens = self.tokenizer.encode(" " + end) # preprending " " for GPT-2 tokenizer
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
            data["ending_tokens"].append(end_tokens)
        
        # handle collation of different length options to zero-patch to same length
        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((4, max_len), dtype=torch.long)
        mask = torch.zeros((4, max_len), dtype=torch.long)
        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, :len(tok_row)] = torch.tensor(tok_row)
            mask[i, :len(mask_row)] = torch.tensor(mask_row)
        
        return data, tokens, mask, label
    
    def _iterate_examples(self, split, rank=0, world_size=1):
        # there are 10,042 examples in total in val
        if rank == 0:
            self._download(split)
        # Make sure all process read after download
        if dist.is_initialized():
            dist.barrier()

        with open(os.path.join(self.data_cache_dir, f"hellaswag_{split}.jsonl"), "r") as f:
            num_lines = sum(1 for _ in f)
        # Calculate start and end for current rank (proc)
        lines_per_rank = num_lines // world_size
        start_idx = rank * lines_per_rank
        end_idx = (rank + 1) * lines_per_rank if rank + 1 != world_size else num_lines # last rank reads until end
        
        with open(os.path.join(self.data_cache_dir, f"hellaswag_{split}.jsonl"), "r") as f:
            line_iterator = itertools.islice(f, start_idx, end_idx)
            for line in line_iterator:
                example = json.loads(line)
                yield example
    
    @torch.no_grad()
    def evaluate_ddp(self, model, device, rank, world_size, compile=False, print_first=10):
        ddp = dist.is_initialized()

        torch.set_float32_matmul_precision('high') # tf32
        model.to(device)
        if compile:
            model = torch.compile(model)
        model.eval()
        
        num_correct_norm_local = 0
        num_correct_local = 0
        num_total_local = 0

        # Print only for master process
        iterable = self._iterate_examples("val", rank, world_size)
        if rank == 0:
            iterable = tqdm(iterable, desc="Evaluating with HellaSwag")
            self.logger.reset()

        for example in iterable:
            data, tokens, mask, label = self._render_example(example)
            tokens = tokens.to(device) # (4, T)
            mask = mask.to(device) # (4, T)

            try:
                logits = model(tokens).logits # (4, T, Emb), pretrained model
            except AttributeError:
                logits, _ = model(tokens) # (4, T, Emb), self-defined model
            # evaluate the autoregressive loss at all positions
            shift_logits = (logits[..., :-1, :]).contiguous() # Slice off the last step since no targets for it
            shift_tokens = (tokens[..., 1:]).contiguous() # Targets
            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_tokens = shift_tokens.view(-1)
            shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none') # (4 * (T - 1),)
            shift_losses = shift_losses.view(tokens.size(0), -1) # (4, T - 1)
            # get the average loss for the completion region, where mask = 1, in each row
            shift_mask = (mask[..., 1:]).contiguous()
            masked_shift_losses = shift_losses * shift_mask # (4, T - 1)
            # sum and divide by the number of 1s in the mask
            sum_loss = masked_shift_losses.sum(dim=1) # (4,)
            avg_loss = sum_loss / shift_mask.sum(dim=1)
            # one with lowest loss should be the most likely
            pred = sum_loss.argmin().item()
            pred_norm = avg_loss.argmin().item()

            # accumulate stats
            num_total_local += 1
            num_correct_local += int(pred == label)
            num_correct_norm_local += int(pred_norm == label)

            # DEBUG: pretty print a few examples, and the losses in each case
            if rank == 0 and num_total_local < print_first:
                print("---")
                print(f"Context:\n {example['ctx']}")
                print(f"Endings:")
                for i, end in enumerate(example["endings"]):
                    print(f"{i} (loss: {avg_loss[i].item():.4f}), {end}")
                print(f"predicted: {pred_norm}, actual: {label}")
            
            # Aggregate DDP results
            if ddp:
                local_results = torch.tensor([num_correct_local, num_correct_norm_local, num_total_local], dtype=torch.long, device=device)
                dist.all_reduce(local_results, op=dist.ReduceOp.SUM)
                num_correct_global = local_results[0].item()
                num_correct_norm_global = local_results[1].item()
                num_total_global = local_results[2].item()
            else:
                num_correct_global = num_correct_local
                num_correct_norm_global = num_correct_norm_local
                num_total_global = num_total_local

            if rank == 0:
                acc_norm = num_correct_norm_global / num_total_global
                acc = num_correct_global / num_total_global # Not used since it's biased towards shorter answer

                log_text = (
                    "--- HellaSwag Final DDP Results ---\n"
                    f"Total Examples: {num_total_global}\n"
                    f"Accuracy: {acc_norm:.4f} ({num_correct_norm_global}/{num_total_global})\n"
                )
                self.logger.log(log_text)
        
            if ddp:
                dist.barrier()

        return acc_norm, num_correct_norm_global, num_total_global
    
    def evaluate(self, model, device, compile=False, print_first=10):
        return self.evaluate_ddp(model=model, device=device, compile=compile, print_first=print_first, rank=0, world_size=1)

if __name__ == "__main__":
    eval = HellaSwagEval()
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    final_acc_norm, _, _ = eval.evaluate(model=model, device="cpu", compile=True)
    print(f"Final acc norm: {final_acc_norm}")