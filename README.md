# Optimized GPT-2 From Scratch

This repository contains a from-scratch implementation of an optimized GPT-2 model in PyTorch with training & evaluation.

The project is focused on providing a clean, simple, and highly efficient framework for training and evaluating language models, incorporating modern performance optimizations and advanced training techniques inspired by the GPT-2 and GPT-3 papers.

Besides original papers, this project also refers to lecture [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) by Andrej Karpathy for applying training optimizations.

## Features

Below is the list of components/features implemented in this project, designed for realizing optimized GPT-2 from scratch with high-performance, distributed training and robust experimentation.

---

### ⚙️ Core Model Architecture

The model implementation is based on the GPT-2 architecture.

* **Standard GPT-2 Blocks:** Implements the standard decoder-only transformer block with multi-head causal self-attention and a position-wise MLP.
* **Weight Sharing:** Shares weights between the token embedding layer and the final language model head, a common practice for improving model performance.
* **Pre-trained Weight Loading:** Includes a utility to load pre-trained weights directly from Hugging Face's `gpt2`, `gpt2-medium`, `gpt2-large`, and `gpt2-xl` models.
* **Simple Inference:** Provides a straightforward `simple_eval` function for generating text samples from a trained model.

---

### 🚀 High-Performance Training

The training script is optimized for speed and memory efficiency on modern hardware.

* **Mixed Precision Training:** Automatically uses `bfloat16` or `float16` to reduce memory usage and accelerate training on supported GPUs (CUDA, MPS) and CPUs.
* **Flash Attention:** Integrates `F.scaled_dot_product_attention` (Flash Attention) to significantly speed up the attention mechanism and reduce memory reads/writes.
* **`torch.compile`:** The model and training loop are fully compatible with `torch.compile`, which JIT-compiles the PyTorch code into optimized kernels for a substantial performance boost.
* **Optimized GPU Kernels:**
    * Uses TensorFloat32 (TF32) on Ampere and newer GPUs for up to 8x faster matrix multiplications.
    * Leverages a fused AdamW optimizer kernel when available on CUDA.
* **Optimal Dimensioning:** Encourages using vocabulary sizes and model dimensions that are powers of 2 to maximize GPU utilization.

---

### 🏋️ Advanced Training Techniques

The training process incorporates best practices from influential research papers to ensure stable and effective learning.

* **Distributed Data Parallel (DDP):** Full support for multi-GPU training using PyTorch's DDP to scale training across multiple devices.
* **Gradient Accumulation:** Simulates a much larger batch size than can fit in memory by accumulating gradients over several smaller micro-batches.
* **GPT-3 Learning Rate Schedule:** Implements the warmup and cosine decay learning rate schedule as described in the GPT-3 paper. The learning rate multiplier can be adjusted for different training speeds.
* **Optimized Weight Decay:** Applies weight decay only to 2D weight matrices (e.g., in `Linear` layers), excluding biases and LayerNorm parameters.
* **GPT-3 Optimizer Parameters:** Uses the recommended AdamW parameters from the GPT-3 paper (`beta1=0.9`, `beta2=0.95`, `eps=1e-8`).
* **Gradient Clipping:** Clips the global norm of gradients to 1.0 to prevent exploding gradients and stabilize training, especially in the early stages.
* **Weight Initialization:** Follows the weight initialization scheme from the GPT-2 paper, including special scaling for residual projection layers to stabilize training in deep networks.

---

### 💾 Efficient Data Handling

The data pipeline is designed to handle massive datasets by processing them in smaller, manageable shards.

* **Data Preparation Script (`fineweb.py`):**
    * Downloads the 10B-token FineWeb-Edu dataset from Hugging Face.
    * Uses multiprocessing to tokenize the data efficiently.
    * Saves the tokenized data into smaller, sharded `.npy` files for fast loading.
    * Includes an option to clear the Hugging Face cache after download to conserve disk space, which is especially useful in environments like Google Colab.
* **Custom Data Loader (`DataLoaderLite`):**
    * A lightweight, efficient data loader that reads from the tokenized shards.
    * Supports loading from both a **local directory** and streaming directly from a **Hugging Face Hub repository**.
    * Fully integrated with DDP, ensuring each process gets a unique slice of the data.
    * Capable of approximately restoring the data loading position when resuming training from a checkpoint.

---

### 🔄 Robust Checkpointing & Resuming

The checkpointing system is designed for fault-tolerant training and seamless resumption.

* **Comprehensive State Saving:** Checkpoints save not only the model weights but also the state of the optimizer, learning rate scheduler, gradient scaler, and RNG (random number generator) to ensure a fully reproducible training state.
* **Hugging Face Hub Integration:**
    * Save checkpoints directly to a Hugging Face Hub repository.
    * Resume training by downloading checkpoints from a repository.
    * Includes utilities to verify Hugging Face login and repository write permissions before training starts.
* **Flexible Resumption:** Can resume training from local checkpoints or from the Hub. The logic automatically handles state dict prefix differences that may arise from using `torch.compile` or DDP.
* **Informative Naming:** Checkpoint files are automatically named with the run timestamp, model configuration, and training step for easy identification.

---

### 📊 Evaluation & Logging

The repository includes tools for monitoring training progress and evaluating model performance.

* **HellaSwag Evaluation:**
    * Integrates HellaSwag, a challenging commonsense reasoning benchmark, into the training loop.
    * The evaluation is also distributed to run efficiently in a multi-GPU setup.
    * Features a live-updating logger that shows progress without cluttering the console.
* **Loss Logging:**
    * Logs training and validation loss to a text file (`log.txt`) during training.
    * A utility script is provided to parse this log file and plot the loss curves for analysis.
* **Performance Metrics:** The training loop logs key metrics, including loss, learning rate, gradient norm, and processing speed in tokens/second.

## Reference

* [Let's reproduce GPT-2 (124M) by Andrej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU)
* [GPT-2 Paper: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [GPT-3 Paper: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)