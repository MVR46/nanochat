# NanoChat Training Guide

A comprehensive guide to training, saving, and trying your own ChatGPT-like language model with nanochat.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start - The Speedrun](#quick-start---the-speedrun)
4. [Understanding the Training Pipeline](#understanding-the-training-pipeline)
5. [How Models are Saved](#how-models-are-saved)
6. [How to Use Your Trained Model](#how-to-use-your-trained-model)
7. [Training Custom Models](#training-custom-models)
8. [Running on Different Hardware](#running-on-different-hardware)

---

## Overview

Nanochat is a complete implementation of a ChatGPT-like language model training pipeline. The process involves:

1. **Tokenizer Training** - Creating a custom BPE tokenizer
2. **Pretraining** - Training the base language model on web text
3. **Midtraining** - Teaching the model conversation format and special tokens
4. **Supervised Fine-tuning (SFT)** - Adapting the model to specific tasks
5. **Reinforcement Learning** (Optional) - Further improving specific capabilities

The entire pipeline is designed to run on a single 8xH100 GPU node, with costs ranging from ~$100 to ~$800 depending on model size.

---

## Prerequisites

### Hardware Requirements
- **Recommended**: 8xH100 GPUs (80GB VRAM each)
- **Also works**: 8xA100 GPUs (80GB VRAM each)
- **Minimum**: Single GPU with 80GB VRAM (will take 8x longer)
- **Lower VRAM**: Reduce `device_batch_size` parameter until it fits

### Software Requirements
- Python 3.10+
- CUDA-capable GPU (for training at scale)
- `uv` package manager (automatically installed by scripts)
- Rust/Cargo (automatically installed by scripts)

### Cloud Providers
The author recommends [Lambda GPU Cloud](https://lambda.ai/service/gpu-cloud) for renting GPU nodes.

---

## Quick Start - The Speedrun

The fastest way to train nanochat is using the `speedrun.sh` script, which trains the ~$100 tier model (d20 - 20 layers, 561M parameters) in about 4 hours on an 8xH100 node.

### Step 1: Launch the Training

```bash
# Simple launch
bash speedrun.sh

# Or launch in a screen session with logging (recommended)
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# With wandb logging (optional but recommended)
WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

### Step 2: Monitor Progress

If using screen, you can:
- Watch it run: `screen -r speedrun`
- Detach: `Ctrl-a d`
- Check logs: `tail -f speedrun.log`

### Step 3: Wait ~4 Hours

The script will:
1. Install dependencies (uv, Rust, Python packages)
2. Download training data (~24GB)
3. Train a custom tokenizer
4. Pretrain the base model
5. Run midtraining with conversation format
6. Fine-tune with supervised learning
7. Generate a report card (`report.md`)

### Step 4: Try Your Model

After training completes:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Chat via web UI (ChatGPT-like interface)
python -m scripts.chat_web

# Then visit http://<your-ip>:8000/
```

**Note**: If on a cloud instance, use the public IP address of your node (e.g., `http://209.20.xxx.xxx:8000/`)

---

## Understanding the Training Pipeline

### Phase 1: Tokenizer Training

**Location**: Lines 49-71 in `speedrun.sh`

```bash
# Build the Rust-based BPE tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download training data (~800MB for tokenizer, ~24GB total)
python -m nanochat.dataset -n 8

# Train tokenizer (vocab size 65,536) on ~2B characters
python -m scripts.tok_train --max_chars=2000000000

# Evaluate compression ratio
python -m scripts.tok_eval
```

**What it does**: Creates a custom Byte Pair Encoding (BPE) tokenizer similar to GPT-4's tokenizer, with 65,536 vocabulary tokens.

**Script**: `scripts/tok_train.py`

---

### Phase 2: Base Model Pretraining

**Location**: Lines 74-99 in `speedrun.sh`

```bash
# Pretrain the d20 model (20 layers, 561M parameters)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=$WANDB_RUN

# Evaluate bits per byte on validation data
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss

# Evaluate CORE metric (from DCLM paper)
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```

**What it does**: 
- Trains a 20-layer transformer on ~11.2B tokens of web text (FineWeb dataset)
- Uses Chinchilla scaling laws: 20x tokens per parameter
- Training time: ~2.5 hours on 8xH100
- Uses Muon optimizer for matrix parameters, AdamW for embeddings

**Key Parameters**:
- `depth=20`: Number of transformer layers
- `max_seq_len=2048`: Maximum context length
- `device_batch_size=32`: Batch size per GPU
- `total_batch_size=524288`: Total batch size in tokens

**Script**: `scripts/base_train.py`

**Model saved to**: `~/.cache/nanochat/base_checkpoints/d20/`

---

### Phase 3: Midtraining

**Location**: Lines 102-110 in `speedrun.sh`

```bash
# Download synthetic identity conversations (2.3MB)
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Run midtraining
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run=$WANDB_RUN

# Evaluate on chat benchmarks
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid
```

**What it does**:
- Teaches the model conversation format with special tokens:
  - `<|user_start|>`, `<|user_end|>`
  - `<|assistant_start|>`, `<|assistant_end|>`
- Trains on mixture of:
  - SmolTalk conversations
  - Multiple choice tasks (ARC, MMLU)
  - Math problems (GSM8K)
  - Code (HumanEval)
  - Custom identity conversations

**Script**: `scripts/mid_train.py`

**Model saved to**: `~/.cache/nanochat/mid_checkpoints/d20/`

---

### Phase 4: Supervised Fine-tuning (SFT)

**Location**: Lines 113-123 in `speedrun.sh`

```bash
# Train SFT model
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN

# Re-evaluate on all benchmarks
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

**What it does**:
- Fine-tunes on ~23K examples from:
  - ARC-Easy (2.3K rows)
  - ARC-Challenge (1.1K rows)
  - GSM8K (8K rows)
  - SmolTalk (10K rows)
  - Identity conversations (1K rows)
  - Spelling tasks (600 rows)
- Typically improves performance by 2-5% on benchmarks

**Script**: `scripts/chat_sft.py`

**Model saved to**: `~/.cache/nanochat/chatsft_checkpoints/d20/`

---

### Phase 5: Reinforcement Learning (Optional)

**Location**: Lines 127-132 in `speedrun.sh` (commented out by default)

```bash
# Run RL training (GSM8K only)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=$WANDB_RUN

# Evaluate on GSM8K
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K
```

**What it does**: Uses reinforcement learning to improve math problem solving (GSM8K tasks)

**Script**: `scripts/chat_rl.py`

**Model saved to**: `~/.cache/nanochat/chatrl_checkpoints/d20/`

---

## How Models are Saved

### Checkpoint Structure

All models are saved to `~/.cache/nanochat/` by default. Each checkpoint directory contains:

```
~/.cache/nanochat/
├── base_checkpoints/
│   └── d20/
│       ├── model_000000.pt       # Model weights
│       ├── optim_000000.pt       # Optimizer state (optional)
│       └── meta_000000.json      # Metadata (config, metrics)
├── mid_checkpoints/
│   └── d20/
│       └── ...
├── chatsft_checkpoints/
│   └── d20/
│       └── ...
└── chatrl_checkpoints/
    └── d20/
        └── ...
```

### Checkpoint Components

1. **`model_XXXXXX.pt`** - PyTorch state dict with model parameters
2. **`optim_XXXXXX.pt`** - Optimizer state (for resuming training)
3. **`meta_XXXXXX.json`** - JSON file with:
   - Model configuration (layers, dimensions, vocab size)
   - Training configuration (learning rates, batch sizes)
   - Evaluation metrics (validation loss, CORE score)
   - Training step number

### Checkpoint Naming

- **Model tag**: `d20` means depth=20 (20 transformer layers)
- **Step number**: `000000` is the training step (6 digits, zero-padded)

### How Saving Works

The `checkpoint_manager.py` module handles all saving/loading:

```python
from nanochat.checkpoint_manager import save_checkpoint

# During training (only rank 0 saves)
save_checkpoint(
    checkpoint_dir,      # e.g. "~/.cache/nanochat/base_checkpoints/d20"
    step,                # training step number
    model.state_dict(),  # model parameters
    optimizer_state,     # optimizer state (or None)
    meta_data           # dict with config and metrics
)
```

**Location in code**: `nanochat/checkpoint_manager.py`

---

## How to Use Your Trained Model

### Method 1: Web UI (Recommended)

Launch a ChatGPT-like web interface:

```bash
# Activate virtual environment
source .venv/bin/activate

# Serve the SFT model (default)
python -m scripts.chat_web

# Or specify a different model
python -m scripts.chat_web -i mid    # Use midtrained model
python -m scripts.chat_web -i base   # Use base model
python -m scripts.chat_web -i rl     # Use RL model

# Use multiple GPUs for load balancing
python -m scripts.chat_web --num-gpus 4

# Customize generation parameters
python -m scripts.chat_web -t 0.7 -k 40 -m 256
```

**Parameters**:
- `-i, --source`: Which model to load (sft/mid/base/rl)
- `-t, --temperature`: Temperature (default: 0.8)
- `-k, --top-k`: Top-k sampling (default: 50)
- `-m, --max-tokens`: Max response length (default: 512)
- `-n, --num-gpus`: Number of GPUs for load balancing
- `-p, --port`: Port number (default: 8000)

**Access**: Visit `http://<server-ip>:8000/` in your browser

---

### Method 2: Command Line Interface

Chat interactively in the terminal:

```bash
# Interactive mode
python -m scripts.chat_cli -i sft

# Single prompt mode
python -m scripts.chat_cli -i sft -p "Why is the sky blue?"

# Customize parameters
python -m scripts.chat_cli -i sft -t 0.7 -k 50
```

**Commands**:
- Type your message and press Enter
- `clear` - Start a new conversation
- `quit` or `exit` - Exit the program
- `Ctrl+C` - Force quit

---

### Method 3: Programmatic Usage

Use the model in your own Python code:

```python
import torch
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, meta = load_model("sft", device, phase="eval")

# Create generation engine
engine = Engine(model, tokenizer)

# Prepare conversation
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")

# Build conversation tokens
conversation = [bos]
conversation.append(user_start)
conversation.extend(tokenizer.encode("Why is the sky blue?"))
conversation.append(user_end)
conversation.append(assistant_start)

# Generate response
response_tokens = []
for token_column, token_masks in engine.generate(
    conversation,
    num_samples=1,
    max_tokens=256,
    temperature=0.7,
    top_k=50
):
    token = token_column[0]
    response_tokens.append(token)
    print(tokenizer.decode([token]), end="", flush=True)
```

---

## Training Custom Models

### Larger Models

To train a larger model (e.g., d26 - 26 layers, ~$300, 12 hours):

```bash
# Modify speedrun.sh or create a new script:

# 1. Download more data shards
python -m nanochat.dataset -n 450 &  # More data for larger model

# 2. Train with increased depth and reduced batch size (to fit in VRAM)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --device_batch_size=16

# 3. Use same batch size for midtraining
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \
    --device_batch_size=16

# 4. Continue with SFT normally
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
```

**Model Sizes**:
- d20: 561M params (~$100, 4 hours)
- d26: ~1.2B params (~$300, 12 hours)
- d32: 1.9B params (~$800, 33 hours)

### Key Parameters to Tune

**Model Architecture** (`scripts/base_train.py`):
- `--depth`: Number of transformer layers (default: 20)
- `--max_seq_len`: Context length (default: 2048)

**Training Duration**:
- `--num_iterations`: Explicit number of steps
- `--target_param_data_ratio`: Tokens per parameter (default: 20, per Chinchilla)

**Memory Management**:
- `--device_batch_size`: Batch size per GPU (default: 32)
  - Reduce if OOM: 32 → 16 → 8 → 4 → 2 → 1
  - Code auto-compensates with gradient accumulation

**Learning Rates**:
- `--embedding_lr`: Embedding layer LR (default: 0.2)
- `--unembedding_lr`: Output layer LR (default: 0.004)
- `--matrix_lr`: Transformer layers LR (default: 0.02, Muon optimizer)

---

### Customizing Model Personality

To give your model a custom identity (see [Discussion #139](https://github.com/karpathy/nanochat/discussions/139)):

1. **Generate synthetic conversations** using `dev/gen_synthetic_data.py`
2. **Create a JSONL file** with your custom conversations:

```json
{"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm YourBot, a helpful AI created by YourCompany."}]}
{"messages": [{"role": "user", "content": "What's your purpose?"}, {"role": "assistant", "content": "I help users with coding and technical questions."}]}
```

3. **Replace the identity file** in `speedrun.sh`:

```bash
# Instead of downloading the default identity file:
# curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl ...

# Use your custom file:
cp my_custom_identity.jsonl $NANOCHAT_BASE_DIR/identity_conversations.jsonl
```

4. **Run the training pipeline** - your identity will be mixed into midtraining and SFT

---

### Training on Single GPU

To train on a single GPU:

```bash
# Remove torchrun, just use python
python -m scripts.base_train -- --depth=20

# The code automatically:
# - Switches to gradient accumulation (instead of data parallel)
# - Produces identical results
# - Takes ~8x longer (4 hours → 32 hours for d20)
```

---

## Running on Different Hardware

### CPU / Mac (MPS)

Nanochat supports CPU and Mac Metal Performance Shaders (MPS):

```bash
# See dev/runcpu.sh for a complete example

# Train a tiny model on CPU/MPS
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --total_batch_size=512 \
    --num_iterations=20 \
    --core_metric_every=-1

# Chat with the model
python -m scripts.chat_cli --device-type mps  # or cpu
```

**Note**: Training large models on CPU/MPS is impractical, but you can:
- Test the code paths
- Train tiny models for experimentation
- Run inference on SFT models

---

### Lower VRAM GPUs

If you have GPUs with less than 80GB VRAM:

1. **Reduce batch size**:
```bash
python -m scripts.base_train --device_batch_size=16  # or 8, 4, 2, 1
```

2. **Train smaller models**:
```bash
python -m scripts.base_train --depth=12  # ~200M params
```

3. **Reduce context length**:
```bash
python -m scripts.base_train --max_seq_len=1024
```

4. **Use gradient checkpointing** (not implemented by default, requires code modification)

---

## Evaluation and Metrics

### Training Metrics

During training, nanochat tracks:

1. **Bits per byte (BPB)**: Validation loss normalized by UTF-8 bytes
   - Lower is better
   - Evaluated every 250 steps by default

2. **CORE metric**: Aggregate score from DCLM paper
   - 0.0 to 1.0 (higher is better)
   - Evaluated every 2000 steps by default

3. **Model Fusion Utilization (MFU)**: GPU efficiency
   - Percentage of theoretical peak FLOPs
   - ~50-70% on H100 is good

### Benchmark Tasks

Post-training evaluation includes:

- **ARC-Challenge**: Science questions (multiple choice)
- **ARC-Easy**: Easier science questions
- **GSM8K**: Grade school math problems
- **HumanEval**: Python coding tasks
- **MMLU**: Multi-domain multiple choice questions
- **ChatCORE**: Aggregate chat benchmark

### Report Card

After training, check `report.md` for a complete summary:

```bash
cat report.md
```

Example metrics table:

```
| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |
```

---

## Advanced Tips

### Weights & Biases Integration

To track training with W&B:

```bash
# 1. Login to wandb
wandb login

# 2. Set WANDB_RUN environment variable
WANDB_RUN=my_experiment bash speedrun.sh
```

### Resuming Training

Nanochat saves checkpoints but doesn't include resume functionality by default. To resume:

1. Modify the training script to load from checkpoint
2. Use the `load_checkpoint()` function from `checkpoint_manager.py`
3. Restore optimizer state and adjust `num_iterations`

### Distributed Training Beyond Single Node

Currently, nanochat is designed for single-node multi-GPU. For multi-node:

1. Modify `torchrun` command with master address
2. Update `compute_init()` in `nanochat/common.py`
3. Test with NCCL backend

---

## Troubleshooting

### Out of Memory (OOM)

**Solution**: Reduce `--device_batch_size`:
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --device_batch_size=16  # Try 16, 8, 4, 2, 1
```

### Slow Training

**Check**:
1. MFU in logs (should be 50-70% on H100)
2. Data loading (should be fast, <1% of step time)
3. Compilation (first few steps are slow, then speeds up)

### Model Quality Issues

**Possible causes**:
1. Insufficient training data (increase data shards)
2. Learning rate too high/low (check warmup/warmdown curves)
3. Not enough training iterations (check tokens:params ratio)

### Import Errors

**Solution**: Make sure virtual environment is activated:
```bash
source .venv/bin/activate
```

---

## Summary

### Training Pipeline Summary

```
1. Tokenizer (~10 mins)
   ↓
2. Base Model (~2.5 hours for d20)
   → Saved to: ~/.cache/nanochat/base_checkpoints/d20/
   ↓
3. Midtraining (~1 hour)
   → Saved to: ~/.cache/nanochat/mid_checkpoints/d20/
   ↓
4. SFT (~30 mins)
   → Saved to: ~/.cache/nanochat/chatsft_checkpoints/d20/
   ↓
5. (Optional) RL
   → Saved to: ~/.cache/nanochat/chatrl_checkpoints/d20/
```

### Quick Commands Cheat Sheet

```bash
# Train everything (d20, ~4 hours, 8xH100)
bash speedrun.sh

# Chat with trained model (web UI)
python -m scripts.chat_web

# Chat (CLI)
python -m scripts.chat_cli -i sft

# Train larger model (d26, ~12 hours)
python -m nanochat.dataset -n 450
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16

# Train on single GPU
python -m scripts.base_train -- --depth=20

# Train on CPU (tiny model)
python -m scripts.base_train --depth=4 --device_batch_size=1 --num_iterations=20
```

---

## Additional Resources

- **Main README**: `README.md` - Overview and introduction
- **Discussions**: 
  - [Introducing nanochat](https://github.com/karpathy/nanochat/discussions/1)
  - [Infusing identity](https://github.com/karpathy/nanochat/discussions/139)
  - [Adding abilities](https://github.com/karpathy/nanochat/discussions/164)
- **Demo**: [nanochat.karpathy.ai](https://nanochat.karpathy.ai/) - Try d32 model
- **Code**: Only ~8K lines in 45 files, designed to be readable and hackable

---

## License

MIT License - See `LICENSE` file

## Author

Andrej Karpathy - [GitHub](https://github.com/karpathy)

---

*Generated on: October 29, 2025*
*Repository: [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)*

