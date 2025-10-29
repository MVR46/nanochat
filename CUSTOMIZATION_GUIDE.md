# NanoChat Training Customization Guide

A comprehensive guide to customizing every aspect of nanochat training - from model architecture to data mixtures, learning rates, and personality.

---

## Table of Contents

1. [Configuration Methods](#configuration-methods)
2. [Model Architecture](#model-architecture)
3. [Training Hyperparameters](#training-hyperparameters)
4. [Data Mixture Customization](#data-mixture-customization)
5. [Customizing Model Personality](#customizing-model-personality)
6. [Learning Rate Schedules](#learning-rate-schedules)
7. [Optimizer Settings](#optimizer-settings)
8. [Memory & Performance Tuning](#memory--performance-tuning)
9. [Evaluation & Logging](#evaluation--logging)
10. [Adding New Capabilities](#adding-new-capabilities)
11. [Advanced Customizations](#advanced-customizations)

---

## Configuration Methods

Nanochat uses a "Poor Man's Configurator" that allows three ways to override training parameters:

### Method 1: Command-Line Arguments

Override any parameter directly from the command line:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --device_batch_size=16 \
    --embedding_lr=0.3 \
    --num_iterations=5000
```

**Syntax**: `--parameter_name=value`

### Method 2: Configuration Files

Create a Python config file with your settings:

```python
# config/my_experiment.py
depth = 26
device_batch_size = 16
embedding_lr = 0.3
num_iterations = 5000
warmup_ratio = 0.1
```

Run with:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train config/my_experiment.py
```

### Method 3: Modify Training Scripts Directly

Edit the default values directly in the training script (e.g., `scripts/base_train.py`):

```python
# User settings (lines 33-62 in base_train.py)
run = "my_experiment"  # wandb run name
depth = 26             # model depth
device_batch_size = 16 # per-device batch size
# ... etc
```

### Configuration Priority

The configurator applies overrides in this order (later overrides earlier):

1. **Script defaults** (hardcoded in the .py file)
2. **Config file** (if provided)
3. **Command-line args** (highest priority)

---

## Model Architecture

### Model Size (Depth)

The `depth` parameter controls the number of transformer layers, which is the primary way to scale model size:

```bash
# Small model: ~200M parameters
--depth=12

# Default speedrun: 561M parameters
--depth=20

# Medium model: ~1.2B parameters
--depth=26

# Large model: 1.9B parameters
--depth=32
```

**Derived dimensions** (automatically calculated):

```python
num_layers = depth
model_dim = depth * 64      # Aspect ratio of 64
num_heads = max(1, (model_dim + 127) // 128)  # Head dim of 128
num_kv_heads = num_heads    # GQA ratio 1:1 (no grouping by default)
```

**Examples**:
- d12: 12 layers, 768 dim, 6 heads → ~200M params
- d20: 20 layers, 1280 dim, 10 heads → 561M params
- d26: 26 layers, 1664 dim, 13 heads → ~1.2B params
- d32: 32 layers, 2048 dim, 16 heads → 1.9B params

### Context Length

Control the maximum sequence length:

```bash
# Default
--max_seq_len=2048

# Shorter (saves memory)
--max_seq_len=1024

# Longer (more memory)
--max_seq_len=4096
```

**Memory impact**: Quadratic in sequence length due to attention mechanism.

### Grouped Query Attention (GQA)

Currently, nanochat uses standard multi-head attention (GQA ratio 1:1). To enable GQA, modify `scripts/base_train.py`:

```python
# Around line 92, change:
num_kv_heads = num_heads  # default: no grouping

# To enable GQA (e.g., 4:1 ratio):
num_kv_heads = max(1, num_heads // 4)  # 4 query heads per KV head
```

**Benefits**: Reduces memory usage for KV cache during inference.

### Custom Architecture

For complete control, modify the model configuration directly:

```bash
# In base_train.py, around line 108:
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,      # Custom number of layers
    n_head=num_heads,        # Custom number of heads
    n_kv_head=num_kv_heads,  # Custom KV heads
    n_embd=model_dim         # Custom embedding dimension
)
```

---

## Training Hyperparameters

### Training Duration

Three ways to control how long training runs (in order of precedence):

#### 1. Explicit Iterations

```bash
--num_iterations=5000  # Train for exactly 5000 steps
```

#### 2. Target FLOPs

```bash
--target_flops=1e19  # Train until 1e19 FLOPs consumed
```

**Use case**: Scaling laws experiments where you want to control compute budget.

#### 3. Data-to-Parameter Ratio (Chinchilla)

```bash
--target_param_data_ratio=20  # Default: Chinchilla optimal (20:1 tokens:params)
```

**Examples**:
- 20:1 (default) - Chinchilla optimal, balanced
- 10:1 - Faster training, potentially underfit
- 40:1 - Longer training, potentially overfit
- 100:1 - Massively overtrained (not recommended)

**Calculation**:
```python
target_tokens = target_param_data_ratio * num_params
num_iterations = target_tokens // total_batch_size
```

### Batch Size

Two separate batch size parameters:

#### Device Batch Size (per GPU)

```bash
--device_batch_size=32  # Default for 80GB GPUs
```

**Tune this based on VRAM**:
- 80GB: 32 (default)
- 40GB: 16 or 8
- 24GB: 4 or 2
- 16GB: 1 or 2

#### Total Batch Size (across all GPUs)

```bash
--total_batch_size=524288  # Total tokens per optimization step
```

**Default**: 524,288 tokens (~512K)

**How it works**:
```python
tokens_per_gpu = device_batch_size * max_seq_len
world_tokens = tokens_per_gpu * num_gpus
grad_accum_steps = total_batch_size // world_tokens
```

The script automatically uses gradient accumulation to reach your target total batch size.

**Examples**:
```bash
# 8 GPUs, device_batch_size=32, max_seq_len=2048
# → 32 * 2048 = 65,536 tokens/GPU
# → 65,536 * 8 = 524,288 tokens/step
# → No gradient accumulation needed

# 4 GPUs, device_batch_size=16, max_seq_len=2048
# → 16 * 2048 = 32,768 tokens/GPU
# → 32,768 * 4 = 131,072 tokens/step
# → Need 4x gradient accumulation to reach 524,288
```

### Learning Rates

Nanochat uses **two separate optimizers** with different learning rates:

#### AdamW (Embeddings and Output Layer)

```bash
--embedding_lr=0.2      # Input embedding layer (default)
--unembedding_lr=0.004  # Output layer (lm_head) (default)
--weight_decay=0.0      # Weight decay for Adam (default: disabled)
```

#### Muon (Transformer Layers)

```bash
--matrix_lr=0.02  # All transformer matrix parameters (default)
```

**Typical ranges**:
- `embedding_lr`: 0.1 - 0.5
- `unembedding_lr`: 0.001 - 0.01
- `matrix_lr`: 0.01 - 0.05

**Example custom configuration**:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --embedding_lr=0.25 \
    --unembedding_lr=0.005 \
    --matrix_lr=0.015 \
    --weight_decay=0.01
```

### Learning Rate Schedule

Control the LR warmup/warmdown behavior:

```bash
--warmup_ratio=0.0    # Fraction of training for LR warmup (default: no warmup)
--warmdown_ratio=0.2  # Fraction of training for LR decay (default: 20%)
--final_lr_frac=0.0   # Final LR as fraction of initial (default: 0, decay to zero)
```

**Schedule visualization**:
```
LR Multiplier
    1.0 ┤     ╭─────────────╮
        │    ╱               ╲
        │   ╱                 ╲
        │  ╱                   ╲
    0.0 ┤─╯                     ╰─
        └─────────────────────────
        0%   warmup   80%   100%
                           warmdown
```

**Example: Add warmup**:

```bash
--warmup_ratio=0.05 \
--warmdown_ratio=0.2 \
--final_lr_frac=0.1  # Don't decay all the way to zero
```

### Gradient Clipping

```bash
--grad_clip=1.0  # Clip gradients to max norm of 1.0 (default)
--grad_clip=0.0  # Disable gradient clipping
```

**Typical range**: 0.5 - 2.0

---

## Data Mixture Customization

### Base Model Training Data

Base training uses the FineWeb dataset. Control how much data to download:

```bash
# Each shard = ~250M characters
python -m nanochat.dataset -n 240  # Download 240 shards (~60B chars)
```

**Calculation for required shards**:

```python
num_params = 561e6  # For d20
tokens_needed = 20 * num_params  # Chinchilla: 20x ratio
chars_needed = tokens_needed * 4.8  # ~4.8 chars/token
shards_needed = chars_needed / 250e6  # ~250M chars/shard

# For d20: 561M * 20 * 4.8 / 250M ≈ 216 shards
```

**Available shards**: 1,822 total in FineWeb (can train up to ~100B parameter models)

### Midtraining Data Mixture

Customize the task mixture in `scripts/mid_train.py` (lines 98-106):

```python
train_dataset = TaskMixture([
    SmolTalk(split="train"),                          # 460K conversation rows
    MMLU(subset="auxiliary_train", split="train"),    # 100K multiple choice
    GSM8K(subset="main", split="train"),              # 8K math problems
    CustomJSON(filepath=identity_conversations_filepath),  # 1K identity convos
    SimpleSpelling(size=200000, split="train"),       # 200K spelling tasks
    SpellingBee(size=80000, split="train"),           # 80K letter counting
])
```

#### Adjust Task Proportions

```python
# Give more weight to identity by including it multiple times:
train_dataset = TaskMixture([
    SmolTalk(split="train"),
    MMLU(subset="auxiliary_train", split="train"),
    GSM8K(subset="main", split="train"),
    CustomJSON(filepath=identity_conversations_filepath),
    CustomJSON(filepath=identity_conversations_filepath),  # 2x
    CustomJSON(filepath=identity_conversations_filepath),  # 3x
    SimpleSpelling(size=200000, split="train"),
    SpellingBee(size=80000, split="train"),
])
```

#### Remove Tasks

```python
# Train only on conversations (no math/spelling):
train_dataset = TaskMixture([
    SmolTalk(split="train"),
    CustomJSON(filepath=identity_conversations_filepath),
])
```

#### Limit Dataset Size

```python
# Use smaller subsets for faster experimentation:
train_dataset = TaskMixture([
    SmolTalk(split="train", stop=50000),    # Only first 50K rows
    GSM8K(subset="main", split="train"),
])
```

#### Add New Tasks

```python
# Add your own JSON conversation file:
from tasks.customjson import CustomJSON

my_task_path = os.path.join(base_dir, "my_custom_data.jsonl")
train_dataset = TaskMixture([
    SmolTalk(split="train"),
    CustomJSON(filepath=my_task_path),  # Your custom task
])
```

### SFT Data Mixture

Customize SFT training data in `scripts/chat_sft.py` (lines 84-92):

```python
train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"),             # 2.3K
    ARC(subset="ARC-Challenge", split="train"),        # 1.1K
    GSM8K(subset="main", split="train"),               # 8K
    SmolTalk(split="train", stop=10_000),              # 10K
    CustomJSON(filepath=identity_conversations_filepath),  # 1K
    SimpleSpelling(size=300, split="train"),           # 300
    SpellingBee(size=300, split="train"),              # 300
])
```

**Example: Focus on math**:

```python
train_ds = TaskMixture([
    GSM8K(subset="main", split="train"),    # 8K rows
    GSM8K(subset="main", split="train"),    # Include 2x for emphasis
    SmolTalk(split="train", stop=5_000),    # Reduce conversations
])
```

**Example: Remove certain capabilities**:

```python
# No spelling tasks:
train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"),
    ARC(subset="ARC-Challenge", split="train"),
    GSM8K(subset="main", split="train"),
    SmolTalk(split="train", stop=10_000),
    CustomJSON(filepath=identity_conversations_filepath),
    # Removed: SimpleSpelling and SpellingBee
])
```

### Control Number of Epochs

In midtraining and SFT, you can control training duration:

```bash
# Midtraining: explicit iterations
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \
    --num_iterations=1000

# SFT: number of epochs over the data
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --num_epochs=2  # Default is 1
```

---

## Customizing Model Personality

### Creating Synthetic Identity Data

Use `dev/gen_synthetic_data.py` as a template to generate custom personality data.

#### Step 1: Define Your Identity

Edit the prompt in the script (around line 49):

```python
prompt = r"""
I want to generate synthetic data for an LLM to teach it about its identity.

The name of the LLM is "MyBot". It is a helpful assistant created by 
MyCompany in 2025. It specializes in technical support and programming help.
MyBot is patient, thorough, and always provides code examples when relevant.

Key traits:
- Professional but friendly tone
- Focuses on practical solutions
- Admits when it doesn't know something
- Prefers step-by-step explanations

Please create a natural conversation between a User and MyBot.
"""
```

#### Step 2: Add Diversity Seeds

Crucial for quality! Add diverse conversation starters:

```python
user_first_prompts = """
Hello, can you help me debug my Python code?
I'm stuck on this React error
What's the best way to optimize database queries?
Can you explain how async/await works?
I need help with deployment to AWS
How do I set up CI/CD for my project?
""".strip().split("\n")
```

**Pro tip**: Add 100-300 diverse starters covering:
- Different greeting styles
- Various technical topics
- Different skill levels
- Edge cases and unusual requests
- Multiple languages (if desired)

#### Step 3: Generate the Data

```bash
# Make sure you have an OpenRouter API key
echo "your-api-key-here" > openroutertoken.txt

# Generate conversations
python dev/gen_synthetic_data.py
```

**Output**: Creates `~/.cache/nanochat/identity_conversations.jsonl`

#### Step 4: Use Custom Data

The generated file is automatically loaded in midtraining and SFT via:

```python
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
CustomJSON(filepath=identity_conversations_filepath)
```

### Multiple Identity Files

Add multiple personality aspects:

```python
# In mid_train.py or chat_sft.py:
train_dataset = TaskMixture([
    SmolTalk(split="train"),
    CustomJSON(filepath=os.path.join(base_dir, "identity_core.jsonl")),
    CustomJSON(filepath=os.path.join(base_dir, "technical_expert.jsonl")),
    CustomJSON(filepath=os.path.join(base_dir, "humor_style.jsonl")),
    # ... other tasks
])
```

### Manual Conversation Curation

Create conversations manually in JSONL format:

```json
{"messages": [{"role": "user", "content": "Who created you?"}, {"role": "assistant", "content": "I'm MyBot, created by MyCompany."}]}
{"messages": [{"role": "user", "content": "What can you help with?"}, {"role": "assistant", "content": "I specialize in programming help, debugging, and technical support."}]}
```

**Format requirements**:
- Each line is a complete JSON object with a `messages` array
- Messages alternate between "user" and "assistant" roles
- First message must be from "user"

---

## Learning Rate Schedules

### Base Training Schedule

Default schedule (in `scripts/base_train.py`, lines 157-166):

```python
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    
    if it < warmup_iters:
        return (it + 1) / warmup_iters          # Linear warmup
    elif it <= num_iterations - warmdown_iters:
        return 1.0                               # Constant
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac  # Linear decay
```

### Custom Schedules

#### Cosine Decay

```python
import math

def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    else:
        # Cosine decay from 1.0 to final_lr_frac
        progress = (it - warmup_iters) / (num_iterations - warmup_iters)
        return final_lr_frac + (1.0 - final_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))
```

#### Step Decay

```python
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it < num_iterations * 0.5:
        return 1.0
    elif it < num_iterations * 0.8:
        return 0.5
    else:
        return 0.1
```

#### No Decay (Constant)

```bash
--warmup_ratio=0.0 \
--warmdown_ratio=0.0 \
--final_lr_frac=1.0
```

### Midtraining Schedule

Midtraining uses a simpler schedule (lines 163-165 in `mid_train.py`):

```python
def get_lr_multiplier(progress):
    # Constant for first 80%, then linear decay
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2
```

### SFT Schedule

SFT uses simple linear decay (lines 164-166 in `chat_sft.py`):

```python
def get_lr_multiplier(it):
    # Linear decay from 1.0 to 0.0
    return 1.0 - it / num_iterations
```

**Customize SFT LR**:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --init_lr_frac=0.1  # Start at 10% of base LR (default: 0.02 = 2%)
```

---

## Optimizer Settings

### Muon Momentum Schedule

Muon optimizer has a momentum warmup (lines 169-172 in `base_train.py`):

```python
def get_muon_momentum(it):
    frac = min(it / 300, 1)  # Warmup over 300 steps
    momentum = (1 - frac) * 0.85 + frac * 0.95  # 0.85 → 0.95
    return momentum
```

**Customize momentum**:

```python
def get_muon_momentum(it):
    frac = min(it / 500, 1)  # Slower warmup
    momentum = (1 - frac) * 0.90 + frac * 0.98  # Higher final momentum
    return momentum
```

### Weight Decay

```bash
--weight_decay=0.0  # Default: no weight decay
--weight_decay=0.01  # Light regularization
--weight_decay=0.1   # Stronger regularization
```

Applied only to AdamW parameters (embeddings and output layer).

### Gradient Clipping

```bash
--grad_clip=1.0  # Default
--grad_clip=0.5  # More aggressive clipping
--grad_clip=5.0  # Looser clipping
--grad_clip=0.0  # Disable clipping (not recommended)
```

---

## Memory & Performance Tuning

### Reduce Memory Usage

#### 1. Decrease Batch Size

```bash
--device_batch_size=16  # From 32
--device_batch_size=8   # Even more aggressive
```

#### 2. Shorter Context

```bash
--max_seq_len=1024  # From 2048
```

#### 3. Smaller Model

```bash
--depth=16  # From 20
```

#### 4. Mixed Precision

Already enabled by default (bfloat16 on CUDA). To use float32:

```bash
# In mid_train.py or chat_sft.py:
--dtype=float32
```

**Warning**: float32 uses 2x memory and is slower.

### Improve Training Speed

#### 1. Increase Batch Size

```bash
# If you have spare VRAM:
--device_batch_size=64  # From 32
```

#### 2. Compile Model

Already enabled by default:
```python
model = torch.compile(model)
```

To disable (for debugging):
```python
# Comment out this line in the training script
# model = torch.compile(model)
```

#### 3. Reduce Evaluation Frequency

```bash
--eval_every=500           # From 250 (base training)
--core_metric_every=5000   # From 2000 (base training)
--sample_every=5000        # From 2000 (base training)
```

#### 4. Optimize Data Loading

Set `OMP_NUM_THREADS` (already done in speedrun.sh):
```bash
export OMP_NUM_THREADS=1
```

### Multi-Node Training

Nanochat is designed for single-node. To extend to multi-node:

```bash
# On each node:
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m scripts.base_train
```

**Note**: Requires modifications to `nanochat/common.py` for proper distributed initialization.

---

## Evaluation & Logging

### Weights & Biases

Enable W&B logging:

```bash
# 1. Login (one-time setup)
wandb login

# 2. Set run name
export WANDB_RUN=my_experiment

# 3. Run training
bash speedrun.sh
```

Or directly:
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --run=my_experiment_name
```

### Evaluation Frequency

#### Base Training

```bash
--eval_every=250            # Validation loss every N steps
--core_metric_every=2000    # CORE metric every N steps (-1 = disable)
--sample_every=2000         # Text samples every N steps
--eval_tokens=10485760      # Tokens to use for validation (default: 20*524288)
```

#### Midtraining

```bash
--eval_every=150       # Default for midtraining
--eval_tokens=10485760
```

#### SFT

```bash
--eval_every=100               # Loss evaluation
--eval_metrics_every=200       # Task metrics (MMLU, ARC)
--eval_steps=100               # Number of validation steps
--eval_metrics_max_problems=1024  # Max problems per task
```

### Disable Evaluation (Fastest Training)

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --eval_every=-1 \
    --core_metric_every=-1 \
    --sample_every=-1
```

**Warning**: You won't see any metrics during training!

### Custom Evaluation Samples

Modify the prompts in `scripts/base_train.py` (lines 223-230):

```python
prompts = [
    "The capital of France is",
    "Write a Python function to",
    "Explain quantum computing in simple terms:",
    # Add your custom prompts here
]
```

---

## Adding New Capabilities

### Step 1: Create a Task Class

Create a new file in `tasks/` folder:

```python
# tasks/my_custom_task.py
from tasks.common import Task

class MyCustomTask(Task):
    def __init__(self, split="train", stop=None):
        super().__init__()
        self.split = split
        self.stop = stop
        self._load_data()
    
    def _load_data(self):
        # Load your data here
        # Each item should be a dict with "messages" key
        self.data = [
            {
                "messages": [
                    {"role": "user", "content": "Question 1?"},
                    {"role": "assistant", "content": "Answer 1"}
                ]
            },
            # ... more examples
        ]
        if self.stop:
            self.data = self.data[:self.stop]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

### Step 2: Add to Training Mixture

In `scripts/mid_train.py` or `scripts/chat_sft.py`:

```python
from tasks.my_custom_task import MyCustomTask

train_dataset = TaskMixture([
    SmolTalk(split="train"),
    MyCustomTask(split="train", stop=5000),  # Your custom task
    # ... other tasks
])
```

### Step 3: Create Evaluation Task

For evaluation, create a subclass in `tasks/`:

```python
# tasks/my_eval_task.py
from tasks.common import MultipleChoiceTask

class MyEvalTask(MultipleChoiceTask):
    def __init__(self, split="test"):
        super().__init__()
        self.split = split
        self._load_data()
    
    def _load_data(self):
        # Load your eval data
        # Format: {"question": "...", "choices": ["A", "B", "C", "D"], "answer": "A"}
        pass
    
    def format_question(self, doc):
        return f"Question: {doc['question']}\nChoices: {', '.join(doc['choices'])}"
    
    def format_answer(self, doc):
        return doc['answer']
```

### Step 4: Add to Evaluation

In `scripts/chat_eval.py`, add your task to the eval mixture.

---

## Advanced Customizations

### Custom Model Architecture

Modify `nanochat/gpt.py` to change the transformer architecture:

```python
# Add new attention mechanism
# Add new layer types  
# Modify position encodings
# etc.
```

### Custom Tokenizer

Train a custom tokenizer vocabulary size:

```bash
# Default is 2^16 = 65,536
python -m scripts.tok_train --max_chars=2000000000

# To change vocab size, modify scripts/tok_train.py:
vocab_size = 32768  # From 65536
```

### Custom Data Source

Replace FineWeb with your own pretraining data:

1. Format data as text shards (one file per shard)
2. Place in `~/.cache/nanochat/data_shards/`
3. Update `nanochat/dataset.py` to point to your data
4. Update tokenizer training to use your data

### Resume Training

Currently not supported by default. To add:

```python
# In base_train.py, after model initialization:
if resume_from_checkpoint:
    model_data, optimizer_data, meta = load_checkpoint(checkpoint_dir, step, device, load_optimizer=True)
    model.load_state_dict(model_data)
    # Restore optimizer state
    for opt, opt_data in zip(optimizers, optimizer_data):
        opt.load_state_dict(opt_data)
    start_step = meta['step']
```

### Custom Loss Function

Modify loss calculation in `nanochat/gpt.py` (forward method):

```python
def forward(self, x, y=None):
    # ... forward pass ...
    
    if y is not None:
        # Standard cross-entropy loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Add custom regularization:
        # loss = loss + 0.01 * custom_regularization_term
        
        return loss
    return logits
```

### Dry Run Mode

Test configuration without saving checkpoints:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \
    --dry_run=1 \
    --num_iterations=10  # Just a few steps
```

### Change Base Directory

By default, all artifacts go to `~/.cache/nanochat/`. To change:

```bash
export NANOCHAT_BASE_DIR="/path/to/custom/dir"
```

---

## Example Customization Recipes

### Recipe 1: Fast Experiment (Tiny Model)

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=8 \
    --max_seq_len=512 \
    --device_batch_size=16 \
    --num_iterations=1000 \
    --eval_every=100 \
    --core_metric_every=-1 \
    --sample_every=-1
```

**Time**: ~15 minutes on 8xH100

### Recipe 2: Math Specialist

```bash
# In chat_sft.py, modify data mixture:
train_ds = TaskMixture([
    GSM8K(subset="main", split="train"),
    GSM8K(subset="main", split="train"),
    GSM8K(subset="main", split="train"),  # 3x emphasis on math
    SmolTalk(split="train", stop=2_000),  # Minimal general chat
])

# Train with more epochs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --num_epochs=3
```

### Recipe 3: Large Model (d26)

```bash
# Download more data
python -m nanochat.dataset -n 450 &

# Train base model
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --device_batch_size=16 \
    --target_param_data_ratio=20

# Midtrain
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \
    --device_batch_size=16

# SFT
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --device_batch_size=2
```

**Time**: ~12 hours on 8xH100  
**Cost**: ~$300

### Recipe 4: Custom Personality Bot

```python
# 1. Create identity in dev/gen_synthetic_data.py
prompt = """
Create conversations for a bot named TechHelper that:
- Specializes in Python and web development
- Is enthusiastic about open source
- Uses casual, friendly language
- Often provides code examples
"""

# 2. Generate data
python dev/gen_synthetic_data.py

# 3. Emphasize identity in SFT (chat_sft.py):
train_ds = TaskMixture([
    CustomJSON(filepath=identity_conversations_filepath),
    CustomJSON(filepath=identity_conversations_filepath),
    CustomJSON(filepath=identity_conversations_filepath),  # 3x
    SmolTalk(split="train", stop=5_000),
    # ... other tasks
])
```

### Recipe 5: Single GPU Training

```bash
# No torchrun, just python
python -m scripts.base_train -- \
    --depth=20 \
    --device_batch_size=4 \
    --target_param_data_ratio=20

# Code automatically uses gradient accumulation
# Takes 8x longer: ~4 hours → ~32 hours
```

---

## Troubleshooting Customizations

### Model Doesn't Converge

**Possible causes**:
- Learning rate too high → reduce by 2-5x
- Batch size too small → increase if possible
- Not enough training data → download more shards

### Model Overfits

**Solutions**:
- Add weight decay: `--weight_decay=0.01`
- Train for fewer iterations
- Increase training data diversity
- Use validation loss for early stopping

### OOM After Changing Config

**Solutions**:
- Reduce `--device_batch_size`
- Reduce `--max_seq_len`
- Reduce `--depth`
- Disable gradient checkpointing (if enabled)

### Changes Not Taking Effect

**Check**:
1. Are you using `--` before arguments in torchrun?
2. Is parameter name spelled correctly?
3. Is it in the `config_keys` list in the script?
4. Is it being overridden by a config file?

---

## Summary

**Key Customization Points**:

1. **Model Size**: `--depth` (12, 20, 26, 32)
2. **Training Duration**: `--num_iterations` or `--target_param_data_ratio`
3. **Memory**: `--device_batch_size` and `--max_seq_len`
4. **Learning Rates**: `--embedding_lr`, `--unembedding_lr`, `--matrix_lr`
5. **Data**: Modify `TaskMixture` in training scripts
6. **Personality**: Generate synthetic data with `gen_synthetic_data.py`
7. **Schedule**: `--warmup_ratio`, `--warmdown_ratio`, `--final_lr_frac`

**Quick Reference Table**:

| Goal | Key Parameter | Typical Values |
|------|--------------|----------------|
| Smaller model | `--depth` | 8, 12, 16 |
| Larger model | `--depth` | 26, 32, 40 |
| Less memory | `--device_batch_size` | 8, 4, 2, 1 |
| Faster training | `--num_iterations` | Reduce by 50% |
| Better quality | `--target_param_data_ratio` | 30, 40, 50 |
| LR warmup | `--warmup_ratio` | 0.01 - 0.1 |
| Regularization | `--weight_decay` | 0.01 - 0.1 |

---

**Next Steps**:
- Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for basic usage
- Check [GitHub Discussions](https://github.com/karpathy/nanochat/discussions) for community tips
- Experiment with small models first before scaling up

---

*Generated on: October 29, 2025*  
*Repository: [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)*

