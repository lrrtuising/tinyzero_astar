# TinyZero A*PO Implementation

A complete implementation of TinyZero using A* Policy Optimization (A*PO) for multiplication task reproduction of DeepSeek R1 Zero.

## Overview

This implementation combines A* search for rollout generation with policy gradient optimization, using a learned value function as the heuristic in A* search. Unlike GRPO which uses rejection sampling, A*PO uses A* search with value function guidance to generate diverse, high-quality rollouts.

## Key Features

- **A* Search**: Guided rollout generation using value function as heuristic
- **Policy Optimization**: PPO-style policy gradient training
- **Value Function**: Neural network for A* heuristic estimation
- **FSDP Support**: PyTorch FSDP for distributed training (up to 8 GPUs)
- **Pure RL**: Training from base model without supervised fine-tuning
- **Multiplication Task**: Focused on arithmetic reasoning

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Base Model    │    │  Value Model    │    │   A* Search     │
│   (Qwen2.5-3B)  │───▶│  (Heuristic)    │───▶│   (Rollouts)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Policy Gradient │    │  Value Loss     │    │   Rewards       │
│   (A*PO)        │    │   (MSE)         │    │ (Accuracy)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Local Testing (Small Scale)

```bash
# Test with small model and dataset
python train.py --config local
```

### Full Training

```bash
# Train with full configuration
python train.py --config training
```

### Distributed Training

```bash
# Multi-GPU training with FSDP
torchrun --nproc_per_node=4 train.py --config training --distributed
```

### Evaluation Only

```bash
# Evaluate a trained model
python train.py --config training --eval-only --checkpoint checkpoints/checkpoint_iter_100.pt
```

## Configuration

The system supports two main configurations:

- **Local**: Small scale for testing (1.5B model, 100 problems)
- **Training**: Full scale for production (3B model, 1000+ problems)

Key parameters can be adjusted in `config.py`:

```python
# Model configuration
model_name = "Qwen/Qwen2.5-3B"
max_seq_length = 512

# Training configuration
num_iterations = 100
batch_size = 4
learning_rate = 1e-5

# A* search configuration
num_rollouts_per_problem = 4
beam_width = 8
max_search_depth = 200

# Value model configuration
hidden_size = 512
update_frequency = 5
```

## File Structure

```
tinyzero_astar/
├── config.py                 # Configuration management
├── data/
│   └── generation.py         # Multiplication problem generation
├── model/
│   ├── model.py             # Base model loading
│   └── fsdp_utils.py        # Distributed training utilities
├── rollout/
│   ├── value_model.py       # Value function for A* heuristic
│   ├── astar_search.py      # A* search algorithm
│   └── generator.py         # Rollout generation
├── training/
│   ├── reward.py            # Reward computation
│   ├── astar_po.py          # A*PO algorithm
│   └── optimizer.py         # Optimizer utilities
├── eval/
│   └── evaluate.py          # Evaluation functions
├── train.py                 # Main training script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Algorithm Details

### A* Search

The A* search algorithm uses:
- **g(n)**: Cumulative log probability (cost from start)
- **h(n)**: Value model estimate (heuristic to goal)
- **f(n)**: f(n) = g(n) + h(n) (total estimated cost)

### A*PO Training

1. **Rollout Generation**: Use A* search to generate diverse rollouts
2. **Reward Computation**: Calculate accuracy and format rewards
3. **Advantage Estimation**: Compute advantages using value function
4. **Policy Update**: PPO-style policy gradient update
5. **Value Update**: MSE loss for value function

### Key Differences from GRPO

- **GRPO**: Uses rejection sampling to generate rollouts
- **A*PO**: Uses A* search with value function guidance
- **Benefit**: More diverse, higher-quality rollouts with better exploration

## Expected Results

- **Initial (iteration 0)**: ~0-5% accuracy (random guessing)
- **After 5-10 iterations**: ~20-40% accuracy (model starts learning)
- **After 50+ iterations**: ~80-95% accuracy (target performance)

## Monitoring

The training script provides detailed logging:
- Policy loss, value loss, rewards, advantages
- Accuracy, format correctness
- Learning rates, sample generations
- A* search statistics (nodes explored, etc.)

