# Sequence-to-Set Model

A PyTorch implementation of **high-recall sequence-to-set prediction models** for multi-label token prediction tasks. This project tackles the challenge of predicting complete sets of relevant output tokens from input sequences, prioritizing recall over precision to ensure no important tokens are missed.

## 🎯 The Goal: High-Recall Set Prediction

**The Problem**: Given a sequence of input tokens, predict ALL relevant output tokens that should be in the result set.

**The Challenge**: Traditional sequence models predict ordered outputs, but many real-world tasks require:
- **Unordered sets** (not sequences) as output
- **High recall** - missing relevant items is worse than including extra ones
- **Variable-length targets** (1-9 tokens) but **fixed-size predictions** for consistent interfaces

**Our Solution**: Three neural architectures that:
1. 🎯 **Maximize Recall** - Better to predict extra tokens than miss important ones
2. 🔄 **Order-Invariant** - Output is treated as a set, not a sequence  
3. 📊 **Multi-Label Classification** - Each token gets an independent prediction score
4. ⚖️ **Weighted Loss** - Heavy penalty for false negatives, light penalty for false positives

## 💡 Why This Matters

**Example Use Cases**:
- **Medical Coding**: Input diagnosis → Predict ALL relevant drug classes
- **Recommendation Systems**: Input user preferences → Suggest ALL relevant items  
- **Code Completion**: Input context → Predict ALL possible completions
- **Tag Prediction**: Input content → Generate ALL relevant tags

**The Key Insight**: In many domains, **missing a relevant prediction is much worse than including an irrelevant one**. Users can filter out extras, but they can't recover what wasn't suggested.

## 🤖 Three Model Architectures

I've implemented **three different approaches** to tackle this sequence-to-set challenge:

### 1. 🟢 MLP Model (`src/mlp.py`) - The Baseline
- **Architecture**: Simple multi-layer perceptron
- **Approach**: Concatenate input embeddings → Feed through FC layers → Direct set prediction
- **Strengths**: Fast, simple, good baseline performance
- **Best For**: When input relationships are straightforward

### 2. 🟡 LSTM Model (`src/lstm.py`) - The Sequence Expert  
- **Architecture**: Encoder-decoder with LSTM layers
- **Approach**: Encode inputs → Autoregressively decode → Max-pool across time steps
- **Strengths**: Captures sequential patterns, handles temporal relationships
- **Best For**: When inputs have sequential structure or dependencies

### 3. 🔴 Transformer Model (`src/transformer.py`) - The Attention Master
- **Architecture**: Transformer encoder-decoder with self-attention
- **Approach**: Self-attend over inputs → Cross-attend during decoding → Max-pool outputs  
- **Strengths**: Complex pattern recognition, attention-based relationships
- **Best For**: When inputs have complex interdependencies requiring attention

## 🔍 The High-Recall Strategy

All three models use the same **recall-optimization strategy**:

```python
# Weighted BCE Loss - Heavily penalize missing tokens (false negatives)
BETA_FN = 3.0  # False negative penalty weight (β > 1 = high recall bias)
pos_weight = torch.full((vocab_size,), BETA_FN)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Translation**: "Missing a relevant token is 3x worse than including an irrelevant one"

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Generate Synthetic Dataset

```bash
python src/synth_dataset_gen.py
```

This creates a synthetic dataset with learnable patterns including:
- Fibonacci-like sequences
- Arithmetic progressions  
- Geometric patterns
- Digit expansions
- Modular arithmetic

### Train Models

```bash
# Train MLP model
python src/mlp.py

# Train LSTM model  
python src/lstm.py

# Train Transformer model
python src/transformer.py
```

### Configuration

Key hyperparameters (configurable in each model file):

```python
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
D_MODEL = 1024
NUM_INPUTS = 2      # Number of input tokens
NUM_OUTPUTS = 6     # Number of output predictions
DROPOUT = 0.2
BETA_FN = 3.0       # False negative penalty weight
```

## 📊 Training & Evaluation

### Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives) 
- **Loss**: Weighted BCE loss with higher penalty for false negatives
