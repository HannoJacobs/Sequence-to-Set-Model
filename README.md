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

## 🔧 Key Implementation Details

### Overall Design Decisions

#### **Fixed Output Size: Always 9 Predictions**
```python
NUM_OUTPUTS = 9     # Number of output predictions
```

**Key Constraint**: All models are designed to make exactly 9 predictions, regardless of how many tokens are actually in the target set (which varies from 1-9). This creates a consistent interface and now perfectly matches the maximum possible target length.

**Why this matters**: Since target sets can have 1-9 tokens and we predict 9, the model now has enough capacity to potentially capture all relevant tokens in even the largest target sets. This reduces the pressure on the model to be overly selective and allows the high-recall strategy to work optimally.

**Implementation enforcement**:

**In `batch_set_metrics()` function** (used during both training and evaluation):
```python
# Line 232 in src/mlp.py, src/lstm.py, src/transformer.py
topk_indices = logits.topk(k, dim=1).indices  # (B, k)
```
Where `k = NUM_OUTPUTS = 9`, meaning we always select exactly the top 9 predicted tokens from the model's output logits.

**Called during training**:
```python
# Line 257 in src/mlp.py (similar in lstm.py, transformer.py)
btp, bfp, bfn = batch_set_metrics(logits.detach(), tgt, k=NUM_OUTPUTS)
```

**Called during evaluation**:
```python
# Line 279 in src/mlp.py (similar in lstm.py, transformer.py) 
btp, bfp, bfn = batch_set_metrics(logits, tgt, k=NUM_OUTPUTS)
```

**In `infer()` function** (standalone inference):
```python
# Line 309 in src/mlp.py (similar in lstm.py, transformer.py)
topk_indices = scores.topk(k).indices  # shape: (k)
```
Where `k = NUM_OUTPUTS = 9` is passed to the inference function.

**During Training Architecture**:
- **MLP**: Outputs logits for entire vocabulary, constraint applied in metrics
- **LSTM**: Decodes for exactly `NUM_OUTPUTS` steps: `for step in range(self.num_outputs)`  
- **Transformer**: Decodes for exactly `NUM_OUTPUTS` steps: `for step in range(self.num_outputs)`

#### **Weighted BCE Loss with Positive Weights**
```python
pos_weight = torch.full((len(tgt_vocab),), BETA_FN, device=DEVICE)
crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(DEVICE)
```

**Purpose**: Creates a recall-biased loss function where missing relevant tokens (false negatives) is penalized more heavily than including irrelevant ones (false positives).

**How it works**: The `pos_weight` parameter in `BCEWithLogitsLoss` multiplies the loss for positive examples by `BETA_FN` (3.0). This means when the model fails to predict a token that should be in the set, the loss is 3x higher than when it incorrectly includes a token that shouldn't be there.

**Result**: The model learns to be "generous" with predictions - it's better to suggest extra tokens than to miss important ones.

---

### MLP Model Specifics

#### **Position-Aware Input Encoding**
```python
flat_embeddings = [e[:, i] for i in range(self.num_inputs)]
flat = torch.cat(flat_embeddings, dim=-1)  # (B, num_inputs * emb_dim)
```

**Purpose**: Explicitly encodes the position of each input token by concatenating their embeddings in order.

**How it works**: Instead of treating inputs as a bag-of-words, this approach preserves positional information by stacking embeddings sequentially. Token 1's embedding is concatenated with Token 2's embedding, creating a fixed-length representation that knows "which token came first."

**Why it matters**: For patterns like sequences or arithmetic progressions, the order of input tokens is crucial for prediction.

---

### LSTM Model Specifics

#### **Encoder-Decoder Architecture**
The LSTM uses a classic encoder-decoder pattern:
- **Encoder**: Processes the input sequence and creates a context representation
- **Decoder**: Autoregressively generates output tokens, using the encoder's context at each step

#### **Greedy Token Selection**
```python
token = logit.argmax(dim=-1, keepdim=True)
```

**Purpose**: During training, select the most likely next token for autoregressive decoding.

**How it works**: At each decoding step, the model predicts a probability distribution over the vocabulary, then greedily selects the token with the highest probability to feed into the next decoding step.

#### **Max-Pool Across Decoding Steps** 
```python
logits = torch.stack(step_logits, dim=1).max(dim=1).values
```

**Purpose**: Convert sequential decoder outputs into a set-based prediction by removing positional information.

**How it works**: The decoder generates `NUM_OUTPUTS` sequential predictions. Max-pooling takes the highest logit score each token received across ALL decoding steps. This means if a token was strongly predicted at any position, it will have a high final score regardless of when it appeared.

**Result**: Transforms ordered sequence prediction into unordered set prediction.

---

### Transformer Model Specifics

#### **Autoregressive Token Appending**
```python
next_tok = torch.cat(
    [next_tok, logit.argmax(dim=-1, keepdim=True)], dim=1
)  # append token: for autoregressive training
```

**Purpose**: Build the decoder input sequence incrementally for autoregressive generation.

**How it works**: At each decoding step, the predicted token is appended to the growing sequence, which becomes the input for the next decoding step. This creates the classic "teacher forcing" during training where the model learns to generate sequences step-by-step.

#### **Order-Invariant Max Pooling**
```python
logits = torch.stack(step_logits, dim=1).max(dim=1).values  # (B, |V|)
```

**Purpose**: Convert the transformer's sequential outputs into set predictions by removing order dependencies.

**How it works**: Similar to the LSTM, this takes the maximum logit score each vocabulary token achieved across all decoding positions. The final prediction score for each token is its highest score from any decoding step.

**Key insight**: This pooling operation is what transforms these sequence-to-sequence architectures into sequence-to-set models, making them suitable for multi-label classification tasks where order doesn't matter.

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
NUM_OUTPUTS = 9     # Number of output predictions
DROPOUT = 0.2
BETA_FN = 3.0       # False negative penalty weight
```

## 📊 Training & Evaluation

### Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives) 
- **Loss**: Weighted BCE loss with higher penalty for false negatives