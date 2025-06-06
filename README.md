## 🤖 Models

### 1. MLP Model (`src/mlp.py`)
- **Architecture**: Simple multi-layer perceptron
- **Input Processing**: Concatenated token embeddings → FC layers
- **Output**: Direct multi-label classification
- **Best For**: Simple patterns, fast training, baseline model

### 2. LSTM Model (`src/lstm.py`)
- **Architecture**: Encoder-decoder with LSTM layers
- **Input Processing**: LSTM encoder processes input pair
- **Output**: Autoregressive decoder with max-pooling across steps
- **Best For**: Sequential patterns, moderate complexity

### 3. Transformer Model (`src/transformer.py`)
- **Architecture**: Transformer encoder-decoder
- **Input Processing**: Self-attention over input tokens
- **Output**: Autoregressive decoder with causal masking + max-pooling
- **Best For**: Complex patterns, attention-based relationships

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

### Output Files
- `models/model_latest.pth`: Latest trained model
- `models/model_{timestamp}.pth`: Timestamped model checkpoint
- `logging/loss_latest.png`: Training curves (loss, precision, recall)

### Sample Training Output
