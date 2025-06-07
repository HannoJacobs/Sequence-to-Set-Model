# pylint: disable=C3001,R0914,R0913,R0917,C0115,C0413,C0116,C0301,C0103
"""
- Simple multi-layer perceptron (MLP) model for multi-label prediction of output set elements.
- Input: a pair of input sequence tokens.
- Output: a set of one or more output set elements likely to be relevant.
- Treats the task as a set prediction problem.
- Optimized to maximize recall (prefers false positives over false negatives).
- Always predicts NUM_OUTPUTS tokens at the output
"""
import os
import time
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATA_PATH = "Datasets/dataset.csv"

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
D_MODEL = 1024
NUM_INPUTS = 2
NUM_OUTPUTS = 9
DROPOUT = 0.2
PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<bos>", "<eos>"

# Weight for false negatives in set-based loss (β > α biases recall)
BETA_FN = 3.0  # 1.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  device = {DEVICE}")


# -------------------- tokenisation helpers --------------------
def tokenize(text: str) -> list[str]:
    """Tokenize by spaces only, preserving all content and case."""
    return text.split()


def build_vocab(texts: list[str], min_freq: int = 1) -> tuple[dict, dict]:
    """Builds vocab and inverse vocab from texts, filtering by min_freq."""
    frequency: dict[str, int] = {}
    for line in texts:
        for tok in tokenize(line):
            frequency[tok] = frequency.get(tok, 0) + 1
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, BOS_TOKEN: 2, EOS_TOKEN: 3}
    for tok in sorted(frequency):
        if frequency[tok] >= min_freq:
            vocab.setdefault(tok, len(vocab))
    inv_vocab = {i: w for w, i in vocab.items()}
    return vocab, inv_vocab


def build_target_vocab(texts: list[str], min_freq: int = 1) -> tuple[dict, dict]:
    """Builds target vocab without pre-inserting special tokens."""
    frequency: dict[str, int] = {}
    for line in texts:
        for tok in tokenize(line):
            frequency[tok] = frequency.get(tok, 0) + 1
    vocab = {}  # Start with an empty vocab
    for tok in sorted(frequency):
        if frequency[tok] >= min_freq:
            # Assign index based on current vocab size
            vocab[tok] = len(vocab)
    inv_vocab = {i: w for w, i in vocab.items()}
    return vocab, inv_vocab


def encode(tokens: list[str], vocab: dict) -> list[int]:
    """Encodes a list of tokens into their corresponding vocab indices."""
    unk = vocab[UNK_TOKEN]
    return [vocab.get(t, unk) for t in tokens]


# -------------------- dataset --------------------
class TranslationDataset(Dataset):
    """
    Returns
        src       – LongTensor (2,)           two input sequence tokens
        tgt_multi – FloatTensor (|V_out|,)   multi-hot output set vector
    """

    def __init__(self, df_: pd.DataFrame, source_vocab: dict, target_vocab: dict):
        self.X, self.Y = [], []
        v_size = len(target_vocab)
        for src_sentence, tgt_sentence in zip(df_["src"], df_["target"]):
            input_ids = encode(tokenize(src_sentence)[:NUM_INPUTS], source_vocab)
            if len(input_ids) < NUM_INPUTS:  # pad / drop rows with <2 tokens
                input_ids += [source_vocab[PAD_TOKEN]] * (NUM_INPUTS - len(input_ids))
            out_vec = torch.zeros(v_size, dtype=torch.float32)
            for tok in tokenize(tgt_sentence):
                if tok in target_vocab:
                    out_vec[target_vocab[tok]] = 1.0
            self.X.append(torch.tensor(input_ids, dtype=torch.long))
            self.Y.append(out_vec)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def collate(batch):
    """simply separate the the batch into src, tgt"""
    xs, ys = zip(*batch)
    X = torch.stack(xs)  # (B,2)
    Y = torch.stack(ys)  # (B, Vocab_size_target)
    return X, Y


class MLPModel(nn.Module):
    """
    Simple multi-layer perceptron (MLP) for multi-label prediction of output set elements from input sequence pairs.

    ---------------------------------------------------------------------------
    Model Purpose and Task Alignment
    ---------------------------------------------------------------------------
    This model is designed to predict a set of output set elements given a pair of input sequence tokens.
    The dataset contains input pairs of tokens and a variable-length set of output set elements (between 1 and 9) as targets.
    The primary goal is to maximize recall: it is preferable to predict extra (false positive) output set elements
    rather than miss any relevant ones (false negatives).

    Why this model is well-suited for the task:

    1. Set-based Output (Order-Invariant Prediction):
        - The output is a set, not a sequence. The order in which output set elements are predicted does not matter.
        - The model produces a score (logit) for every possible output set element, and the top `num_outputs` elements are selected as the prediction set.

    2. Simple, Interpretable Architecture:
        - Each input sequence token is embedded, and the embeddings are concatenated.
        - The concatenated vector is passed through a configurable stack of fully-connected layers with ReLU activations and dropout.
        - The final layer outputs logits for each output set element, representing the model's confidence in the presence of each element.

    3. Multi-Label Loss and Recall Optimization:
        - Uses a multi-label binary cross-entropy (BCE) loss, suitable for predicting multiple output set elements per input.
        - The loss is weighted to penalize false negatives more than false positives, directly optimizing for high recall.

    4. User Interface Alignment:
        - Always predicts a fixed number of output set elements (`num_outputs`), matching the UI requirement where users are presented with a set of suggestions.
        - No recalculation is needed based on user selection; the model's output is a single, high-recall set of output set elements.

    ---------------------------------------------------------------------------
    Model Details
    ---------------------------------------------------------------------------
    - Embeds each input sequence token and concatenates the embeddings.
    - Passes the concatenated vector through a configurable stack of fully-connected layers.
    - Outputs logits for each output set element, suitable for multi-label (set-based) prediction.
    - Optimized for recall by using a weighted BCE loss.

    Args:
        source_vocab (dict): Vocabulary mapping for input sequence tokens.
        target_vocab (dict): Vocabulary mapping for output set elements.
        emb_dim (int): Embedding dimension for input sequence tokens.
        hidden (list[int]): List of hidden layer sizes for the MLP.
        p_drop (float): Dropout probability.
        num_inputs (int): Number of input sequence tokens.

    Returns:
        logits (Tensor): Tensor of shape (B, |V|), where each value is the
            logit for that output set element, suitable for multi-label (set-based) prediction.
    """

    def __init__(
        self,
        source_vocab: dict,
        target_vocab: dict,
        emb_dim: int = 128,
        hidden: list[int] = None,
        p_drop: float = 0.2,
        num_inputs: int = 2,
    ):
        if hidden is None:
            hidden = [256, 128]

        super().__init__()
        self.num_inputs = num_inputs
        # Embedding layer for input sequence tokens
        self.embed = nn.Embedding(
            num_embeddings=len(source_vocab),
            embedding_dim=emb_dim,
            padding_idx=source_vocab[PAD_TOKEN],
        )

        # Build the MLP layers
        layers = []
        in_dim = emb_dim * self.num_inputs  # concat of the input embeddings
        for h in hidden:
            layers += [
                nn.Linear(in_features=in_dim, out_features=h),
                nn.ReLU(),
                nn.Dropout(p=p_drop),
            ]
            in_dim = h
        layers.append(
            nn.Linear(in_features=in_dim, out_features=len(target_vocab))
        )  # logits
        self.mlp = nn.Sequential(*layers)

    def forward(self, src):  # src: (B, num_inputs)
        # Embed each input sequence token in the input pair
        e = self.embed(src)  # (B, num_inputs, emb_dim)

        # flat_embeddings = [[B, emb_dim] , [B, emb_dim]] # for num_inputs=2
        flat_embeddings = [e[:, i] for i in range(self.num_inputs)]

        # Stack the embeddings for all input sequence tokens
        # token_1 on top of token_2 in one long row. Then have B number of them
        flat = torch.cat(flat_embeddings, dim=-1)  # (B, num_inputs * emb_dim)

        return self.mlp(flat)  # (B,|V_out|)


# -------------------- recall-oriented loss --------------------
def custom_loss(logits, targets, crit_):
    return crit_(logits, targets)


def batch_set_metrics(logits, targets, k: int):
    # logits.shape: (B, Vocab_size_target)
    # targets.shape: (B, Vocab_size_target)

    # Ensure k is not larger than the number of classes
    k = min(k, logits.size(1))
    pred = torch.zeros_like(targets).bool()  # (B, Vocab_size_target)
    if k > 0:  # if k is 0, no predictions are made
        topk_indices = logits.topk(k, dim=1).indices  # (B, k)
        pred.scatter_(dim=1, index=topk_indices, value=True)

    ground_truth = targets.bool()  # (B, Vocab_size_target)

    tp = (pred & ground_truth).sum()
    fp = (pred & ~ground_truth).sum()
    fn = (~pred & ground_truth).sum()
    return tp, fp, fn


# -------------------- training & evaluation loops --------------------
def train_epoch(model_, loader, optimizer_, crit_):
    model_.train()
    tl = tp = fp = fn = 0.0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        optimizer_.zero_grad()
        logits = model_(src)
        loss = custom_loss(logits, tgt, crit_)
        loss.backward()
        optimizer_.step()

        # Determine the metrics. NB: logits.detatch() to prevent grad calcs
        btp, bfp, bfn = batch_set_metrics(logits.detach(), tgt, k=NUM_OUTPUTS)
        tl += loss.item()
        tp += btp
        fp += bfp
        fn += bfn

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return tl / (len(loader) or 1), prec.item(), rec.item()


@torch.no_grad()
def eval_epoch(model_, loader, crit_):
    model_.eval()
    tl = tp = fp = fn = 0.0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        logits = model_(src)
        tl += custom_loss(logits, tgt, crit_).item()

        # Determine the metrics. No need for logits.detatch()
        btp, bfp, bfn = batch_set_metrics(logits, tgt, k=NUM_OUTPUTS)
        tp += btp
        fp += bfp
        fn += bfn

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return tl / (len(loader) or 1), prec.item(), rec.item()


@torch.no_grad()
def infer(
    model_,
    sentence: str,
    source_vocab: dict,
    inv_target_vocab: dict,
    k: int = NUM_OUTPUTS,
):
    model_.eval()
    input_indexes = encode(tokenize(sentence)[:NUM_INPUTS], source_vocab)
    if len(input_indexes) < NUM_INPUTS:
        input_indexes += [source_vocab[PAD_TOKEN]] * (NUM_INPUTS - len(input_indexes))
    x = torch.tensor(input_indexes, device=DEVICE).unsqueeze(0)  # (1, num_inputs)
    scores = model_(x).sigmoid().squeeze(0)  # shape: (Vocab_size_target)

    # Ensure k is not larger than the number of classes (scores.size(0))
    k = min(k, scores.size(0))
    if k == 0:  # Handle case where k or vocab size is 0
        return ""

    topk_indices = scores.topk(k).indices  # shape: (k)
    codes_list = [inv_target_vocab[i.item()] for i in topk_indices]
    return " ".join(codes_list)


if __name__ == "__main__":
    start_time = time.time()
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} sentence pairs")
    avg_tgt_tokens = df["target"].apply(lambda x: len(tokenize(str(x)))).mean()
    print(f"Average number of tokens in tgt: {avg_tgt_tokens:.2f}")

    # 1. Split data
    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,  # For reproducibility
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # 2. Vocab: Build from TRAINING data ONLY
    src_vocab, inv_src_vocab = build_vocab(train_df["src"], 1)
    tgt_vocab, inv_tgt_vocab = build_target_vocab(train_df["target"], 1)
    print(
        f"Src vocab (from train): {len(src_vocab):,} | Tgt vocab (from train): {len(tgt_vocab):,}"
    )

    # 3. Dataset / DataLoader
    train_ds = TranslationDataset(
        df_=train_df, source_vocab=src_vocab, target_vocab=tgt_vocab
    )
    val_ds = TranslationDataset(
        df_=val_df, source_vocab=src_vocab, target_vocab=tgt_vocab
    )

    collate_fn = lambda batch: collate(batch=batch)
    train_dl = DataLoader(
        dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_dl = DataLoader(
        dataset=val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # 4. Model / Optim
    model = MLPModel(
        src_vocab,
        tgt_vocab,
        emb_dim=D_MODEL // 2,
        hidden=[D_MODEL, D_MODEL // 2],
        p_drop=DROPOUT,
        num_inputs=NUM_INPUTS,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # pos-weight vector for recall-tilted BCE
    pos_weight = torch.full((len(tgt_vocab),), BETA_FN, device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(DEVICE)

    # 5. Training loop
    train_losses = []
    val_losses = []
    train_prec_metrics = []
    train_rec_metrics = []
    val_prec_metrics = []
    val_rec_metrics = []
    for ep in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        tr_loss, tr_p, tr_r = train_epoch(
            model,
            train_dl,
            optimizer,
            crit,
        )
        vl_loss, vl_p, vl_r = eval_epoch(model, val_dl, crit)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_prec_metrics.append(tr_p)
        train_rec_metrics.append(tr_r)
        val_prec_metrics.append(vl_p)
        val_rec_metrics.append(vl_r)

        epoch_end_time = time.time()
        epoch_minutes, epoch_seconds = divmod(
            int(epoch_end_time - epoch_start_time), 60
        )
        print(
            f"Epoch {ep:02d}/{EPOCHS} │ "
            f"TRAIN loss={tr_loss:.4f} P={tr_p:.3f} R={tr_r:.3f} │ "
            f"VALID loss={vl_loss:.4f} P={vl_p:.3f} R={vl_r:.3f} │ "
            f"Time: {epoch_minutes}m {epoch_seconds}s"
        )

    # 6. Save and plot
    ts = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    os.makedirs("models", exist_ok=True)
    os.makedirs("logging", exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
        },
        f"models/model_{ts}.pth",
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
        },
        "models/model_latest.pth",
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Add a title to the figure
    script_name = os.path.basename(__file__)
    fig.suptitle(
        f"{script_name}\n{DATA_PATH}\nEpochs: {EPOCHS}",
        fontsize=16,
    )

    epochs = range(1, EPOCHS + 1)
    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_prec_metrics, label="Train Precision")
    ax2.plot(epochs, val_prec_metrics, label="Validation Precision")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Precision")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)

    ax3.plot(epochs, train_rec_metrics, label="Train Recall")
    ax3.plot(epochs, val_rec_metrics, label="Validation Recall")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Recall")
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"logging/loss_{ts}.png")
    plt.savefig("logging/loss_latest.png")
    plt.close(fig)

    total_seconds = int(time.time() - start_time)
    minutes, seconds = divmod(total_seconds, 60)
    print(f"\nTotal runtime: {minutes}m {seconds}s")

    # Load dataset and sample random examples for testing
    print("\n--- Demo Inference ---")
    print("Testing on random samples from dataset (TOKEN-LEVEL RECALL):")
    print("=" * 60)

    # TESTING RANDOM samples from the dataset
    NUM_SAMPLES = 100
    test_samples = df.sample(n=NUM_SAMPLES, random_state=42).reset_index(drop=True)

    total_recall_sum = 0.0
    high_recall_count = 0  # Count samples with >=80% recall
    medium_recall_count = 0  # Count samples with 50-79% recall

    for i, row in test_samples.iterrows():
        src_text = row["src"]
        expected_text = row["target"]
        predicted_text = infer(model, src_text, src_vocab, inv_tgt_vocab, k=NUM_OUTPUTS)

        expected_tokens = set(expected_text.split())
        predicted_tokens = set(predicted_text.split()) if predicted_text else set()

        # Calculate token-level recall (same as training)
        true_positives = len(expected_tokens.intersection(predicted_tokens))
        total_expected = len(expected_tokens)
        token_recall = true_positives / total_expected if total_expected > 0 else 0.0

        total_recall_sum += token_recall

        # Status and counting based on recall level
        if token_recall >= 0.8:
            status = "✅"  # High recall (>=80%)
            high_recall_count += 1
        elif token_recall >= 0.5:
            status = "🟡"  # Medium recall (50-79%)
            medium_recall_count += 1
        else:
            status = "❌"  # Low recall (<50%)

        missing_tokens = expected_tokens - predicted_tokens

        print(f"\n[{i+1:2d}] INPUT: {src_text}")
        print(f"     EXPECTED: {expected_text}")
        print(f"     PREDICTED: {predicted_text} {status}")
        print(
            f"     TOKEN RECALL: {true_positives}/{total_expected} = {token_recall:.1%}"
        )

        if missing_tokens:
            print(f"     MISSING: {' '.join(sorted(missing_tokens))}")
        if len(predicted_tokens) > len(expected_tokens):
            extra_tokens = predicted_tokens - expected_tokens
            print(f"     EXTRA: {' '.join(sorted(extra_tokens))}")

    # Summary using token-level recall
    avg_token_recall = total_recall_sum / NUM_SAMPLES
    low_recall_count = NUM_SAMPLES - high_recall_count - medium_recall_count

    print(f"\n{'='*60}")
    print(f"TOKEN-LEVEL RECALL RESULTS:")
    print(f"Average Recall: {avg_token_recall:.1%} (should match train/val metrics)")
    print(f"✅ High recall (≥80%): {high_recall_count}")
    print(f"🟡 Medium recall (50-79%): {medium_recall_count}")
    print(f"❌ Low recall (<50%): {low_recall_count}")
