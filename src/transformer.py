# pylint: disable=C3001,R0914,R0913,R0917,C0115,C0413,C0116,C0301,C0103
"""
- Transformer-based model for multi-label prediction of ATC codes from ICD10 code pairs.
- Input: A pair of ICD10 codes.
- Output: A set of NUM_OUTPUTS ATC codes predicted autoregressively.
- The model uses a Transformer encoder-decoder architecture. The encoder processes the input ICD10 codes, and the decoder autoregressively generates ATC codes.
- Logits from each decoding step are max-pooled to produce final set logits, allowing the use of a multi-label BCE loss.
- Optimized to maximize recall through a weighted BCE loss, preferring false positives over false negatives.
"""
import os
import time
import datetime
import math
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATA_PATH = "Datasets/dataset.csv"

BATCH_SIZE = 64
EPOCHS = 10
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
    """
    Builds target vocab and inserts BOS / EOS so the decoder can start/stop.
    """
    frequency: dict[str, int] = {}
    for line in texts:
        for tok in tokenize(line):
            frequency[tok] = frequency.get(tok, 0) + 1

    vocab = {BOS_TOKEN: 0, EOS_TOKEN: 1}  # <-- CORRECTED
    for tok in sorted(frequency):
        if frequency[tok] >= min_freq:
            vocab.setdefault(tok, len(vocab))
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
        src       – LongTensor (2,)           two ICD-10 ids
        tgt_multi – FloatTensor (|V_atc|,)   multi-hot ATC vector
    """

    def __init__(self, df_: pd.DataFrame, source_vocab: dict, target_vocab: dict):
        self.X, self.Y = [], []
        v_size = len(target_vocab)
        for src_sentence, tgt_sentence in zip(df_["src"], df_["target"]):
            icd_ids = encode(tokenize(src_sentence)[:NUM_INPUTS], source_vocab)
            if len(icd_ids) < NUM_INPUTS:  # pad / drop rows with <2 codes
                icd_ids += [source_vocab[PAD_TOKEN]] * (NUM_INPUTS - len(icd_ids))
            atc_vec = torch.zeros(v_size, dtype=torch.float32)
            for tok in tokenize(tgt_sentence):
                if tok in target_vocab:
                    atc_vec[target_vocab[tok]] = 1.0
            self.X.append(torch.tensor(icd_ids, dtype=torch.long))
            self.Y.append(atc_vec)

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


class PositionalEncoding(nn.Module):
    """
    Standard sine/cosine positional encoding for Transformer models.
    Adds positional information to token embeddings for both encoder and decoder.
    Works with batch-first tensors of shape (B, S, D).
    """

    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, S, D)
        return x + self.pe[:, : x.size(1)]


class TransformerSetDecoder(nn.Module):
    """
    Transformer-based model for multi-label prediction of ATC codes from ICD10 code pairs.

    ---------------------------------------------------------------------------
    Model Purpose and Task Alignment
    ---------------------------------------------------------------------------
    This model is designed to predict a set of ATC codes (drug classes) given a pair of ICD10 codes (diagnoses).
    The dataset contains input pairs of ICD10 codes and a variable-length set of ATC codes (between 1 and 9) as targets.
    The primary goal is to maximize recall: it is preferable to predict extra (false positive) ATC codes
    rather than miss any relevant ones (false negatives).

    Why this model is well-suited for the task:

    1. **Set-based Output (Order-Invariant Prediction):**
        - The output is a set of ATC codes, not a sequence. The order in which ATC codes are predicted does not matter.
        - The model generates a fixed number (`num_outputs`) of predictions, but the final output is treated as a set.

    2. **Transformer Encoder-Decoder Architecture:**
        - The encoder processes the two ICD10 codes and summarizes them into a context representation using self-attention.
        - The decoder autoregressively generates `num_outputs` ATC code predictions, starting from a BOS token, using cross-attention to the encoded ICD10 context.

    3. **Max-Pooling Across Decoding Steps:**
        - At each decoding step, the model produces logits for all possible ATC codes.
        - Max-pooling across all decoding steps ensures that, for each ATC code, the maximum logit is used as the final score.
        - This removes any positional information, making the output order-invariant and set-based.

    4. **Multi-Label Loss and Recall Optimization:**
        - Uses a multi-label binary cross-entropy (BCE) loss, suitable for predicting multiple ATC codes per input.
        - The loss is weighted to penalize false negatives more than false positives, directly optimizing for high recall.

    5. **User Interface Alignment:**
        - Always predicts a fixed number of ATC codes (`num_outputs`), matching the UI requirement where users are presented with a set of suggestions.
        - No recalculation is needed based on user selection; the model's output is a single, high-recall set of ATC codes.

    ---------------------------------------------------------------------------
    Model Details
    ---------------------------------------------------------------------------
    - Encodes the two ICD10 codes using a Transformer encoder.
    - Autoregressively decodes `num_outputs` ATC tokens using a Transformer decoder.
    - The decoder starts with a BOS_TOKEN and greedily selects the next token.
    - Logits from each decoding step are aggregated using max pooling over the time dimension.
    - Optimized for recall by using a weighted BCE loss.

    Args:
        source_vocab (dict): Vocabulary mapping for ICD10 codes.
        target_vocab (dict): Vocabulary mapping for ATC codes.
        emb_dim (int): Embedding dimension for both source and target.
        nhead (int): Number of attention heads in the Transformer.
        num_layers (int): Number of layers in Transformer encoder/decoder.
        p_drop (float): Dropout probability.
        num_outputs (int): Number of ATC codes to predict (fixed set size).

    Returns:
        logits (Tensor): Tensor of shape (B, |V|), where each value is the
            max logit for that ATC code across all decoding steps, suitable
            for multi-label (set-based) prediction.
    """

    def __init__(
        self,
        source_vocab: dict,
        target_vocab: dict,
        emb_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        p_drop: float = 0.2,
        num_outputs: int = 9,
    ):
        super().__init__()
        self.src_embed = nn.Embedding(
            len(source_vocab), emb_dim, padding_idx=source_vocab[PAD_TOKEN]
        )
        self.tgt_embed = nn.Embedding(
            len(target_vocab), emb_dim, padding_idx=target_vocab[BOS_TOKEN]
        )
        self.pos_enc = PositionalEncoding(emb_dim)

        enc_layer = TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=p_drop, batch_first=True
        )
        self.encoder = TransformerEncoder(enc_layer, num_layers=num_layers)

        dec_layer = TransformerDecoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=p_drop, batch_first=True
        )
        self.decoder = TransformerDecoder(dec_layer, num_layers=num_layers)

        self.proj = nn.Linear(emb_dim, len(target_vocab))
        self.dropout = nn.Dropout(p_drop)
        self.num_outputs = num_outputs
        self.bos_idx = target_vocab[BOS_TOKEN]

    @staticmethod
    def _subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Causal mask for size sz (True = mask)."""
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        B = src.size(0)

        # Encode the two input tokens into contextualized representations
        src_emb = self.pos_enc(self.dropout(self.src_embed(src)))  # (B,2,E)
        memory = self.encoder(src_emb)  # (B,2,E) - contextualized input representations

        # Autoregressively decode num_outputs output tokens
        step_logits = []  # Will collect logits for each decoding step
        next_tok = torch.full(
            (B, 1), self.bos_idx, dtype=torch.long, device=src.device
        )  # (B,1) - start with BOS token

        for _ in range(self.num_outputs):
            # Embed current sequence of decoder tokens and add positional encoding
            tgt_emb = self.pos_enc(self.dropout(self.tgt_embed(next_tok)))  # (B,t,E)
            # Create causal mask to prevent attention to future tokens
            tgt_mask = self._subsequent_mask(
                tgt_emb.size(1), device=src.device
            )  # (t,t)
            # Decode using cross-attention to encoder memory and self-attention within decoder
            out = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
            )  # (B,t,E)
            this_step = out[:, -1, :]  # (B,E) - take the last token's representation
            logit = self.proj(this_step)  # (B,|V|) - logits for all output tokens
            step_logits.append(logit)
            # Greedy: feed back top prediction for next decoding step
            next_tok = torch.cat(
                [next_tok, logit.argmax(dim=-1, keepdim=True)], dim=1
            )  # append token for autoregressive decoding

        # stack & pool → (B, NUM_OUTPUTS, |V|) → (B, |V|)
        # Max-pool across decoding steps: for each output token, take the maximum logit
        # seen at any step. This removes output position information and enables
        # set-based (multi-label) prediction.
        logits = torch.stack(step_logits, dim=1).max(dim=1).values  # (B, |V|)
        return logits


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
    k: int,
):
    model_.eval()
    ids = encode(tokenize(sentence)[:NUM_INPUTS], source_vocab)
    if len(ids) < NUM_INPUTS:
        ids += [source_vocab[PAD_TOKEN]] * (NUM_INPUTS - len(ids))
    src = torch.tensor(ids, device=DEVICE).unsqueeze(0)  # (1,2)

    logits = model_(src)  # (1, |V|)
    scores = logits.sigmoid().squeeze(0)

    # grab top-k predictions
    actual_k = min(k, scores.size(0))
    if actual_k == 0:
        return ""
    topk_indices = scores.topk(actual_k).indices
    return " ".join(inv_target_vocab[i.item()] for i in topk_indices)


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
    model = TransformerSetDecoder(
        src_vocab,
        tgt_vocab,
        emb_dim=D_MODEL // 4,
        nhead=8,
        num_layers=1,
        p_drop=DROPOUT,
        num_outputs=NUM_OUTPUTS,
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

    # 6. Save
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

    # runtime
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
