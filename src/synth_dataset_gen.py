"""Generate a synthetic seq2seq dataset with learnable patterns"""

import os
import time
import random

import numpy as np
import pandas as pd

# Configuration
INPUT_TOKEN_MIN = 1
INPUT_TOKEN_MAX = 20
MIN_OUTPUT_LEN = 1
MAX_OUTPUT_LEN = 9
NUM_SAMPLES_TO_GENERATE = 5000
DATASET_FOLDER = "Datasets/"


def pattern_fibonacci_like(a, b, length):
    """Generate Fibonacci-like sequence starting with a, b"""
    sequence = [a % 50, b % 50]  # Ensure initial values are bounded
    for _ in range(length - 2):
        next_val = (sequence[-1] + sequence[-2]) % 50  # Keep numbers smaller
        sequence.append(next_val)
    return sequence[:length]


def pattern_arithmetic_progression(a, b, length):
    """Generate arithmetic progression with difference (b-a), bounded to positive numbers"""
    diff = b - a
    sequence = []
    for i in range(length):
        val = a + i * diff
        # Keep numbers positive and reasonable (0-99)
        val = abs(val) % 100
        sequence.append(val)
    return sequence


def pattern_geometric_like(a, b, length):
    """Generate sequence where each term is previous term + (b-a)"""
    if a == 0:
        a = 1
    sequence = [a % 50]  # Ensure first element is bounded
    step = b - a if b != a else 1
    for i in range(1, length):
        next_val = sequence[-1] + step * (i % 3 + 1)  # Varying step
        sequence.append(abs(next_val) % 50)  # Keep numbers reasonable
    return sequence


def pattern_digit_expansion(a, b, length):
    """Generate sequence based on digits of a*b and operations, bounded"""
    product = a * b
    digits = [int(d) for d in str(product)]

    # If we need more digits, apply some operations but keep bounded
    while len(digits) < length:
        extensions = [(d + (a % 3)) % 10 for d in digits[-2:]]  # Keep single digits
        digits.extend(extensions)

    return digits[:length]


def pattern_modular_arithmetic(a, b, length):
    """Generate sequence using modular arithmetic patterns"""
    sequence = [a % 10, b % 10]
    modulus = (a + b) % 7 + 3  # Modulus between 3-9

    for i in range(2, length):
        next_val = (sequence[-1] * 2 + sequence[-2]) % modulus
        sequence.append(next_val)

    return sequence[:length]


def generate_sample():
    """Generate a single src->tgt sample with learnable pattern"""
    # Generate two input tokens
    a = np.random.randint(INPUT_TOKEN_MIN, INPUT_TOKEN_MAX + 1)
    b = np.random.randint(INPUT_TOKEN_MIN, INPUT_TOKEN_MAX + 1)

    # Choose random output length
    output_length = np.random.randint(MIN_OUTPUT_LEN, MAX_OUTPUT_LEN + 1)

    # Choose random pattern
    patterns = [
        pattern_fibonacci_like,
        pattern_arithmetic_progression,
        pattern_geometric_like,
        pattern_digit_expansion,
        pattern_modular_arithmetic,
    ]

    pattern_func = random.choice(patterns)
    target_sequence = pattern_func(a, b, output_length)

    # Shuffle the target sequence since we're learning sets (unordered)
    random.shuffle(target_sequence)

    # Format as space-separated strings
    src = f"{a} {b}"
    tgt = " ".join(map(str, target_sequence))

    return src, tgt


def generate_dataset():
    """Generate and save a synthetic seq2seq dataset"""
    data = []

    print(f"Generating {NUM_SAMPLES_TO_GENERATE:,} samples...")
    for i in range(NUM_SAMPLES_TO_GENERATE):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1:,} samples...")

        src, tgt = generate_sample()
        data.append({"src": src, "target": tgt})

    df = pd.DataFrame(data)

    # Create output directory
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    # Generate filename
    output_filename = os.path.join(DATASET_FOLDER, "dataset.csv")

    # Save dataset
    df.to_csv(output_filename, index=False)
    print(f"\nSaved {len(df)} samples to {output_filename}")

    # Show some examples
    print("\nSample data:")
    print(df.head(10).to_string(index=False))

    return df


def analyze_patterns():
    """Analyze the patterns in generated data"""
    samples = []
    for _ in range(20):
        src, tgt = generate_sample()
        samples.append((src, tgt))

    print("\nPattern Examples:")
    print("SRC -> TARGET")
    print("-" * 40)
    for src, tgt in samples:
        print(f"{src:8} -> {tgt}")


if __name__ == "__main__":
    print("Seq2Seq Synthetic Dataset Generator")
    print("=" * 50)

    # Show pattern examples first
    analyze_patterns()

    # Generate full dataset
    start_time = time.time()
    df = generate_dataset()
    elapsed = time.time() - start_time

    print(f"\nGeneration completed in {elapsed:.2f} seconds")
    print(f"Average time per sample: {elapsed/NUM_SAMPLES_TO_GENERATE*1000:.2f}ms")

    # Basic statistics
    df["src_len"] = df["src"].str.split().str.len()
    df["tgt_len"] = df["target"].str.split().str.len()

    print("\nDataset Statistics:")
    print(f"Total samples: {len(df):,}")
    print("Source length: always 2 tokens")
    print(f"Target length: {df['tgt_len'].min()}-{df['tgt_len'].max()} tokens")
    print(f"Average target length: {df['tgt_len'].mean():.1f}")
