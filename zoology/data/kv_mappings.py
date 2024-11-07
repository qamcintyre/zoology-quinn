import numpy as np
import torch
from typing import Dict, Optional
from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment

class KeyValueMemorizationConfig(DataSegmentConfig):
    name: str = "key_value_memorization"
    num_examples: int = 1000
    num_kv_pairs: int = 50
    input_seq_len: int = 20
    test_split: float = 0.1

    def build(self, seed: int) -> DataSegment:
        np.random.seed(seed)
        # Generate keys and values
        num_kv_pairs = self.num_kv_pairs
        keys = np.arange(1, num_kv_pairs + 1)
        values = np.arange(num_kv_pairs + 1, num_kv_pairs * 2 + 1)
        kv_dict = dict(zip(keys, values))
        # Adjust vocab_size to include all keys and values plus the padding token (0)
        self.vocab_size = num_kv_pairs * 2 + 1  # +1 for padding token
        return key_value_memorization(self, kv_dict, seed=seed)

def key_value_memorization(
    config: KeyValueMemorizationConfig,
    kv_dict: Dict[int, int],
    seed: int,
    **kwargs
) -> DataSegment:
    np.random.seed(seed)

    vocab_size = config.vocab_size
    num_examples = config.num_examples
    input_seq_len = config.input_seq_len
    test_split = config.test_split

    keys = list(kv_dict.keys())
    values = list(kv_dict.values())
    vocab = keys + values

    inputs = []
    labels = []

    num_train_examples = int(num_examples * (1 - test_split))
    num_test_examples = num_examples - num_train_examples

    for _ in range(num_train_examples):
        sequence = []
        label_seq = []

        num_pairs_in_sequence = np.random.randint(1, len(keys) + 1)
        selected_keys = np.random.choice(keys, size=num_pairs_in_sequence, replace=False)

        for key in selected_keys:
            value = kv_dict[key]
            # Remove noise tokens or adjust as needed
            sequence.extend([key, value])

        sequence = sequence[:input_seq_len]

        label_seq = [-100] * len(sequence)
        for idx in range(1, len(sequence)):
            if sequence[idx - 1] in keys:
                key_prev = sequence[idx - 1]
                label_seq[idx] = kv_dict[key_prev]

        pad_length = input_seq_len - len(sequence)
        if pad_length > 0:
            sequence.extend([0] * pad_length)
            label_seq.extend([-100] * pad_length)

        input_tensor = torch.tensor(sequence, dtype=torch.int64)
        label_tensor = torch.tensor(label_seq, dtype=torch.int64)

        inputs.append(input_tensor)
        labels.append(label_tensor)

    for _ in range(num_test_examples):
        key = np.random.choice(keys)
        value = kv_dict[key]

        sequence = [key] + [0] * (input_seq_len - 1)
        label_seq = [-100] * input_seq_len
        label_seq[1] = value

        input_tensor = torch.tensor(sequence, dtype=torch.int64)
        label_tensor = torch.tensor(label_seq, dtype=torch.int64)

        inputs.append(input_tensor)
        labels.append(label_tensor)

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)

    return DataSegment(
        inputs,
        labels,
        slices={"num_kv_pairs": len(keys), "input_seq_len": input_seq_len}
    )
