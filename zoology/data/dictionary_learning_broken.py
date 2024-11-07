import numpy as np
import torch
from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment

_kv_mappings = {}

class DictionaryLearningConfig(DataSegmentConfig):
    name: str = "dictionary_learning"
    num_examples: int = 1000
    input_seq_len: int = 16
    vocab_size: int = 100
    num_kv_pairs: int = 10

    def build(self, seed: int) -> DataSegment:
        return dictionary_learning(**self.model_dump(), seed=seed)

def get_or_create_mapping(vocab_size: int, num_kv_pairs: int, seed: int) -> dict:
    """Create or retrieve a consistent key-value mapping for a given seed."""
    key = (vocab_size, num_kv_pairs, seed)
    if key not in _kv_mappings:
        np.random.seed(seed)
        keys = np.arange(1, num_kv_pairs + 1)
        values = np.random.choice(np.arange(num_kv_pairs + 1, vocab_size), 
                                size=num_kv_pairs, replace=False)
        _kv_mappings[key] = dict(zip(keys, values))
    return _kv_mappings[key]

def print_kv_mapping(vocab_size: int, num_kv_pairs: int, seed: int):
    """Print the key-value mapping for a given configuration."""
    mapping = get_or_create_mapping(vocab_size, num_kv_pairs, seed)
    print(f"\nKV Mapping (vocab_size={vocab_size}, num_pairs={num_kv_pairs}, seed={seed}):")
    for k, v in sorted(mapping.items()):
        print(f"Key {k} -> Value {v}")
    return mapping

def dictionary_learning(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    num_kv_pairs: int,
    seed: int,
    verbose: bool = False,
    **kwargs
) -> DataSegment:
    """
    Generates synthetic data for a dictionary learning task where the model needs to
    memorize key-value pairs. The mapping from keys to values is consistent across
    the dataset, requiring the model to store this mapping in its parameters.

    Args:
        vocab_size (int): The size of the vocabulary.
        num_examples (int): The number of examples to generate.
        input_seq_len (int): The length of each input sequence.
        num_kv_pairs (int): The number of unique key-value pairs to memorize.
        seed (int): Random seed for reproducibility.

    Returns:
        DataSegment: Contains inputs and labels for training/testing.
    """
    mapping = get_or_create_mapping(vocab_size, num_kv_pairs, seed)
    if verbose:
        print(f"\nGenerating dataset with {num_examples} examples")
        print_kv_mapping(vocab_size, num_kv_pairs, seed)
    
    example_seed = hash((seed, num_examples)) % (2**32)
    np.random.seed(example_seed)

    keys = list(mapping.keys())
    inputs = []
    labels = []

    for _ in range(num_examples):
        input_seq = np.random.randint(1, vocab_size, size=input_seq_len)
        key = np.random.choice(keys)
        value = mapping[key]

        key_pos = np.random.randint(0, input_seq_len)
        input_seq[key_pos] = key

        label_seq = np.full_like(input_seq, -100)
        label_seq[key_pos] = value

        inputs.append(input_seq)
        labels.append(label_seq)

    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)

    return DataSegment(
        inputs,
        labels,
        slices={"input_seq_len": input_seq_len}
    )

def main():
    """Test that mappings are consistent across different calls."""
    config = DictionaryLearningConfig(
        num_examples=5,
        input_seq_len=16,
        vocab_size=100,
        num_kv_pairs=10
    )

    data1 = config.build(seed=42)
    data2 = config.build(seed=42)

    print("Dataset 1 mappings:")
    print_mappings(data1)
    print("\nDataset 2 mappings:")
    print_mappings(data2)

def print_mappings(data_segment):
    """Helper to print KV mappings found in a dataset."""
    mapping = {}
    inputs = data_segment.inputs.numpy()
    labels = data_segment.labels.numpy()
    
    for i in range(len(inputs)):
        key_pos = np.where(labels[i] != -100)[0]
        for pos in key_pos:
            key = inputs[i][pos]
            value = labels[i][pos]
            mapping[key] = value
    
    for k, v in sorted(mapping.items()):
        print(f"Key {k} -> Value {v}")

if __name__ == "__main__":
    # Test configuration
    VOCAB_SIZE = 100
    NUM_KV_PAIRS = 10
    SEED = 42

    print("Training Dataset Mapping:")
    train_mapping = print_kv_mapping(VOCAB_SIZE, NUM_KV_PAIRS, SEED)
    
    print("\nTest Dataset Mapping:")
    test_mapping = print_kv_mapping(VOCAB_SIZE, NUM_KV_PAIRS, SEED)
    
    assert train_mapping == test_mapping, "Train and test mappings differ!"
    print("\nVerified: Train and test mappings are identical")