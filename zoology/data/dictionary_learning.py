import numpy as np
import torch
from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment

class DictionaryLearningConfig(DataSegmentConfig):
    name: str = "dictionary_learning"
    num_examples: int = 1000
    input_seq_len: int = 16
    vocab_size: int = 100
    num_kv_pairs: int = 10

    def build(self, seed: int) -> DataSegment:
        return dictionary_learning(**self.model_dump(), seed=seed)

def dictionary_learning(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    num_kv_pairs: int,
    seed: int,
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
    np.random.seed(seed)

    keys = np.arange(1, num_kv_pairs + 1)
    values = np.random.choice(np.arange(num_kv_pairs + 1, vocab_size), size=num_kv_pairs, replace=False)
    mapping = dict(zip(keys, values))

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
    config = DictionaryLearningConfig(
        num_examples=1000,
        input_seq_len=16,
        vocab_size=100,
        num_kv_pairs=10
    )

    data_segment = config.build(seed=42)

    inputs = data_segment.inputs
    labels = data_segment.labels

    print("Sample Inputs and Labels:")
    for i in range(5):
        input_seq = inputs[i].tolist()
        label_seq = labels[i].tolist()
        print(f"Input {i}: {input_seq}")
        print(f"Label {i}: {label_seq}")

        key_positions = [idx for idx, val in enumerate(label_seq) if val != -100]
        for pos in key_positions:
            key = input_seq[pos]
            value = label_seq[pos]
            print(f"At position {pos}, key {key} maps to value {value}")
        print()

if __name__ == "__main__":
    main()