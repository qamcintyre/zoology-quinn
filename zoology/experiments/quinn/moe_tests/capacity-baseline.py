from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig
from zoology.data.utils import DataSegment
from zoology.data.dictionary_learning import DictionaryLearningConfig
from pathlib import Path

train_config = DictionaryLearningConfig(
    vocab_size=100,
    input_seq_len=16,
    num_examples=10_000,
    num_kv_pairs=10
)
test_config = DictionaryLearningConfig(
    vocab_size=100,
    input_seq_len=16,
    num_examples=1_000,
    num_kv_pairs=10
)

# Training Configuration Setup
config = TrainConfig(
    data=DataConfig(
        cache_dir="/data/quinn/dictionary_learning",
        train_configs=[train_config],
        test_configs=[test_config],
        batch_size=64,
    ),
    model=ModelConfig(
        vocab_size=100,
        max_position_embeddings=16,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"dropout": 0.1, "num_heads": 2}
        )
    ),
)

configs = [config]
