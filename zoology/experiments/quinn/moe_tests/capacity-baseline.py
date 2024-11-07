from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig
from zoology.data.kv_mappings import KeyValueMemorizationConfig

import numpy as np

seed = 42
np.random.seed(seed)

vocab_size = 4096
num_kv_pairs = 2000
input_seq_len = 64

keys = np.random.choice(range(1, vocab_size // 2), size=num_kv_pairs, replace=False)
values = np.random.choice(range(vocab_size // 2, vocab_size), size=num_kv_pairs, replace=False)
shared_kv_dict = dict(zip(keys, values))

base_config = KeyValueMemorizationConfig(
    vocab_size=vocab_size,
    input_seq_len=input_seq_len,
    num_examples=10_000,
    num_kv_pairs=num_kv_pairs,
    test_split=0.0,
    kv_dict=shared_kv_dict
)

test_config = KeyValueMemorizationConfig(
    vocab_size=vocab_size,
    input_seq_len=input_seq_len,
    num_examples=1_000,
    num_kv_pairs=num_kv_pairs,
    test_split=1.0,
    kv_dict=shared_kv_dict
)

config = TrainConfig(
    data=DataConfig(
        cache_dir="/data/quinn/zoology",
        train_configs=[base_config],
        test_configs=[test_config],
        batch_size=256,
    ),
    model=ModelConfig(
        vocab_size=vocab_size,
        max_position_embeddings=input_seq_len,
        d_model=12,
        n_layers=1,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"dropout": 0.1, "num_heads": 1}
        ),
        mlp=ModuleConfig(
            name="zoology.models.mlp.FF",
            kwargs={"hidden_size": 32, "dropout": 0.1}
        ),
    ),
)

configs = [config]
