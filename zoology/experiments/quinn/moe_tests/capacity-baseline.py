from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig
from zoology.data.kv_mappings import KeyValueMemorizationConfig

base_config = KeyValueMemorizationConfig(
    num_examples=10_000,
    num_kv_pairs=50,
    input_seq_len=64,
)
test_config = KeyValueMemorizationConfig(
    num_examples=1_000,
    num_kv_pairs=50,
    input_seq_len=64,
)

config = TrainConfig(
    data=DataConfig(
        cache_dir="/data/quinn/zoology",
        train_configs=[base_config],
        test_configs=[test_config],
        batch_size=256,
    ),
    model=ModelConfig(
        vocab_size=base_config.vocab_size,
        max_position_embeddings=base_config.input_seq_len,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"dropout": 0.1, "num_heads": 1}
        )
    ),
)

configs = [config]
