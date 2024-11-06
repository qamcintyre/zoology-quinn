from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig
from zoology.data.associative_recall import MQARConfig

base_config = MQARConfig(vocab_size=256, input_seq_len=64, num_examples=10_000, num_kv_pairs=4)
test_config = MQARConfig(vocab_size=256, input_seq_len=64, num_examples=1_000, num_kv_pairs=4)

config = TrainConfig(
    data=DataConfig(
        cache_dir="/data/quinn/zoology",
        train_configs=[base_config],
        test_configs=[test_config],
        batch_size=256,
    ),
    model=ModelConfig(
        vocab_size=256,
        max_position_embeddings=64,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"dropout": 0.1, "num_heads": 1}
        )
    ),
)

configs = [config]