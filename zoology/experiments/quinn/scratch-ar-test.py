import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "hydra_attn" + sweep_id


VOCAB_SIZE = 8_192

train_configs = [    
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16),
    # MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=32),
    # MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=64),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=16),
    # MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=1_000, num_kv_pairs=32),
    # MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=1_000, num_kv_pairs=64),
    # MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=1_000, num_kv_pairs=128),
    # MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
]

input_seq_len=max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 256
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    # can pass a tuple if you want a different batch size for train and test
    batch_size=(batch_size, batch_size / 8),
    cache_dir="/home/quinn/quinn_data/synthetics"
)

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}), "vocab_size": VOCAB_SIZE,
}

conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
    }
)

# set up models

models = []

# scratch transformer
for d_model in [
    # 8,
    # 16,
    32,
    # 64,
    ]:
    for num_heads in [
        2,
        # 4,
    ]:
        for num_scratch in [
            1,
            # 2,
            # 4,
            ]:
            for scratch in [
                "add",
                # "replace",
                ]:
                attention_mixer = dict(
                    name="zoology.mixers.attention.MHA",
                    kwargs={
                        "dropout": 0.1,
                        "num_heads": num_heads
                    },
                )
                mixer = ModuleConfig(
                    name="zoology.mixers.hybrid.Hybrid",
                    kwargs={"configs": [conv_mixer, attention_mixer]}
                )
                model = ModelConfig(
                    block_type = "TransformerBlock",
                    d_model=d_model,
                    n_layers=2,
                    sequence_mixer=conv_mixer,
                    max_position_embeddings=0,
                    transformer="scratch",
                    scratch=scratch,
                    num_scratch=num_scratch,
                    name=f"scratch-dim-{d_model}-num-{num_scratch}-{scratch}",
                    **model_factory_kwargs
                )
                models.append(model)


# # attention
# for d_model in [
#     #16, 
#     32, 
#     64
#     ]:
#     attention_mixer = dict(
#         name="zoology.mixers.attention.MHA",
#         kwargs={
#             "dropout": 0.1,
#             "num_heads": 2
#         },
#     )
#     mixer = ModuleConfig(
#         name="zoology.mixers.hybrid.Hybrid",
#         kwargs={"configs": [conv_mixer, attention_mixer]}
#     )
#     model = ModelConfig(
#         block_type = "TransformerBlock",
#         d_model=d_model,
#         n_layers=2,
#         sequence_mixer=mixer,
#         max_position_embeddings=0,
#         name="attention",
#         **model_factory_kwargs
#     )
#     models.append(model)

# load training configs

configs = []
for model in models:
    for lr in np.logspace(-3, -1.5, 4):
        run_id = f"{model.name}-lr{lr:.1e}"
        config = TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            logger=LoggerConfig(
                project_name="scratch-ar-test",
                entity="negative-loss"
            ),
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
            predictions_path=f"/home/quinn/quinn_data/synthetics/predictions/{run_id}",
            collect_predictions=True,
        )
        configs.append(config)