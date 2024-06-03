import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.circuits import ParityConfig


sweep_id = uuid.uuid4().hex[:6]

SEQLEN=64
sweep_name = f"v2-mix-parity-L{SEQLEN}" + sweep_id
VOCAB_SIZE = 3

# 1. First we are going to create the data configuration

train_configs = [
    ParityConfig(vocab_size=VOCAB_SIZE, input_seq_len=4, num_examples=100_000),
    ParityConfig(vocab_size=VOCAB_SIZE, input_seq_len=8, num_examples=20_000),
    ParityConfig(vocab_size=VOCAB_SIZE, input_seq_len=16, num_examples=20_000),
    ParityConfig(vocab_size=VOCAB_SIZE, input_seq_len=32, num_examples=20_000),
    ParityConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=20_000),
    ]
test_configs = [
    ParityConfig(vocab_size=VOCAB_SIZE, input_seq_len=4, num_examples=1_000),
    ParityConfig(vocab_size=VOCAB_SIZE, input_seq_len=8, num_examples=1_000),
    ParityConfig(vocab_size=VOCAB_SIZE, input_seq_len=16, num_examples=1_000),
    ParityConfig(vocab_size=VOCAB_SIZE, input_seq_len=32, num_examples=1_000),
    ParityConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000),
    ]

input_seq_len=max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 256
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    # can pass a tuple if you want a different batch size for train and test
    batch_size=(batch_size, batch_size / 8),
    cache_dir="/home/quinn/quinn_data/synthetics",
)

# 2. Next, we are going to collect all the different model configs we want to sweep
models = []
model_factory_kwargs = {
    # "state_mixer": dict(name="torch.nn.Identity", kwargs={}), 
    "state_mixer": dict(name="zoology.mixers.mlp.GLU", kwargs={"hidden_mult": 4}),
    "vocab_size": VOCAB_SIZE,
}

# define this conv outside of if/else block because it is used in multiple models
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
    }
)

# Recurrent attention
for d_model in [
#    8, 
    16, 
    32,
]:
    for num_heads in [
        2, 
        4,
        8,
        16,
    ]:
        for num_layers in [
            # 1,
            2,
            # 3,
            # 4,
#            5,
        ]:
            for num_reps in [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10
            ]:
                for rep_dim in [
                    'q',
                    'k'
                ]:
                    attention_mixer = dict(
                        name="zoology.mixers.recurrent_attn_explore.fixed_recurrent.MHR",
                        kwargs={
                            "dropout": 0.1,
                            "num_heads": num_heads,
                            "repetitions": num_reps,
                            "rep_dim": 'k',
                        },
                    )
                    mixer = ModuleConfig(
                        name="zoology.mixers.hybrid.Hybrid",
                        kwargs={"configs": [conv_mixer, attention_mixer]}
                    )
                    model = ModelConfig(
                        block_type = "TransformerBlock",
                        d_model=d_model,
                        n_layers=num_layers,
                        sequence_mixer=mixer,
                        max_position_embeddings=0,
                        name=f"repeat-attention-dim-{d_model}-heads-{num_heads}-layers-{num_layers}-reps-{num_reps}-rep_dim-{rep_dim}",
                        **model_factory_kwargs
                    )
                    models.append(model)

# attention
for d_model in [
#    8, 
    16, 
    32
    ]:
    for num_heads in [
        2, 
        4,
        8,
        ]:
        for num_layers in [
            # 1,
            2,
            # 3,
            # 4,
            # 5
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
                n_layers=num_layers,
                sequence_mixer=mixer,
                max_position_embeddings=0,
                name=f"attention-dim-{d_model}-heads-{num_heads}-layers-{num_layers}",
                **model_factory_kwargs
            )
            models.append(model)

# 3. Finally we'll create a train config for each
configs = []
for model in models:
    for i, lr in enumerate(np.logspace(-3.5, -2, 4)):
        run_id = f"{model.name}-lr{lr:.1e}"
        config = TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            logger=LoggerConfig(
                project_name="CurriculumParity2LayerRecurrentTransition",
                entity="hazy-research"
            ),
            slice_keys=['input_seq_len'],
            sweep_id=sweep_name,
            run_id=run_id,
            predictions_path=f"/home/quinn/quinn_data/synthetics/predictions/{run_id}",
            collect_predictions=True,
        )
        configs.append(config)