import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.fuzzy_recall import FuzzyInContextRecallConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "hydra_attn" + sweep_id


VOCAB_SIZE = 8_192


train_configs = [
    FuzzyInContextRecallConfig(vocab_size=VOCAB_SIZE, seq_len=64, k_motif_size=3, v_motif_size=3, multi_query=True),
    #FuzzyInContextRecallConfig(vocab_size=VOCAB_SIZE, seq_len=128, k_motif_size=3, v_motif_size=3, multi_query=True),
    #FuzzyInContextRecallConfig(vocab_size=VOCAB_SIZE, seq_len=256, k_motif_size=4, v_motif_size=4, multi_query=True),
]

test_configs = [
    FuzzyInContextRecallConfig(vocab_size=VOCAB_SIZE, seq_len=64, k_motif_size=3, v_motif_size=3, multi_query=True),
    #FuzzyInContextRecallConfig(vocab_size=VOCAB_SIZE, seq_len=128, k_motif_size=4, v_motif_size=4, multi_query=True),
    #FuzzyInContextRecallConfig(vocab_size=VOCAB_SIZE, seq_len=256, k_motif_size=4, v_motif_size=4, multi_query=True),
]

input_seq_len=max([c.seq_len for c in train_configs + test_configs])
batch_size = 256
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    # can pass a tuple if you want a different batch size for train and test
    batch_size=(batch_size, batch_size / 8),
    cache_dir="/home/quinn/quinn_data/synthetics/"
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

# attention
for d_model in [8, 16, 32]:
    for num_heads in [2, 4]:
        attention_mixer = dict(
            name="zoology.mixers.attention.MHA",
            kwargs={
                "dropout": 0.0,
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
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name=f"attention-dim-{d_model}-heads-{num_heads}",
            **model_factory_kwargs
        )
        models.append(model)

# hydra attn
for d_model in [8, 16, 32]:
    for num_heads in [2, 4]:
        for combiner in [
            "scalar", 
            "dual", 
            "content_aware", 
            "softmax_content"
            ]:
            attention_mixer = dict(
                name="zoology.mixers.hydra.grouped_hydra_attn.MHH",
                kwargs={
                    "num_heads": num_heads,
                    "head_groups": 1, 
                    "combiner": combiner,  
                    "sum_dim": "q",  
                    "scalar_act": "identity",  
                    "bias": False,  
                    "dropout": 0,  
                }
            )
            mixer = ModuleConfig(
                name="zoology.mixers.hybrid.Hybrid",
                kwargs={"configs": [conv_mixer, attention_mixer]}
            )
            model=ModelConfig(
                block_type = "TransformerBlock",
                d_model=d_model,
                n_layers=2,
                sequence_mixer=mixer,
                max_position_embeddings=0,
                name=f"hydra_attention-{combiner}-dim-{d_model}-heads-{num_heads}",
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
            max_epochs=16,
            logger=LoggerConfig(
                project_name="HydraFuzzyRecall",
                entity="hazy-research"
            ),
            slice_keys=['input_seq_len'],
            sweep_id=sweep_name,
            run_id=run_id,
            predictions_path=f"/home/quinn/quinn_data/synthetics/predictions/{run_id}",
            collect_predictions=True,
        )
        configs.append(config)