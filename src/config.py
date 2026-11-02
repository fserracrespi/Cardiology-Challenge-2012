from pathlib import Path

BASE_CONFIG = {
    "csv_path": "data/processed/physionet2012_transformed.csv",
    "seed": 42,

    # datos
    "batch_size": 64,
    "max_len": 880,
    "num_workers": 0,

    # modelo
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "d_ff": 256,
    "dropout": 0.1,

    # entrenamiento
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 50,
    "patience": 10,
    "use_pos_weight": True,
}


CONFIGS = [
    {
        **BASE_CONFIG,
        "name": "d128_h4_l4",
        "checkpoint":  Path.cwd() / "models/checkpoints/d128_h4_l4.pt",
    },

    {
        **BASE_CONFIG,
        "name": "d256_h4_l4",
        "d_model": 256,
        "d_ff": 512,
        "checkpoint":  Path.cwd() / "models/checkpoints/d256_h4_l4.pt",
    },

    {
        **BASE_CONFIG,
        "name": "d256_h8_l4",
        "d_model": 256,
        "n_heads": 8,
        "d_ff": 512,
        "checkpoint": Path.cwd() /"models/checkpoints/d256_h8_l4.pt",
    },

    {
        **BASE_CONFIG,
        "name": "d128_h4_l6",
        "n_layers": 6,
        "checkpoint": Path.cwd() /"models/checkpoints/d128_h4_l6.pt",
    },
]