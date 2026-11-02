import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Columnas del CSV physionet2012_transformed.csv
STATIC_COLS = ["Age", "Gender", "Height", "Weight",
               "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"]
CONT_STATIC_COLS = ["Age", "Height", "Weight"]   # se normalizan con media/std de train
LABEL_COL = "In-hospital_death"
T_MAX = 48.0                                      # horas; t se escala a [0, 1]


class PhysioNetDataset(Dataset):
    """Un elemento = un paciente: secuencia de observaciones (t, v, x̃) + estáticos + etiqueta."""

    def __init__(self, df, patients, var2idx, max_len=880):
        # df: filas de observaciones de un split · patients: tabla por RecordID con estáticos ya normalizados
        self.samples = []
        df = df.sort_values(["RecordID", "Time"], kind="stable")

        for rid, g in df.groupby("RecordID", sort=False):
            g = g.iloc[-max_len:]                 # si hay más de max_len, se quedan las más recientes

            t = torch.tensor(g["Time"].to_numpy() / T_MAX, dtype=torch.float32)
            v = torch.tensor(g["Variable"].map(var2idx).to_numpy(), dtype=torch.long)
            x = torch.tensor(g["Value"].to_numpy(), dtype=torch.float32)

            p = patients.loc[rid]
            static = torch.tensor(p[STATIC_COLS].to_numpy(dtype="float32"), dtype=torch.float32)
            y = torch.tensor(float(p[LABEL_COL]), dtype=torch.float32)

            self.samples.append((t, v, x, static, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Rellena hasta la secuencia más larga del batch y construye la máscara de padding."""
    B = len(batch)
    L = max(item[0].size(0) for item in batch)

    t = torch.zeros(B, L, dtype=torch.float32)
    v = torch.zeros(B, L, dtype=torch.long)
    x = torch.zeros(B, L, dtype=torch.float32)
    pad = torch.ones(B, L, dtype=torch.bool)      # True = padding

    for i, (ti, vi, xi, _, _) in enumerate(batch):
        n = ti.size(0)
        t[i, :n] = ti
        v[i, :n] = vi
        x[i, :n] = xi
        pad[i, :n] = False

    static = torch.stack([item[3] for item in batch])   # (B, n_static)
    y = torch.stack([item[4] for item in batch])        # (B,)
    return t, v, x, static, pad, y


def make_dataloaders(csv_path, batch_size=32, max_len=880, num_workers=0):
    """Devuelve ({'train','val','test'} -> DataLoader, var_names, n_static).
    var_names[i] es el nombre de la variable con índice v = i."""
    df = pd.read_csv(csv_path)

    # Vocabulario de variables (orden fijo)
    variables = sorted(df["Variable"].unique())
    var2idx = {name: i for i, name in enumerate(variables)}

    # Tabla de pacientes: estáticos, etiqueta y split
    patients = df.groupby("RecordID")[STATIC_COLS + [LABEL_COL, "Split"]].first()

    # Normalización de estáticos continuos con estadísticas de train (por paciente)
    train_p = patients[patients["Split"] == "train"]
    mean = train_p[CONT_STATIC_COLS].mean()
    std = train_p[CONT_STATIC_COLS].std()
    patients[CONT_STATIC_COLS] = (patients[CONT_STATIC_COLS] - mean) / std

    loaders = {}
    for split in ["train", "val", "test"]:
        ds = PhysioNetDataset(df[df["Split"] == split], patients, var2idx, max_len)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders, variables, len(STATIC_COLS)

