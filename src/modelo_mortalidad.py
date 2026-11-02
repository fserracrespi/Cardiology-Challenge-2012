import math
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


BERT_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Texto que se le pasa a ClinicalBERT por cada variable (las siglas solas dan embeddings pobres)
VAR_DESCRIPTIONS = {
    "ALP": "alkaline phosphatase",
    "ALT": "alanine transaminase",
    "AST": "aspartate transaminase",
    "Albumin": "serum albumin",
    "BUN": "blood urea nitrogen",
    "Bilirubin": "serum bilirubin",
    "Cholesterol": "serum cholesterol",
    "Creatinine": "serum creatinine",
    "DiasABP": "invasive diastolic arterial blood pressure",
    "FiO2": "fraction of inspired oxygen",
    "GCS": "glasgow coma score",
    "Glucose": "serum glucose",
    "HCO3": "serum bicarbonate",
    "HCT": "hematocrit",
    "HR": "heart rate",
    "K": "serum potassium",
    "Lactate": "serum lactate",
    "MAP": "invasive mean arterial blood pressure",
    "MechVent": "mechanical ventilation",
    "Mg": "serum magnesium",
    "NIDiasABP": "non-invasive diastolic arterial blood pressure",
    "NIMAP": "non-invasive mean arterial blood pressure",
    "NISysABP": "non-invasive systolic arterial blood pressure",
    "Na": "serum sodium",
    "PaCO2": "partial pressure of arterial carbon dioxide",
    "PaO2": "partial pressure of arterial oxygen",
    "Platelets": "platelet count",
    "RespRate": "respiratory rate",
    "SaO2": "arterial oxygen saturation",
    "SysABP": "invasive systolic arterial blood pressure",
    "Temp": "body temperature",
    "TroponinI": "troponin I",
    "TroponinT": "troponin T",
    "Urine": "urine output",
    "WBC": "white blood cell count",
    "pH": "arterial blood pH",
}


class ClinicalBERTVarEmbedding(nn.Module):
    """Embedding de la variable v a partir de Bio_ClinicalBERT, comprimido a d_model.

    ClinicalBERT codifica una sola vez el nombre de cada variable (768 dim, mean pooling)
    y se guarda como tabla congelada. La compresión 768 -> d_model sí se entrena."""

    def __init__(self, var_names, d_model=64, model_name=BERT_NAME):
        super().__init__()
        texts = [VAR_DESCRIPTIONS.get(name, name) for name in var_names]
        bert_table = self._encode(texts, model_name)                # (n_vars, 768)
        self.register_buffer("bert_table", bert_table)

        bert_dim = bert_table.size(1)
        self.norm = nn.LayerNorm(bert_dim)
        self.compress = nn.Linear(bert_dim, d_model)                # compresión 768 -> 64

    @staticmethod
    @torch.no_grad()
    def _encode(texts, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert = AutoModel.from_pretrained(model_name)
        bert.eval()
        enc = tokenizer(texts, padding=True, return_tensors="pt")
        out = bert(**enc).last_hidden_state                         # (n_vars, T, 768)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return (out * mask).sum(dim=1) / mask.sum(dim=1)            # mean pooling -> (n_vars, 768)

    def forward(self, v):                     # v: (B, N) long
        table = self.norm(self.bert_table)
        table = self.compress(table)          # (n_vars, d)
        return table[v]                       # (B, N, d)


class CVE(nn.Module):
    """Continuous Value Embedding: escalar -> vector de d_model.
    Linear(1, h) -> tanh -> Linear(h, d_model)."""

    def __init__(self, d_model, hidden=None):
        super().__init__()
        hidden = hidden or int(math.sqrt(d_model))  # como en STraTS
        self.linear1 = nn.Linear(1, hidden)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):                     # x: (B, N)
        x = x.unsqueeze(-1)                   # (B, N, 1)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x                              # (B, N, d)


class ObservationEmbedding(nn.Module):
    """e_j = ClinicalBERT(v_j) comprimido + CVE(x̃_j) + CVE(t_j)."""

    def __init__(self, var_names, d_model=64, dropout=0.1):
        super().__init__()
        self.var_emb = ClinicalBERTVarEmbedding(var_names, d_model)
        self.value_cve = CVE(d_model)
        self.time_cve = CVE(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t, v, x):
        # t, x: (B, N) float · v: (B, N) long
        e = self.var_emb(v) + self.value_cve(x) + self.time_cve(t)
        return self.dropout(e)                # (B, N, d)


class StaticEncoder(nn.Module):
    """MLP sobre los rasgos estáticos -> token e_0."""

    def __init__(self, n_static=8, d_model=64, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(n_static, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, s):                     # s: (B, n_static)
        s = self.linear1(s)
        s = self.relu(s)
        s = self.dropout(s)
        s = self.linear2(s)
        return s                              # (B, d)


class AttentionPooling(nn.Module):
    """z = Σ_j α_j h_j, ignorando el padding."""

    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(d_model, 1, bias=False)

    def forward(self, h, pad):                # h: (B, L, d) · pad: (B, L) bool, True = padding
        scores = self.linear1(h)
        scores = self.tanh(scores)
        scores = self.linear2(scores).squeeze(-1)                   # (B, L)
        scores = scores.masked_fill(pad, float("-inf"))
        alpha = torch.softmax(scores, dim=1)
        z = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return z, alpha                       # (B, d), (B, L)


class Classifier(nn.Module):
    """MLP sobre [z ; e_0] -> logit."""

    def __init__(self, d_model=64, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(2 * d_model, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, 1)

    def forward(self, z, e0):                 # z, e0: (B, d)
        x = torch.cat([z, e0], dim=-1)        # (B, 2d)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x.squeeze(-1)                  # (B,)


class MortalityTransformer(nn.Module):
    """Clasificador binario de mortalidad.
    Devuelve un logit; P(muerte) = sigmoid(logit). Entrenar con BCEWithLogitsLoss.

    var_names: lista de nombres de variables en el mismo orden que los índices v
               (la devuelve make_dataloaders)."""

    def __init__(self, var_names, n_static=8, d_model=64, n_heads=4,
                 n_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.obs_emb = ObservationEmbedding(var_names, d_model, dropout)
        self.static_enc = StaticEncoder(n_static, d_model, dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        self.pool = AttentionPooling(d_model)
        self.classifier = Classifier(d_model, dropout)

    def forward(self, t, v, x, static, pad):
        # t, x: (B, N) float · v: (B, N) long · static: (B, n_static) · pad: (B, N) bool, True = padding
        e_obs = self.obs_emb(t, v, x)                               # (B, N, d)
        e0 = self.static_enc(static)                                # (B, d)
        seq = torch.cat([e0.unsqueeze(1), e_obs], dim=1)            # (B, N+1, d)

        # El token estático nunca es padding: evita filas totalmente enmascaradas
        pad_full = torch.cat([pad.new_zeros(pad.size(0), 1), pad], dim=1)
        h = self.encoder(seq, src_key_padding_mask=pad_full)        # (B, N+1, d)

        z, alpha = self.pool(h, pad_full)                           # (B, d), (B, N+1)
        logit = self.classifier(z, e0)                              # (B,)
        return logit, alpha