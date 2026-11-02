import mlflow
import mlflow.pytorch

from config import CONFIGS
from train import training_module


mlflow.set_experiment("MortalityTransformer")


for cfg in CONFIGS:

    with mlflow.start_run(run_name=cfg["name"]):

        print(f"\n{'='*50}")
        print(f"RUN: {cfg['name']}")
        print(f"{'='*50}")

        # Parámetros que quieres comparar en MLflow
        mlflow.log_params({
            "seed": cfg["seed"],
            "batch_size": cfg["batch_size"],
            "max_len": cfg["max_len"],
            "d_model": cfg["d_model"],
            "n_heads": cfg["n_heads"],
            "n_layers": cfg["n_layers"],
            "d_ff": cfg["d_ff"],
            "dropout": cfg["dropout"],
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "patience": cfg["patience"],
            "use_pos_weight": cfg["use_pos_weight"],
        })

        # Entrenamiento
        model, test_metrics, history = training_module(cfg)

        # Métricas por época
        for epoch in range(len(history["train_loss"])):
        
            mlflow.log_metric("train_loss", history["train_loss"][epoch], step=epoch + 1 )
            mlflow.log_metric("val_loss", history["val_loss"][epoch],step=epoch + 1)
            mlflow.log_metric("val_auroc", history["val_auroc"][epoch], step=epoch + 1)
            mlflow.log_metric("val_aucpr", history["val_aucpr"][epoch], step=epoch + 1)

        # Métricas de test
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", float(value))

        # Guardar modelo
        mlflow.pytorch.log_model(model, name="model")