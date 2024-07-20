import importlib
import random
import os

import mlflow
import numpy as np
import pandas as pd
from zenml.client import Client
import torch
import hydra
from skorch.regressor import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from src.model import load_features
from src.download_models import retrieve_model_with_alias

def evaluate(data_version, model_name, model_alias = "champion") -> None:
    X, y = load_features(name="features_target", version=data_version)
    X = X.values.astype(np.float32)
    y = y.values.astype(np.float32).reshape(-1, 1)
    model = retrieve_model_with_alias(model_name=model_name, model_alias=model_alias)
    model_uri = f"models:/{model_name}@{model_alias}"
    with mlflow.start_run() as run:
        predictions = model.predict(X)

        eval_data = pd.DataFrame(y.numpy(), columns=["target"])
        eval_data["predictions"] = predictions.detach().numpy()

        result = mlflow.evaluate(
            model_uri,
            data=eval_data,
            model_type="regressor",
            targets= "target",
            predictions="predictions",
            evaluators = ["default"]
        )


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    print(cfg)
    model_name = cfg.model_name
    model_alias = cfg.model_alias
    data_version = cfg.data_version

    evaluate(data_version, model_name, model_alias)

if __name__ == "__main__":
    main()