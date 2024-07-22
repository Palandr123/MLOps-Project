import os
import shutil

import mlflow
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
import giskard

from src.model import load_features


def evaluate(data_version, model_name, model_alias = "champion") -> None:
    X, y = load_features(name="features_target", version=data_version)
    X = X.values.astype(np.float32)
    y = y.values.astype(np.float32).reshape(-1, 1)

    model_uri = f"models:/{model_name}@{model_alias}"
    loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)

    experiment_name = model_name + "_eval" 

    try:
        # Create a new MLflow Experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id  # type: ignore

    with mlflow.start_run(run_name=f"{model_name}_{model_alias}_{data_version}_eval", experiment_id=experiment_id) as run:
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("model_alias", model_alias)
        mlflow.set_tag("data_version", data_version)

        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_alias", model_alias)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("model_uri", model_uri)

        predictions = loaded_model.predict(X)

        eval_data = pd.DataFrame(y)
        eval_data.columns = ["label"]
        eval_data["predictions"] = predictions

        eval_data.to_csv("eval_data.csv", index=False)
        mlflow.log_artifact("eval_data.csv")
        os.remove("eval_data.csv")
        mlflow.sklearn.save_model(loaded_model, "model")
        mlflow.log_artifact("model")
        shutil.rmtree("model")

        results = mlflow.evaluate(
            data=eval_data,
            model_type="regressor",  # Correct model type for regression
            targets="label",
            predictions="predictions",
            evaluators=["default"],
        )
        mlflow.log_metrics(results.metrics)


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    print(cfg)
    model_name = cfg.model_name
    model_alias = cfg.model_alias
    data_version = cfg.data_version

    evaluate(data_version, model_name, model_alias)

if __name__ == "__main__":
    main()