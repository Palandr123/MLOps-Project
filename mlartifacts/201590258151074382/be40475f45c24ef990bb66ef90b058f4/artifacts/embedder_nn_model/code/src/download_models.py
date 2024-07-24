from pathlib import Path

import mlflow
import hydra


def retrieve_model_with_alias(model_name, model_alias = "champion") -> mlflow.pyfunc.PyFuncModel:

    best_model:mlflow.pyfunc.PyFuncModel = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{model_alias}")

    # best_model
    return best_model


@hydra.main(config_path="../configs", config_name="download_models", version_base=None)
def main(cfg=None):
    client = mlflow.MlflowClient()

    for alias in cfg.aliases:
        # Get the model version by alias
        model_version = client.get_model_version_by_alias(name=cfg.model_name, alias=alias)
        
        # Get the model URI
        model_uri = f"models:/{cfg.model_name}/{model_version.version}"
        
        # Download the model
        model_dst_path = Path("models") / f"{cfg.model_name}_{alias}"
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=model_dst_path)


if __name__ == "__main__":
    main()
