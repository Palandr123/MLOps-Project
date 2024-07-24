import os
import shutil

import mlflow
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
from src.model import load_features
from src.data import read_datastore, preprocess_data
from collections.abc import Generator
import giskard
import giskard.testing
import yaml
import json


mlflow.set_tracking_uri(uri="http://localhost:5000")


def clear_tags_and_aliases(client: mlflow.client.MlflowClient, models: list[str]) -> None:
    for model in models:
        for mv in client.search_model_versions(f"name='{model}'"):
            model_info = dict(mv)
            for alias in model_info['aliases']:
                client.delete_registered_model_alias(model, alias)
            for tag in model_info['tags']:
                client.delete_registered_model_tag(model, tag)

def set_challengers(client: mlflow.client.MlflowClient, models: list[str], n_versions: int) -> None:
    challenger_num = 1

    for model in models:
        model_metadata = client.get_latest_versions(model, stages=["None"])
        latest_model_version = model_metadata[0].version
        for i in range(int(latest_model_version) - n_versions + 1, int(latest_model_version) + 1):
            client.set_registered_model_alias(model, f"challenger{challenger_num}", str(i))
            client.set_model_version_tag(model, str(i), key="challenger", value=True)
            challenger_num += 1

def get_challengers(client: mlflow.client.MlflowClient, models: list[str]) -> Generator[mlflow.pyfunc.PyFuncModel, None, None]:
    for model in models:
        for mv in client.search_model_versions(f"name='{model}'"):
            model_info = dict(mv)
            if len(model_info['aliases']) > 0 and "challenger" in model_info["tags"] and model_info["tags"]["challenger"]:
                model_uri = f"models:/{model}@{model_info['aliases'][0]}"
                loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)
                yield loaded_model, model_info["version"], model_info["name"]

def get_giskard_dataset(version: str) -> giskard.Dataset:
    data_cfg = hydra.compose("data")
    ohe_cfg = hydra.compose("ohe_out_names")
    label_cfg = hydra.compose("label_out_names")
    X, y = load_features("features_target", version)

    df = pd.merge(X, y, right_index=True, left_index=True)

    TARGET_COLUMN = data_cfg.data.target_cols[0]
    CATEGORICAL_COLUMNS = list(ohe_cfg.ohe_cols) + list(label_cfg.label_cols)
    dataset_name = data_cfg.data.dataset_name
    giskard_dataset = giskard.Dataset(
        df=df,  # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
        target=TARGET_COLUMN,  # Ground truth variable
        name=dataset_name, # Optional: Give a name to your dataset
        cat_columns=CATEGORICAL_COLUMNS  # List of categorical columns. Optional, but improves quality of results if available.
    )

    return giskard_dataset, dataset_name, version, df.columns

def predict(model: mlflow.pyfunc.PyFuncModel, df: pd.DataFrame) -> np.ndarray:
    X = df.values.astype(np.float32)
    return model.predict(X)[:, 0].astype(float)

def choose_champion(client: mlflow.client.MlflowClient, dataset_name: str, dataset_ver: str) -> None:
    validation_results = hydra.compose("validation_results")
    print(validation_results)
    succeded_models = validation_results.succeded_models
    best_model_info = None
    min_major_issues = 1e9
    for model_info in succeded_models:
        model_name, model_ver = model_info[list(model_info.keys())[0]]
        
        with open(f"reports/validation_results_{model_name}_v{model_ver}_{dataset_name}_{dataset_ver}.json") as f:
            model_report = json.load(f)
        
        n_major_issues = 0
        for detector_type in model_report:
            if "major" in model_report[detector_type]:
                n_major_issues += len(model_report[detector_type]["major"])
        if n_major_issues < min_major_issues:
            min_major_issues = n_major_issues
            best_model_info = (model_name, model_ver)
    
    if best_model_info is None:
        print("No models passed the success criteria")
    else:
        client.set_registered_model_alias(best_model_info[0], "champion", best_model_info[1])
        print(f"Marked {best_model_info[0]}_v{best_model_info[1]} as a champion")

@hydra.main(config_path="../configs", config_name="validate", version_base=None)
def main(cfg: DictConfig):
    client = mlflow.client.MlflowClient()
    
    # clear_tags_and_aliases(client, cfg.models)
    # set_challengers(client, cfg.models, cfg.n_versions)
    succeded_models = []
    failed_models = []
    for model, model_ver, model_name in get_challengers(client, cfg.models):
        custom_pred = lambda df: predict(model, df)
        giskard_dataset, dataset_name, dataset_ver, columns = get_giskard_dataset(cfg.data_version)

        giskard_model = giskard.Model(
            model=custom_pred,
            model_type = "regression",
            feature_names = columns,
            name=f"model_{model_name}_v{model_ver}_{dataset_name}_{dataset_ver}",
        )
        scan_results = giskard.scan(giskard_model, giskard_dataset)

        # Save the results in `html` file
        scan_results_path = f"reports/validation_results_{model_name}_v{model_ver}_{dataset_name}_{dataset_ver}.html"
        scan_results.to_html(scan_results_path)
        scan_results.to_json(f"reports/validation_results_{model_name}_v{model_ver}_{dataset_name}_{dataset_ver}.json")

        suite_name = f"test_suite_{model_name}_v{model_ver}_{dataset_name}_{dataset_ver}"
        test_suite = giskard.Suite(name = suite_name)
        test1 = giskard.testing.test_r2(model = giskard_model, 
                                dataset = giskard_dataset,
                                threshold=cfg.model.r2_threshold)
        test2 = giskard.testing.test_mae(model = giskard_model, 
                                dataset = giskard_dataset,
                                threshold=cfg.model.mae_threshold)
        test_suite.add_test(test1)
        test_suite.add_test(test2)

        test_results = test_suite.run()
        if (test_results.passed):
            succeded_models.append({f"{model_name}_model_ver": [model_name, model_ver]})
            print(f"Model {model_name}_v{model_ver}_{dataset_name}_{dataset_ver} passed model validation!")
        else:
            failed_models.append({f"{model_name}_model_ver": [model_name, model_ver]})
            print(f"Model {model_name}_v{model_ver}_{dataset_name}_{dataset_ver} has vulnerabilities!")

        with open('configs/validation_results.yaml', 'w') as outfile:
            yaml.dump({'succeded_models': succeded_models, 'failed_models': failed_models}, outfile)
        
    data_cfg = hydra.compose("data")
    choose_champion(client, data_cfg.data.dataset_name, cfg.data_version)


if __name__ == "__main__":
    main()