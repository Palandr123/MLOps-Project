import importlib
import random
import os

import mlflow
import numpy as np
import pandas as pd
from zenml.client import Client
import torch
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from nn import NNWrapper


mlflow.set_tracking_uri(uri="http://localhost:5000")


def load_features(name, version, size=1):
    client = Client()

    # Fetch all artifacts with the specified name and version
    artifacts = client.list_artifacts(name=name, version=version)

    # Ensure consistent sorting of artifacts
    artifacts = sorted(artifacts, key=lambda x: x.version, reverse=True)

    df = artifacts[0].load()
    for column in df.columns:
        if column.startswith("label__"):
            df = df[df[column] != -1]
    df = df.sample(
        frac=size, random_state=88
    )  # Ensure reproducibility with a fixed random state

    print("size of df is ", df.shape)
    print("df columns: ", df.columns)

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    print("shapes of X,y = ", X.shape, y.shape)

    return X, y


def train(X_train, y_train, cfg):
    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)

    # Define the model hyperparameters
    params = cfg.model.params

    # Train the model
    module_name = cfg.model.module_name
    class_name = cfg.model.class_name

    # Load "module.submodule.MyClass"
    class_instance = getattr(importlib.import_module(module_name), class_name)
    optimizer = getattr(
        importlib.import_module(cfg.model.optimizer.module_name),
        cfg.model.optimizer.class_name,
    )

    estimator = NNWrapper(module=class_instance, optimizer=optimizer)

    param_grid = dict(params)

    scoring = list(cfg.model.metrics.values())
    evaluation_metric = cfg.model.evaluation_metric

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=cfg.cv_n_jobs,
        refit=evaluation_metric,
        cv=cfg.model.folds,
        verbose=1,
        return_train_score=True,
    )

    gs.fit(X_train, y_train)

    return gs


def plot_and_log_metrics(estimator, plot_name, artifact_path):
    """Plot and log training and validation loss."""
    train_losses = estimator.history[:, "train_loss"]
    valid_losses = estimator.history[:, "valid_loss"]

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plot_path = f"{plot_name}.png"
    plt.savefig(plot_path)
    plt.close()

    mlflow.log_artifact(plot_path, artifact_path=artifact_path)
    os.remove(plot_path)

    return plot_path


def log_metadata(cfg, gs, X_train, y_train, X_test, y_test):

    cv_results = (
        pd.DataFrame(gs.cv_results_)
        .filter(regex=r"std_|mean_|param_")
        .sort_index(axis=1)
    )
    best_metrics_values = [
        result[1][gs.best_index_] for result in gs.cv_results_.items()
    ]
    best_metrics_keys = [metric for metric in gs.cv_results_]
    best_metrics_dict = {
        k: v
        for k, v in zip(best_metrics_keys, best_metrics_values)
        if "mean" in k or "std" in k
    }

    # print(cv_results, cv_results.columns)

    params = best_metrics_dict

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    experiment_name = cfg.model.model_name + "_" + cfg.experiment_name

    try:
        # Create a new MLflow Experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id  # type: ignore

    print("experiment-id : ", experiment_id)

    cv_evaluation_metric = cfg.model.cv_evaluation_metric
    run_name = "_".join([cfg.run_name, cfg.model.model_name, cfg.model.evaluation_metric, str(params[cv_evaluation_metric]).replace(".", "_")])  # type: ignore
    print("run name: ", run_name)

    if mlflow.active_run():
        mlflow.end_run()

    # Fake run
    with mlflow.start_run():
        pass

    # Parent run
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        plot_path = plot_and_log_metrics(gs.best_estimator_, f"loss_{cfg.model.model_name}_best", cfg.model.artifact_path)
        mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path=f"{cfg.model.artifact_path}/{plot_path}", dst_path="results")

        df_train_dataset = mlflow.data.pandas_dataset.from_pandas(df=df_train, targets=cfg.data.target_cols[0])  # type: ignore
        df_test_dataset = mlflow.data.pandas_dataset.from_pandas(df=df_test, targets=cfg.data.target_cols[0])  # type: ignore
        mlflow.log_input(df_train_dataset, "training")
        mlflow.log_input(df_test_dataset, "testing")

        # Log the hyperparameters
        mlflow.log_params(gs.best_params_)

        # Log the performance metrics
        mlflow.log_metrics(best_metrics_dict)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag(cfg.model.tag_key, cfg.model.tag_value)

        # Infer the model signature
        signature = mlflow.models.infer_signature(X_train, gs.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=gs.best_estimator_,
            artifact_path=cfg.model.artifact_path,
            signature=signature,
            input_example=X_train.iloc[0].to_numpy(),
            registered_model_name=cfg.model.model_name,
            pyfunc_predict_fn=cfg.model.pyfunc_predict_fn,
            code_paths=['src'],
        )

        client = mlflow.client.MlflowClient()
        client.set_model_version_tag(
            name=cfg.model.model_name,
            version=model_info.registered_model_version,
            key="source",
            value="best_Grid_search_model",
        )

        # Evaluate the best model
        predictions = gs.best_estimator_.predict(X_test)  # type: ignore
        eval_data = pd.DataFrame(y_test)
        eval_data.columns = ["label"]
        eval_data["predictions"] = predictions

        results = mlflow.evaluate(
            data=eval_data,
            model_type="regressor",  # Correct model type for regression
            targets="label",
            predictions="predictions",
            evaluators=["default"],
        )

        mlflow.log_metrics(results.metrics)

        print(f"Best model metrics:\n{results.metrics}")

        for index, result in cv_results.iterrows():

            child_run_name = "_".join(["child", run_name, str(index)])  # type: ignore
            with mlflow.start_run(
                run_name=child_run_name, experiment_id=experiment_id, nested=True
            ) as child_run:  # , tags=best_metrics_dict):
                ps = result.filter(regex="param_").to_dict()
                ms = result.filter(regex="mean_").to_dict()
                stds = result.filter(regex="std_").to_dict()

                # Remove param_ from the beginning of the keys
                ps = {k.replace("param_", ""): v for (k, v) in ps.items()}
                if 'max_epochs' in ps:
                    ps['max_epochs'] = int(ps['max_epochs'])

                mlflow.log_params(ps)
                mlflow.log_metrics(ms)
                mlflow.log_metrics(stds)

                # We will create the estimator at runtime
                module_name = cfg.model.module_name  # e.g. "sklearn.linear_model"
                class_name = cfg.model.class_name  # e.g. "LogisticRegression"

                # Load "module.submodule.MyClass"
                class_instance = getattr(
                    importlib.import_module(module_name), class_name
                )

                optimizer = getattr(
                    importlib.import_module(cfg.model.optimizer.module_name),
                    cfg.model.optimizer.class_name,
                )
                estimator = NNWrapper(
                    module=class_instance, optimizer=optimizer, **ps
                )

                estimator.fit(X_train, y_train)

                plot_path = plot_and_log_metrics(estimator, f"loss_{cfg.model.model_name}_{index}", cfg.model.artifact_path)
                mlflow.artifacts.download_artifacts(run_id=child_run.info.run_id, artifact_path=f"{cfg.model.artifact_path}/{plot_path}", dst_path="results")
                
                train_losses = estimator.history[:, "train_loss"]
                valid_losses = estimator.history[:, "valid_loss"]

                for idx, train_loss in enumerate(train_losses):
                    mlflow.log_metric("train_loss", train_loss, step=idx)
                for idx, valid_loss in enumerate(valid_losses):
                    mlflow.log_metric("valid_loss", valid_loss, step=idx)

                signature = mlflow.models.infer_signature(
                    X_train, estimator.predict(X_train)
                )

                model_info = mlflow.sklearn.log_model(
                    sk_model=estimator,
                    artifact_path=cfg.model.artifact_path,
                    signature=signature,
                    input_example=X_train.iloc[0].to_numpy(),
                    registered_model_name=cfg.model.model_name,
                    pyfunc_predict_fn=cfg.model.pyfunc_predict_fn,
                    code_paths=['src'],
                )

                model_uri = model_info.model_uri
                loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)

                predictions = loaded_model.predict(X_test)  # type: ignore

                eval_data = pd.DataFrame(y_test)
                eval_data.columns = ["label"]
                eval_data["predictions"] = predictions

                results = mlflow.evaluate(
                    data=eval_data,
                    model_type="regressor",
                    targets="label",
                    predictions="predictions",
                    evaluators=["default"],
                )

                print(f"metrics:\n{results.metrics}")
