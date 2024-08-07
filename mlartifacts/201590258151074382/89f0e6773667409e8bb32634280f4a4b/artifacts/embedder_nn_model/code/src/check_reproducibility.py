import random
from pathlib import Path
import importlib

import numpy as np
import torch
import zenml
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from omegaconf import open_dict

from src.model import load_features, train
from src.main import get_num_unique

def evaluate_model(gs, X_test, y_test, metrics_eval):
    predictions = gs.best_estimator_.predict(X_test)
    metrics = {}
    for metric_name, metric in metrics_eval.items():
        class_instance = getattr(
            importlib.import_module(metric.module_name), metric.class_name
        )
        metrics[metric_name] = class_instance(y_test, predictions)
    
    return metrics

@hydra.main(config_path="../configs", config_name="check_reproducibility")
def main(cfg: DictConfig):
    results_dir = Path.cwd() / Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    performance_metrics = {}
    metrics_eval = {}
    for seed in cfg.seeds:
        # Set the random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path="../configs"):
            main_config = compose(config_name="main", overrides=[f"random_state={seed}"])

        with hydra.initialize(version_base=None, config_path="../configs/model"):
            model_config = hydra.compose(config_name=f"{cfg.model_config}", overrides=[f"params.module__seed=[{seed}]"])
        
        with open_dict(main_config.model):
            main_config.model.merge_with(model_config)

        # Load training and testing data
        X_train, y_train = load_features(name="features_target", version=main_config.train_data_version)
        X_test, y_test = load_features(name="features_target", version=main_config.test_data_version)

        # Overwrite args.model.params.input_units with the number of features
        main_config.model.params.module__input_size = [X_train.shape[1]]
        if main_config.model.has_embeds:
            for i, col in enumerate(list(X_train.columns)):
                if col == main_config.model.region_column:
                    main_config.model.params.module__region_idx = [i]
                elif col == main_config.model.wmi_column:
                    main_config.model.params.module__wmi_idx = [i]
                elif col == main_config.model.vds_column:
                    main_config.model.params.module__vds_idx = [i]
                elif col == main_config.model.model_column:
                    main_config.model.params.module__model_idx = [i]
            client = zenml.client.Client()
            artifacts = client.list_artifacts(name="cat_transform")
            cat_transformer = artifacts[-1].load()
        
            main_config.model.params.module__num_regions = [get_num_unique(cat_transformer, "region")]
            main_config.model.params.module__num_wmis = [get_num_unique(cat_transformer, "WMI")]
            main_config.model.params.module__num_vds = [get_num_unique(cat_transformer, "VDS")]
            main_config.model.params.module__num_models = [get_num_unique(cat_transformer, "model")]

        # Train the model
        gs = train(X_train, y_train, cfg=main_config)

        # Get the best evaluation metric
        for metric_name, metric in main_config.model.metrics.items():
            best_score = gs.cv_results_['mean_test_' + metric][gs.best_index_]
            if metric_name in performance_metrics:
                performance_metrics[metric_name].append(best_score)
            else:
                performance_metrics[metric_name] = [best_score]
        results_eval = evaluate_model(gs, X_test, y_test, cfg.metrics_eval)
        for metric_name, value in results_eval.items():
            if metric_name not in metrics_eval:
                metrics_eval[metric_name] = [value]
            else:
                metrics_eval[metric_name].append(value)

    # Save the results to a text file
    results_file = results_dir / f"reproducibility_results_{cfg.model_config}.txt"
    with open(str(results_file), "w") as f:
        f.write(f"Performance metrics over {len(cfg.seeds)} experiments with different seeds:\n")
        for metric_name, metric in main_config.model.metrics.items():
            # Calculate average and variance of the performance metrics
            avg_performance = np.mean(performance_metrics[metric_name])
            var_performance = np.var(performance_metrics[metric_name])
            f.write(f"Results for {metric_name}\n")
            f.write("\n".join([f"Seed {seed}: {score}" for seed, score in zip(cfg.seeds, performance_metrics[metric_name])]))
            f.write(f"\nAverage performance: {avg_performance}\n")
            f.write(f"Variance in performance: {var_performance}\n\n")

        f.write(f"\nMetrics on the test dataset:\n")
        for metric_name, metric in metrics_eval.items():
            # Calculate average and variance of the performance metrics
            avg_performance = np.mean(metric)
            var_performance = np.var(metric)
            f.write(f"Results for {metric_name}\n")
            f.write("\n".join([f"Seed {seed}: {score}" for seed, score in zip(cfg.seeds, metric)]))
            f.write(f"\nAverage performance: {avg_performance}\n")
            f.write(f"Variance in performance: {var_performance}\n\n")

if __name__ == "__main__":
    main()
