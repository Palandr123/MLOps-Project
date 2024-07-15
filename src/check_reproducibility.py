import random
from pathlib import Path

import numpy as np
import torch
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error
from omegaconf import open_dict

from model import load_features, train

def evaluate_model(gs, X_test, y_test):
    X_test_np = X_test.values.astype(np.float32)
    y_test_np = y_test.values.astype(np.float32).reshape(-1, 1)
    predictions = gs.predict(X_test_np)
    mse = mean_squared_error(y_test_np, predictions)
    return mse

@hydra.main(config_path="../configs", config_name="check_reproducibility")
def main(cfg: DictConfig):
    results_dir = Path.cwd() / Path(cfg.results_dir)
    print(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    performance_metrics = []
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

        # Overwrite args.model.params.input_units with the number of features
        main_config.model.params.module__input_size = [X_train.shape[1]]

        # Train the model
        gs = train(X_train, y_train, cfg=main_config)

        # Get the best evaluation metric
        evaluation_metric = main_config.model.evaluation_metric
        best_score = gs.cv_results_['mean_test_' + evaluation_metric][gs.best_index_]
        performance_metrics.append(best_score)
        print(f"Seed: {seed}, Best {evaluation_metric}: {best_score}")

    # Calculate average and variance of the performance metrics
    avg_performance = np.mean(performance_metrics)
    var_performance = np.var(performance_metrics)

    # Save the results to a text file
    results_file = results_dir / f"reproducibility_results_{cfg.model_config}.txt"
    with open(str(results_file), "w") as f:
        f.write(f"Performance metrics over {len(cfg.seeds)} experiments with different seeds:\n")
        f.write("\n".join([f"Seed {seed}: {score}" for seed, score in zip(cfg.seeds, performance_metrics)]))
        f.write(f"\n\nAverage performance: {avg_performance}\n")
        f.write(f"Variance in performance: {var_performance}\n")

if __name__ == "__main__":
    main()
