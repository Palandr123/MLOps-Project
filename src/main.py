import random

import hydra
import numpy as np
import pandas as pd
import torch

from model import load_features, train, log_metadata


def run(args):
    cfg = args

    # Ensure reproducibility by setting random seeds
    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)

    train_data_version = cfg.train_data_version
    X_train, y_train = load_features(name="features_target", version=train_data_version)

    test_data_version = cfg.test_data_version
    X_test, y_test = load_features(name="features_target", version=test_data_version)

    # Overwrite args.model.params.input_units with the number of features
    cfg.model.params.module__input_size = [X_train.shape[1]]
    print(int(X_train.iloc[:, cfg.model.params.module__region_idx].nunique().iloc[0]))
    print(int(X_test.iloc[:, cfg.model.params.module__region_idx].nunique().iloc[0]))
    cfg.model.params.module__num_regions = [int(X_train.iloc[:, cfg.model.params.module__region_idx].nunique().iloc[0])]

    gs = train(X_train, y_train, cfg=cfg)

    # # Ensure consistent logging (uncomment if log_metadata is defined)
    log_metadata(cfg, gs, X_train, y_train, X_test, y_test)


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg=None):
    run(cfg)


if __name__ == "__main__":
    main()
