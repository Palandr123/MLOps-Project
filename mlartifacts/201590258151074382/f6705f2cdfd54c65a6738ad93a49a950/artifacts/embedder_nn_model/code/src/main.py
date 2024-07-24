import random

import hydra
import numpy as np
import torch
import zenml
import giskard

from model import load_features, train, log_metadata


def get_num_unique(cat_transformer, column):
    ordinal_encoder = cat_transformer.named_transformers_['label']
    index = cat_transformer.transformers[1][2].index(column)
    categories = ordinal_encoder.categories_[index]
    return len(categories)


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

    if cfg.model.has_embeds:
        for i, col in enumerate(list(X_train.columns)):
            if col == cfg.model.region_column:
                cfg.model.params.module__region_idx = [i]
            elif col == cfg.model.wmi_column:
                cfg.model.params.module__wmi_idx = [i]
            elif col == cfg.model.vds_column:
                cfg.model.params.module__vds_idx = [i]
            elif col == cfg.model.model_column:
                cfg.model.params.module__model_idx = [i]
        client = zenml.client.Client()
        artifacts = client.list_artifacts(name="cat_transform")
        cat_transformer = artifacts[-1].load()
        
        cfg.model.params.module__num_regions = [get_num_unique(cat_transformer, "region")]
        cfg.model.params.module__num_wmis = [get_num_unique(cat_transformer, "WMI")]
        cfg.model.params.module__num_vds = [get_num_unique(cat_transformer, "VDS")]
        cfg.model.params.module__num_models = [get_num_unique(cat_transformer, "model")]

    gs = train(X_train, y_train, cfg=cfg)

    # # Ensure consistent logging (uncomment if log_metadata is defined)
    log_metadata(cfg, gs, X_train, y_train, X_test, y_test)


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg=None):
    run(cfg)


if __name__ == "__main__":
    main()
