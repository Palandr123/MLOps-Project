import hydra

from model import load_features


def run(args):
    cfg = args

    train_data_version = cfg.train_data_version

    X_train, y_train = load_features(name = "features", version=train_data_version)

    test_data_version = cfg.test_data_version

    X_test, y_test = load_features(name = "features_target", version=test_data_version)


@hydra.main(config_path="../configs", config_name="main", version_base=None) # type: ignore
def main(cfg=None):
    run(cfg)


if __name__=="__main__":
    main()
