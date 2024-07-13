import importlib

from zenml.client import Client
from sklearn.model_selection import KFold, GridSearchCV


def load_features(name, version, size = 1):
    client = Client()

    # Fetch all artifacts with the specified name and version
    artifacts = client.list_artifacts(name=name, version=version)
    
    # Sort artifacts by version if needed
    artifacts = sorted(artifacts, key=lambda x: x.version, reverse=True)

    df = artifacts[0].load()
    df = df.sample(frac = size, random_state = 88)

    print("size of df is ", df.shape)
    print("df columns: ", df.columns)

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    print("shapes of X,y = ", X.shape, y.shape)

    return X, y


def train(X_train, y_train, cfg):

    # Define the model hyperparameters
    params = cfg.model.params

    # Train the model
    module_name = cfg.model.module_name
    class_name  = cfg.model.class_name

    # Load "module.submodule.MyClass"
    class_instance = getattr(importlib.import_module(module_name), class_name)

    estimator = class_instance(**params)

    # Grid search with cross validation
    cv = KFold(n_splits=cfg.model.folds, random_state=cfg.random_state, shuffle=True)

    param_grid = dict(params)

    scoring = list(cfg.model.metrics.values())

    evaluation_metric = cfg.model.evaluation_metric

    gs = GridSearchCV(
        estimator = estimator,
        param_grid = param_grid,
        scoring = scoring,
        n_jobs = cfg.cv_n_jobs,
        refit = evaluation_metric,
        cv = cv,
        verbose = 1,
        return_train_score = True
    )

    gs.fit(X_train, y_train)

    return gs
