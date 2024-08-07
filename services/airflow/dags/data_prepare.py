import pandas as pd
from typing_extensions import Tuple, Annotated
from zenml import step, pipeline, ArtifactConfig
from src.data import read_datastore, preprocess_data, validate_features, load_features


@step(enable_cache=False)
def extract() -> Tuple[
    Annotated[
        pd.DataFrame, ArtifactConfig(name="extracted_data", tags=["data_preparation"])
    ],
    Annotated[str, ArtifactConfig(name="data_version", tags=["data_preparation"])],
]:

    df, version = read_datastore()

    return df, str(version)


@step(enable_cache=False)
def transform(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[
        pd.DataFrame, ArtifactConfig(name="input_features", tags=["data_preparation"])
    ],
    Annotated[
        pd.DataFrame, ArtifactConfig(name="input_target", tags=["data_preparation"])
    ],
]:

    # Your data transformation code
    X, y = preprocess_data(df)

    return X, y


@step(enable_cache=False)
def validate(
    X: pd.DataFrame, y: pd.DataFrame
) -> Tuple[
    Annotated[
        pd.DataFrame,
        ArtifactConfig(name="valid_input_features", tags=["data_preparation"]),
    ],
    Annotated[
        pd.DataFrame, ArtifactConfig(name="valid_target", tags=["data_preparation"])
    ],
]:

    X, y = validate_features(X, y)

    return X, y


@step(enable_cache=False)
def load(
    X: pd.DataFrame, y: pd.DataFrame, version: str
) -> Annotated[
    pd.DataFrame, ArtifactConfig(name="features_target", tags=["data_preparation"])
]:

    return pd.concat([X, y], axis=1)


@pipeline()
def data_prepare_pipeline():
    df, version = extract()
    X, y = transform(df)
    X, y = validate(X, y)
    df = load(X, y, version)


if __name__ == "__main__":
    data_prepare_pipeline()
