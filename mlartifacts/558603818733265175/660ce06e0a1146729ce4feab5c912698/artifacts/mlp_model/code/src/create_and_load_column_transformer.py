import pandas as pd
import numpy as np
import zenml
from zenml.integrations.sklearn.materializers.sklearn_materializer import SklearnMaterializer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from hydra import initialize, compose
from sklearn.impute import SimpleImputer


initialize("../configs")

cfg = compose(config_name="data")

df = pd.read_csv("data/raw/vehicles.csv")

labels = cfg.data.target_cols

df = df[(df[labels[0]] >= cfg.data.target_low) & (df[labels[0]] <= cfg.data.target_high)]
df = df.dropna(subset=cfg.data.drop_rows)
df = df.reset_index(drop=True)
X_cols = [col for col in df.columns if col not in labels]
X = df[X_cols]
y = df[labels]

for dt_feature in list(cfg.data.dt_feature):
    X[dt_feature] = pd.to_datetime(X[dt_feature])
    X[dt_feature] = X[dt_feature].apply(lambda x: np.nan if x is pd.NaT else x.timestamp())

X["WMI"] = X["VIN"].apply(lambda x: x[:3])
X["VDS"] = X["VIN"].apply(lambda x: x[3:8])

most_freq_imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
for imp_m_f in cfg.data.impute_most_frequent:
    X[[imp_m_f]] = most_freq_imp.fit_transform(X[[imp_m_f]])

median_imp = SimpleImputer(missing_values=np.nan, strategy="median")
for imp_median in cfg.data.impute_median:
    X[[imp_median]] = median_imp.fit_transform(X[[imp_median]])

mean_imp = SimpleImputer(missing_values=np.nan, strategy="mean")
for imp_mean in cfg.data.impute_mean:
    X[[imp_mean]] = mean_imp.fit_transform(X[[imp_mean]])


categorical_transformer = ColumnTransformer([
    ("ohe", OneHotEncoder(handle_unknown='ignore'), list(cfg.data.ohe_cols)),
    ("label", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), list(cfg.data.label_cols))
])

cat_transform_model = categorical_transformer.fit(X)

zenml.save_artifact(data=cat_transform_model, name="cat_transform", tags=["v1"], materializer=SklearnMaterializer)