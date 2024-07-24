import hydra
import pandas as pd
import json
import os


def save_uniques_to_json(df, cat_cols):
    
    out = dict()
    for col in cat_cols:
        out[col] = list(df[col].dropna().unique())
    print(os.getcwd())
    with open("./configs/cat_cols_unique.json", "w") as f:
        json.dump(out, f, indent=4)

def get_unique_values():
    with open("./configs/cat_cols_unique.json", "r") as f:
        unique_vals = json.load(f)
    return unique_vals

def main():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="data")
    df = pd.read_csv("data/raw/vehicles.csv")
    save_uniques_to_json(df, cfg.data.categorical_cols)


if __name__ == "__main__":
    main()