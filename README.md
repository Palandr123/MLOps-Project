# MLOps-Project
![Test code workflow](https://github.com/Palandr123/MLOps-Project/actions/workflows/test-code.yaml/badge.svg)
![Validate model workflow](https://github.com/Palandr123/MLOps-Project/actions/workflows/validate-model.yaml/badge.svg)
## Repository structure
```
├───README.md          # Repo docs
├───.gitignore         # gitignore file
├───pyproject.toml     # Python dependencies using Poetry
├───poetry.lock        # Python packages using Poetry
├───configs            # Hydra configuration management
├───data               # All data
├───docs               # Project docs like reports or figures
├───models             # ML models
├───notebooks          # Jupyter notebooks
├───outputs            # Outputs of Hydra
├───pipelines          # A Soft link to DAGs of Apache Airflow
├───reports            # Generated reports 
├───scripts            # Shell scripts (.sh)
├───services           # Metadata of services (PostgreSQL, Feast, Apache airflow, ...etc)
├───sql                # SQL files
├───src                # Python scripts
└───tests              # Scripts for testing Python code
```

## Installation guide
1. Create Python virtual environment (conda, venv, poetry)
2. Run the following command:
    ```sh scripts/install.sh```

## Deployment
### API-only
To deploy Flask based API you just need to run the following command:

```
mlflow run . --env-manager local -e deploy
```

It will build and run docker image with our model on port 5152. You can check the deployment using the command:

```
curl -X POST -d '{"inputs": {"year": 0.9743589743589745, "odometer": -1.0009959752152948, "ohe__title_status_clean": 1.0, "ohe__title_status_lien": 0.0, "ohe__title_status_missing": 0.0, "ohe__title_status_parts only": 0.0, "ohe__title_status_rebuilt": 0.0, "ohe__title_status_salvage": 0.0, "ohe__transmission_automatic": 0.0, "ohe__transmission_manual": 0.0, "ohe__transmission_other": 1.0, "ohe__fuel_diesel": 0.0, "ohe__fuel_electric": 0.0, "ohe__fuel_gas": 1.0, "ohe__fuel_hybrid": 0.0, "ohe__fuel_other": 0.0, "ohe__state_ak": 0.0, "ohe__state_al": 0.0, "ohe__state_ar": 0.0, "ohe__state_az": 0.0, "ohe__state_ca": 0.0, "ohe__state_co": 0.0, "ohe__state_ct": 0.0, "ohe__state_dc": 0.0, "ohe__state_de": 0.0, "ohe__state_fl": 0.0, "ohe__state_ga": 0.0, "ohe__state_hi": 0.0, "ohe__state_ia": 0.0, "ohe__state_id": 0.0, "ohe__state_il": 0.0, "ohe__state_in": 0.0, "ohe__state_ks": 0.0, "ohe__state_ky": 0.0, "ohe__state_la": 0.0, "ohe__state_ma": 0.0, "ohe__state_md": 0.0, "ohe__state_me": 0.0, "ohe__state_mi": 0.0, "ohe__state_mn": 0.0, "ohe__state_mo": 0.0, "ohe__state_ms": 0.0, "ohe__state_mt": 0.0, "ohe__state_nc": 0.0, "ohe__state_nd": 0.0, "ohe__state_ne": 0.0, "ohe__state_nh": 0.0, "ohe__state_nj": 0.0, "ohe__state_nm": 0.0, "ohe__state_nv": 0.0, "ohe__state_ny": 1.0, "ohe__state_oh": 0.0, "ohe__state_ok": 0.0, "ohe__state_or": 0.0, "ohe__state_pa": 0.0, "ohe__state_ri": 0.0, "ohe__state_sc": 0.0, "ohe__state_sd": 0.0, "ohe__state_tn": 0.0, "ohe__state_tx": 0.0, "ohe__state_ut": 0.0, "ohe__state_va": 0.0, "ohe__state_vt": 0.0, "ohe__state_wa": 0.0, "ohe__state_wi": 0.0, "ohe__state_wv": 0.0, "ohe__state_wy": 0.0, "ohe__manufacturer_acura": 0.0, "ohe__manufacturer_alfa-romeo": 0.0, "ohe__manufacturer_aston-martin": 0.0, "ohe__manufacturer_audi": 0.0, "ohe__manufacturer_bmw": 0.0, "ohe__manufacturer_buick": 0.0, "ohe__manufacturer_cadillac": 0.0, "ohe__manufacturer_chevrolet": 0.0, "ohe__manufacturer_chrysler": 0.0, "ohe__manufacturer_datsun": 0.0, "ohe__manufacturer_dodge": 0.0, "ohe__manufacturer_ferrari": 0.0, "ohe__manufacturer_fiat": 0.0, "ohe__manufacturer_ford": 0.0, "ohe__manufacturer_gmc": 0.0, "ohe__manufacturer_harley-davidson": 0.0, "ohe__manufacturer_honda": 0.0, "ohe__manufacturer_hyundai": 0.0, "ohe__manufacturer_infiniti": 0.0, "ohe__manufacturer_jaguar": 0.0, "ohe__manufacturer_jeep": 0.0, "ohe__manufacturer_kia": 0.0, "ohe__manufacturer_land rover": 0.0, "ohe__manufacturer_lexus": 0.0, "ohe__manufacturer_lincoln": 0.0, "ohe__manufacturer_mazda": 0.0, "ohe__manufacturer_mercedes-benz": 0.0, "ohe__manufacturer_mercury": 0.0, "ohe__manufacturer_mini": 0.0, "ohe__manufacturer_mitsubishi": 0.0, "ohe__manufacturer_nissan": 0.0, "ohe__manufacturer_pontiac": 0.0, "ohe__manufacturer_porsche": 0.0, "ohe__manufacturer_ram": 0.0, "ohe__manufacturer_rover": 0.0, "ohe__manufacturer_saturn": 0.0, "ohe__manufacturer_subaru": 0.0, "ohe__manufacturer_tesla": 0.0, "ohe__manufacturer_toyota": 1.0, "ohe__manufacturer_volkswagen": 0.0, "ohe__manufacturer_volvo": 0.0, "label__WMI": 849.0, "label__VDS": 7533.0, "label__model": 4626.0, "label__region": 235.0, "lat_sin": -0.988860535448249, "lat_cos": -0.14884502488495285, "long_sin": -0.5248775448593879, "long_cos": -0.8511777504742363, "posting_date_month_sin": 0.49999999999999994, "posting_date_month_cos": -0.8660254037844387, "posting_date_day_sin": 0.39435585511331855, "posting_date_day_cos": 0.9189578116202306}}' -H 'Content-Type: application/json' http://localhost:5152/invocations
```

### Gradio demo
Alternatively, you can access our Flask API and use Gradio-based frontend to interact with our model.

For that you firstly need to launch Flask API:

```
python api/app.py
```

Then you can deploy the Gradio itself:

```
python src/app.py
```

After that you could open the `localhost:5155` and interact with our model via frontent.


## NOTE
We do not have requirements.txt file because pyproject.toml and poetry.lock are the replacements for it. We prefer to use Poetry and our TA confirmed that we can use it.

We also have not pushed any ```*.pkl``` file since they are too large to be stored in GitHub.