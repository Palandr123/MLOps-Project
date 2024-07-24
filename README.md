# MLOps-Project
![Test code workflow](https://github.com/Palandr123/MLOps-Project/actions/workflows/test-code.yaml/badge.svg)

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

## NOTE
We do not have requirements.txt file because pyproject.toml and poetry.lock are the replacements for it. We prefer to use Poetry and our TA confirmed that we can use it.

We also have not pushed any ```*.pkl``` file since they are too large to be stored in GitHub.