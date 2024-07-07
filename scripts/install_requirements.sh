#!/bin/bash

# Check if the symbolic link already exists
if [ ! -L "./pipelines" ]; then
  # If it doesn't exist, create the symbolic link
  ln -s ./services/airflow/dags ./pipelines
fi
pip install poetry==1.8.1
poetry install