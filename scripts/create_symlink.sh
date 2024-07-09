#!/bin/bash

# Check if the symbolic link already exists
if [ ! -L "./pipelines" ]; then
  # If it doesn't exist, create the symbolic link
  ln -s ./services/airflow/dags ./pipelines
fi