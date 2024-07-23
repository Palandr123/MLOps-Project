#!/bin/bash

array=(55 56 57 58 59)

for sample in ${array[*]}
do
    mlflow run . --env-manager local -e predict -P example_version=$sample -P port=5152 -P random_state=15
done