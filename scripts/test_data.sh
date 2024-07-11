#!/bin/bash

# Backup config file
cp configs/sample_data.yaml configs/sample_data.yaml.bkp

# Get sample_num from config
sample_num=$(yq .sample_num configs/sample_data.yaml)

# Validate samples
echo "Testing batch #$sample_num"
python src/data.py

if [ $? -eq 0 ]
then
    echo "Validation succeeded. Saving a new version of data"
    dvc add data/samples/sample.csv
    git add data/samples/sample.csv.dvc
    git commit -m "Added batch $sample_num"
    git push
    git tag -f -a "v1.0.$sample_num" -m "Added batch $sample_num"
    git push -f --tags
    dvc push
else
    echo "Validation failed"
fi

# Cleanup
mv configs/sample_data.yaml.bkp configs/sample_data.yaml
python src/data.py > /dev/null