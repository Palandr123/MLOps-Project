#!/bin/bash

# Backup config file
cp configs/sample_data.yaml configs/sample_data.yaml.bkp

# Validate samples
for i in $(seq 1 5)
do
    echo "Testing batch #$i"
    # Modify config
    sed -i "s/sample_num.*/sample_num: $i/" configs/sample_data.yaml
    python src/data.py

    if [ $? -eq 0 ]
    then
        echo "Validation succeeded. Saving a new version of data"
        dvc add data/samples/sample.csv
        git add data/samples/sample.csv.dvc
        git commit -m "Added batch $i"
        git push
        git tag -f -a "v1.0.$i" -m "Added batch $i"
        git push -f --tags
        dvc push
    else
        echo "Validation failed"
    fi
done

# Cleanup
mv configs/sample_data.yaml.bkp configs/sample_data.yaml
python src/data.py > /dev/null