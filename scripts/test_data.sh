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
done

# Cleanup
mv configs/sample_data.yaml.bkp configs/sample_data.yaml
python src/data.py > /dev/null