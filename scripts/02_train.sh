#!/bin/bash

python3 src/train.py --config_file=config/01_hipercel.yaml
python3 src/train.py --config_file=config/02_membra.yaml
python3 src/train.py --config_file=config/03_sclerosis.yaml
python3 src/train.py --config_file=config/04_normal.yaml
python3 src/train.py --config_file=config/05_podoc.yaml
python3 src/train.py --config_file=config/06_cresce.yaml