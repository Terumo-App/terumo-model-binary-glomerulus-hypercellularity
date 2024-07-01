#!/bin/bash

python3 -m src.train --config_file=config/bracs/0_N.yaml
python3 -m src.train --config_file=config/bracs/1_PB.yaml
python3 -m src.train --config_file=config/bracs/2_UDH.yaml
python3 -m src.train --config_file=config/bracs/3_FEA.yaml
python3 -m src.train --config_file=config/bracs/4_ADH.yaml
python3 -m src.train --config_file=config/bracs/5_DCIS.yaml
python3 -m src.train --config_file=config/bracs/6_IC.yaml