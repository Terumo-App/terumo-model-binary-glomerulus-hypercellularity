

REM Run Python files
call python src/train.py --config_file=config/03_sclerosis.yaml
call python src/train.py --config_file=config/04_normal.yaml
call python src/train.py --config_file=config/01_hipercel.yaml
call python src/train.py --config_file=config/02_membra.yaml



