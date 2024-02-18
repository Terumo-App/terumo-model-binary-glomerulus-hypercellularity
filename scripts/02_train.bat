REM Run Python files
call python3 -m src.train --config_file=config/01_hipercel.yaml
call python3 -m src.train --config_file=config/02_membra.yaml
call python3 -m src.train --config_file=config/03_sclerosis.yaml
call python3 -m src.train --config_file=config/04_normal.yaml
call python3 -m src.train --config_file=config/05_podoc.yaml
call python3 -m src.train --config_file=config/06_cresce.yaml



