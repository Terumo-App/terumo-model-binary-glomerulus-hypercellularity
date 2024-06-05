REM Run Python files
@REM call python3 -m src.train --config_file=config/01_hipercel.yaml
@REM call python3 -m src.train --config_file=config/02_membra.yaml
call python -m src.train --config_file=config/03_sclerosis.yaml
@REM call python3 -m src.train --config_file=config/04_normal.yaml
call python -m src.train --config_file=config/05_podoc.yaml
@REM call python3 -m src.train --config_file=config/06_cresce.yaml



