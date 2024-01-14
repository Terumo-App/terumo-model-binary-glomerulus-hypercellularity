build_dataset:
	python -m data_processing.01_generate_dataset

train:
	bash scripts/02_train.sh

test:
	bash scripts/03_trial_run.sh

# Delete all compiled Python files
clean:
	find . -type d -name "__pycache__" ! -path "./env_tcc_eeg/*" -exec rm -rv {} \;