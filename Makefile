build_dataset:
	bash scripts/00_organize_data_folders.sh
	bash scripts/01_generate_dataset_copy.sh

train:
	bash scripts/02_train.sh

test:
	bash scripts/03_trial_run.sh

# Delete all compiled Python files
clean:
	find . -type d -name "__pycache__" ! -path "./env_tcc_eeg/*" -exec rm -rv {} \;