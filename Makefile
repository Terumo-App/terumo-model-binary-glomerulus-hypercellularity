build_dataset:
	bash scripts/00_organize_data_folders.sh
	bash scripts/01_generate_dataset_copy.sh

train:
	bash scripts/02_train.sh

test:
	bash scripts/03_trial_run.sh

inference:
	python3 -m scripts.perform_inference_all --checkpoint_dir="artifacts" --test_dataset_root="data/test_binary" --device="cuda" --mode="min_loss" > logs/inference_log_min_loss
	python3 -m scripts.perform_inference_all --checkpoint_dir="artifacts" --test_dataset_root="data/test_binary" --device="cuda" --mode="max_acc" > logs/inference_log_max_acc

# Delete all compiled Python files
clean:
	find . -type d -name "__pycache__" ! -path "./env_tcc_eeg/*" -exec rm -rv {} \;