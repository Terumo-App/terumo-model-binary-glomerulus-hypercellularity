build_dataset:
	bash scripts/00_organize_data_folders.sh
	python3 -m data_processing.01_generate_dataset

build_test_dataset:
	python3 -m data_processing.01_generate_dataset data test_raw test_binary

train:
	bash scripts/02_train.sh

test:
	bash scripts/03_trial_run.sh

inference:
	mkdir -p logs
	python3 -m scripts.perform_inference_all --checkpoint_dir="artifacts" --test_dataset_root="data/test_binary" \
	 --device="cuda" --mode="min_loss" > logs/inference_log_min_loss \
	 --best_model_dir="best_model" --overwrite_best_model_dir --dropout=0.5 --mcd=100
	python3 -m scripts.perform_inference_all --checkpoint_dir="artifacts" --test_dataset_root="data/test_binary" \
	--device="cuda" --mode="max_acc" > logs/inference_log_max_acc --dropout=0.5 --mcd=100

# Delete all compiled Python files
clean:
	find . -type d -name "__pycache__" ! -path "./env_tcc_eeg/*" -exec rm -rv {} \;
