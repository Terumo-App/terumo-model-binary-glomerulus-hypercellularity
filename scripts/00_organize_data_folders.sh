#!/bin/bash

# Organize the "data/raw/Terumo_<Class>_<Pigmentation>" folders into the "data/raw/<Class>" folders
# (with class names fixed)

classes=("Normal" "Crescent" "Hypercelularidade" "Podocitopatia" "Sclerosis" "Membranous")
target_classes=("Normal" "Crescent" "Hypercellularity" "Podocytopathy" "Sclerosis" "Membranous")
colorings=( "AZAN" "HE" "PAMS" "PAS" "PICRO" )

for ((i=0;i<${#classes[@]};++i)); do
	mkdir data/raw/${target_classes[i]}
    for coloring in "${colorings[@]}"; do
      dataset_folder="data/raw/Terumo_${classes[i]}_${coloring}"
      if [[ -d "${dataset_folder}" ]]; then
          echo "Moving '${dataset_folder}' to 'data/raw/${target_classes[i]}'"
          mv ${dataset_folder} data/raw/${target_classes[i]}
      fi
      dataset_folder="data/raw/Terumo_${classes[i]}__${coloring}"
      if [[ -d "${dataset_folder}" ]]; then
          echo "Moving '${dataset_folder}' to 'data/raw/${target_classes[i]}'"
          mv ${dataset_folder} data/raw/${target_classes[i]}
      fi
    done
done
