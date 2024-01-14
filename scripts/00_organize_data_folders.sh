#!/bin/bash

# Organize the "data/raw/Terumo_<Class>_<Pigmentation>" folders into the "data/raw/<Class>" folders
# (with class names fixed)

classes=("Normal" "Crescent" "Hypercelularidade" "Podocitopatia" "Sclerosis" "Membranous")
target_classes=("Normal" "Crescent" "Hypercellularity" "Podocitopathy" "Sclerosis" "Membranous")

for dataset_folder in data/raw/Terumo_*; do
    for ((i=0;i<${#classes[@]};++i)); do
        if [[ $dataset_folder == *"Terumo_${classes[i]}"* ]]; then
            echo "Moving ${dataset_folder} to data/raw/${target_classes[i]}"
            mv $dataset_folder data/raw/${target_classes[i]}
        fi
    done
done