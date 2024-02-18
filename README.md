# Terumo: binary classifiers for glomerular data

- Usage:

1. Copy all the dataset folders to the `data/raw` folder.
2. Run `make build_dataset`. This will run in two steps: 
    1. First, it will arrange the dataset into 6 sub-folders of the `data/raw` folder, according to class names:
        1. `Hypercellularity`
        2. `Membranous`
        3. `Sclerosis`
        4. `Normal`
        5. `Podocytopathy`
        6. `Crescent`
    2. Then, it will create the 6 copies that will be used to train each binary classifier
3. (Optional) If you want a test run (of 1 epoch, and only a single model) run `make test`
4. Run `make train` to train all 6 models sequentially, in the order above (Hypercellularity, Membranous, Sclerosis, Normal, Podocytopathy, Crescent)