# 0 installation
Like in the original readme

# 1 data processing
`competing_methods/sciplex/chemcpa/chemCPA/experiments/dom_experiments/combine_adata_biolord.ipynb`
`competing_methods/sciplex/chemcpa/chemCPA/experiments/dom_experiments/compute_embedding_rdkit.ipynb`
`competing_methods/sciplex/chemcpa/chemCPA/experiments/dom_experiments/generate_splits.ipynb`

Adjust dataset_path in `competing_methods/sciplex/chemcpa/chemCPA/config/dataset/biolord.yaml` for the desired split.

# 2 execution
`python chemCPA/train_hydra.py`

# 3 results analysis
`/Users/arturszalata/PycharmProjects/ot_pert_reproducibility/competing_methods/sciplex/chemcpa/chemCPA/load_lightning.ipynb`
