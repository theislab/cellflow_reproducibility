# !/bin/bash

echo "Start new_hyper_scvi.sh."

python new_hyper_scvi.py \
    --output_dir=/home/haicu/soeren.becker/repos/ot_pert_reproducibility/norman2019/norman_preprocessed_adata \
    --filename_adata_train=adata_train_split_0.h5ad \
    --filename_adata_val=adata_val_split_0.h5ad \
    --filename_adata_test=adata_test_split_0.h5ad

echo "Finished new_hyper_scvi.sh."