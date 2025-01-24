import scanpy as sc
import numpy as np
import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm 
from collections import defaultdict
import itertools
import argparse

import sys 
sys.path.insert(0, "..")
from utils import get_DE_genes

import os
import pickle as pkl
from cfp import preprocessing as cfpp

def split_and_save_adata(args):
    """The function to split and 
    """
    # Input parameters 
    rng = np.random.default_rng(seed=42)  
    hvg = args.hvg
    pca_dim = args.pca_dim
    ms = args.ms

    # The pathways and the ood_condition 
    ood_pathway = args.ood_pathway
    ood_cell_type = args.ood_cell_type
    # Pathway string use to parse .h5ad
    pathway = 'IFNG_IFNB_TNFA_TGFB_INS'
    
    # The final output dir
    output_dir = output_dir = "/lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_pathway_cell_type/" + ood_cell_type + "_" + ood_pathway 
    os.makedirs(output_dir, exist_ok=True)

    genes_from_paper = [
        "AHNAK", "RNF213", "APOL6", "ASTN2", "B2M", "CFH", "CXCL9", "DENND4A", 
        "DOCK9", "EFNA5", "ERAP2", "FAT1", "GBP1", "GBP4", "HAPLN3", "HSPG2", 
        "IDO1", "IFI6", "IRF1", "LAP3", "LI", "LINC02328", "MAGI1", "MUC4", 
        "NLRC5", "NUB1", "PARP14", "PARP9", "RARRES1", "RNF213", "ROR1", "SCN9A", 
        "SERPING1", "ST5", "STAT1", "TAP1", "TAP2", "THBS1", "THSD4", "TPM1", "VCL", 
        "WARS", "XRN1"
    ]

    # Read the data 
    datasets = []
    for pw in pathway.split('_'):
        if ms == None:
            data_path = '/lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/ood_cell_type/satija_merged/' + pw + '_Perturb_seq.h5ad' # '_Perturb_seq_ms_0.5.h5ad'
        else:
            data_path = '/lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/ood_cell_type/satija_merged/' + pw + '_Perturb_seq_ms_' + str(ms) + '.h5ad'
        print('Loading dataset from ' + data_path)
        dataset = sc.read_h5ad(data_path)
        dataset.obs['pathway'] = pw
        datasets.append(dataset)

    # Create common anndata 
    adata = ad.concat(datasets, join='outer')
    print('Datasets concatenated')
    
    # Make the variable names unique
    adata.obs_names_make_unique()

    # Drop unused columns 
    columns_to_drop = ['orig.ident', 'nCount_RNA', 'nFeature_RNA', 'sample', 'percent.mito', 'sample_ID', 'Batch_info', 'bc1_well', 'bc2_well', 'bc3_well', 'guide', 'mixscale_score', 'RNA_snn_res.0.9', 'seurat_clusters']
    adata.obs.drop(columns=columns_to_drop, inplace=True)
    print('Unnecessary columns dropped')
                
    # Add specific columns to adata.obs 
    adata.obs['condition'] = adata.obs.apply(lambda x: "_".join([x.cell_type, x.pathway, x.gene]), axis=1)
    adata.obs['background'] = adata.obs.apply(lambda x: "_".join([x.cell_type, x.pathway]), axis=1)
    
    # Filter very rare perturbation classes 
    condition_counts = adata.obs['condition'].value_counts()
    filtered_conditions = condition_counts[condition_counts >= 100]  # Keep only some conditions
    adata = adata[adata.obs['condition'].isin(filtered_conditions.index)]
    print(f"Filtered adata for perturbation count: {adata.shape[0]} observations remaining")

    adata.layers["counts"] = adata.layers["counts"].astype(np.float32)
    adata.X = csr_matrix(adata.layers["counts"])
    del adata.layers['counts']
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Highly variable gene selection 
    highly_var_genes = {}
    for bg in tqdm(adata.obs['background'].unique()):
        temp = adata[adata.obs['background'] == bg, :]
        sc.pp.highly_variable_genes(temp, inplace=True, n_top_genes=hvg)
        temp = adata[:, temp.var["highly_variable"]==True]
        highly_var_genes[bg] = set(temp.var.index)
        del temp 
        
    # Compile the union list and add the genes from the paper 
    combined_set = set()
    for key in highly_var_genes:
        combined_set.update(highly_var_genes[key])
    combined_set = combined_set.union(set(genes_from_paper))
    adata = adata[:, adata.var.index.isin(combined_set)]
    print("Highly variable genes selected")

    adata = get_DE_genes(adata, by='condition', covariate='background')
    print('DE genes calculated')    

    for col in adata.obs.select_dtypes(include=["category"]):
        adata.obs[col].cat.remove_unused_categories()

    # Filter the condition
    ood_condition = ood_cell_type + '_' + ood_pathway
    filtered_conditions = adata.obs['condition'].unique() # unnecessary
    perturbations = list(adata.obs[adata.obs['gene'] != 'NT']["condition"].unique())
    ood_conditions = [c for c in perturbations if c.startswith(ood_condition) and c in filtered_conditions]
    
    adata.obs["is_ood"] = adata.obs.apply(lambda x: x["condition"] in ood_conditions, axis=1)
    adata_train = adata[~adata.obs["is_ood"]]
    adata_ood = adata[adata.obs["is_ood"]]
    print(f"Dataset split: {adata_train.shape} train, {adata_ood.shape} ood")
    
    rng = np.random.default_rng(0)
    split_dfs = []
    adata_train.obs["split"] = "not_included"

    for c in adata_train.obs["condition"].unique():
        n_cells = adata_train[(adata_train.obs["condition"]==c)].n_obs
        # Subsample the controls, not treated 
        if c.endswith('_NT'):
            idx_test = rng.choice(np.arange(n_cells), 500, replace=False)
            remaining_idcs = set(np.arange(n_cells)) - set(idx_test)
            adata_train.obs.loc[adata_train.obs['condition'] == c, 'split'] = ["test" if idx in idx_test else "train" for idx in range(n_cells)]
        elif n_cells>300:
            idx_test = rng.choice(np.arange(n_cells), 100, replace=False)
            adata_train.obs.loc[adata_train.obs['condition'] == c, 'split'] = ["test" if idx in idx_test else "train" for idx in range(n_cells)]

    adata_train_final = adata_train[adata_train.obs["split"]=="train"]
    adata_test_final = adata_train[adata_train.obs["split"]=="test"]
    # For evluation
    adata_ood_final = ad.concat((adata_ood, adata_test_final[adata_test_final.obs["condition"].str.endswith('_NT')]))
    adata_ood_final.uns = adata_ood.uns
    print(adata_ood_final.obs['control'].value_counts())
    
    adata_train_final = adata_train_final[~((adata_train_final.obs['pathway'] == ood_pathway) & (adata_train_final.obs['cell_type'] == ood_cell_type)), :]
    adata_test_final = adata_test_final[~((adata_test_final.obs['pathway'] == ood_pathway) & (adata_test_final.obs['cell_type'] == ood_cell_type)), :]
    adata_ood_final = adata_ood_final[((adata_ood_final.obs['pathway'] == ood_pathway) & (adata_ood_final.obs['cell_type'] == ood_cell_type)), :]
    
    # Compute centered pca on the training data
    cfpp.centered_pca(adata_train_final, n_comps=pca_dim)
    print("Run centred PCA")

    # Initialize a log-count layer
    adata_train_final.layers["X_log1p"] = adata_train_final.X.copy()
    # Training data mean 
    adata_train_final_mean = adata_train_final.varm["X_mean"].flatten()

    # Define the gene means for the anndata train and ood as the training one 
    adata_ood_final.varm["X_mean"] = adata_train_final.varm["X_mean"]
    adata_test_final.varm["X_mean"] = adata_train_final.varm["X_mean"]

    # Center both test and ood data by the mean of the training set and compute PCA based on this
    adata_test_final.layers["centered_X"] = csr_matrix(adata_test_final.X.toarray() - adata_train_final_mean)
    adata_ood_final.layers["centered_X"] = csr_matrix(adata_ood_final.X.toarray() - adata_train_final_mean)
    adata_test_final.obsm["X_pca"] = np.matmul(adata_test_final.layers["centered_X"].toarray(), adata_train_final.varm["PCs"])
    adata_ood_final.obsm["X_pca"] = np.matmul(adata_ood_final.layers["centered_X"].toarray(), adata_train_final.varm["PCs"])

    # Add the control key to the obs data frame
    adata_train_final.obs['control'] = adata_train_final.obs.apply(lambda x: x['gene'] == 'NT', axis=1)
    adata_test_final.obs['control'] = adata_test_final.obs.apply(lambda x: x['gene'] == 'NT', axis=1)
    adata_ood_final.obs['control'] = adata_ood_final.obs.apply(lambda x: x['gene'] == 'NT', axis=1)
    
    # Add embeddings 
    path_to_embeddings = os.path.join('/lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/embeddings/perturb_emb/satijas_v2', 'gene_embeddings.pkl')
    # Gene KO embeddings 
    ko_embeddings = pkl.load(open(path_to_embeddings, 'rb'))
    ko_embeddings = pd.DataFrame(ko_embeddings).T
    ko_embeddings = ko_embeddings.astype(np.float32)
    gene_embeddings_dict = dict(zip(ko_embeddings.index, ko_embeddings.values))

    # Cell line embedding 
    cell_embeddings = pd.read_csv('/lustre/groups/ml01/workspace/ot_perturbation/data/satija/embeddings/cell_line_embedding_full_ccle_300_normalized.csv', index_col=0)
    cell_embeddings = cell_embeddings.astype(np.float32)
    cell_embeddings_dict = dict(zip(cell_embeddings.index, cell_embeddings.values))
    cell_embeddings_dict = {k: v for k, v in cell_embeddings_dict.items() if k in adata_train_final.obs['cell_type'].unique()}

    # Control embedding as zero 
    gene_embeddings_dict['NT'] = np.zeros(gene_embeddings_dict['IFNG'].shape)
    pathway_embeddings = {k: v for k, v in gene_embeddings_dict.items() if k in adata_train_final.obs['pathway'].unique()}

    # Add all the embeddings to the uns of the adata 
    adata_train_final.uns['gene_emb'] = gene_embeddings_dict
    adata_train_final.uns['cell_type_emb'] = cell_embeddings_dict
    adata_train_final.uns['pathway_emb'] = pathway_embeddings

    adata_test_final.uns['gene_emb'] = gene_embeddings_dict
    adata_test_final.uns['cell_type_emb'] = cell_embeddings_dict
    adata_test_final.uns['pathway_emb'] = pathway_embeddings

    adata_ood_final.uns['gene_emb'] = gene_embeddings_dict
    adata_ood_final.uns['cell_type_emb'] = cell_embeddings_dict
    adata_ood_final.uns['pathway_emb'] = pathway_embeddings

    # Subset for cells for which we have the embeddings 
    adata_train_final = adata_train_final[adata_train_final.obs['cell_type'].isin(cell_embeddings_dict.keys()), :]
    adata_train_final = adata_train_final[adata_train_final.obs['pathway'].isin(pathway_embeddings.keys()), :]
    adata_train_final = adata_train_final[(adata_train_final.obs['gene'].isin(gene_embeddings_dict.keys()) | (adata_train_final.obs['gene'] == 'NT')), :]

    adata_test_final = adata_test_final[adata_test_final.obs['cell_type'].isin(cell_embeddings_dict.keys()), :]
    adata_test_final = adata_test_final[adata_test_final.obs['pathway'].isin(pathway_embeddings.keys()), :]
    adata_test_final = adata_test_final[(adata_test_final.obs['gene'].isin(gene_embeddings_dict.keys()) | (adata_test_final.obs['gene'] == 'NT')), :]

    adata_ood_final = adata_ood_final[adata_ood_final.obs['cell_type'].isin(cell_embeddings_dict.keys()), :]
    adata_ood_final = adata_ood_final[adata_ood_final.obs['pathway'].isin(pathway_embeddings.keys()), :]
    adata_ood_final = adata_ood_final[(adata_ood_final.obs['gene'].isin(gene_embeddings_dict.keys()) | (adata_ood_final.obs['gene'] == 'NT')), :]

    print("Shape train", adata_train_final.shape)
    print("Shape test", adata_test_final.shape)
    print("Shape ood", adata_ood_final.shape) 
    
    adata_train_final = adata_train_final[~((adata_train_final.obs['pathway'] == ood_pathway) & (adata_train_final.obs['cell_type'] == ood_cell_type)), :]
    adata_test_final = adata_test_final[~((adata_test_final.obs['pathway'] == ood_pathway) & (adata_test_final.obs['cell_type'] == ood_cell_type)), :]

    # Save the anndata 
    adata_train_final.write(os.path.join(output_dir, "adata_train_" + ood_condition + ".h5ad"))
    adata_ood_final.write(os.path.join(output_dir, "adata_ood_" + ood_condition + ".h5ad"))
    adata_test_final.write(os.path.join(output_dir, "adata_test_" + ood_condition + ".h5ad"))
    print("Saved data")
    
def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process the OOD (Out-of-Distribution) path.")
    
    # Add argument for "ood_path"
    parser.add_argument(
        '--ood_pathway', 
        type=str, 
        required=True, 
        help="The split to use"
    )
    
    parser.add_argument(
        '--ood_cell_type', 
        type=str, 
        required=True, 
        help="The split to use"
    )
    
    parser.add_argument(
        '--hvg', 
        type=int, 
        required=True, 
        help="The number of highly variable genes."
    )
    
    parser.add_argument(
        '--pca_dim', 
        type=int, 
        required=True, 
        help="PCA dimension."
    )
    
    parser.add_argument(
        '--ms', 
        type=float, 
        required=True, 
        help="MixScape score."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Pass the ood_path to the mock function
    split_and_save_adata(args)

if __name__ == "__main__":
    main()
