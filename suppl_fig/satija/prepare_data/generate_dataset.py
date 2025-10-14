import os
import scanpy as sc
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import get_DE_genes
import anndata
from scipy.sparse import csr_matrix
from cfp import preprocessing as cfpp

hvg = 500
pca_dim = 100
ms = 0.5
pathway = 'IFNG_IFNB_TNFA_TGFB_INS'
ood_pathway = 'IFNB'
ood_cell_type = 'MCF7'
output_dir = "/lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/ood_cell_type/" + pathway + '_ct-' + ood_cell_type + '_hvg-' + str(hvg) + '_pca-' + str(pca_dim) + '_counts' + '_ms_' + str(ms)

genes_from_paper = [
    "AHNAK", "RNF213", "APOL6", "ASTN2", "B2M", "CFH", "CXCL9", "DENND4A", 
    "DOCK9", "EFNA5", "ERAP2", "FAT1", "GBP1", "GBP4", "HAPLN3", "HSPG2", 
    "IDO1", "IFI6", "IRF1", "LAP3", "LI", "LINC02328", "MAGI1", "MUC4", 
    "NLRC5", "NUB1", "PARP14", "PARP9", "RARRES1", "RNF213", "ROR1", "SCN9A", 
    "SERPING1", "ST5", "STAT1", "TAP1", "TAP2", "THBS1", "THSD4", "TPM1", "VCL", 
    "WARS", "XRN1"
]

os.makedirs(output_dir, exist_ok=True)

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

adata = anndata.concat(datasets, join='outer')
print('Datasets concatenated')

adata.obs_names_make_unique()

adata.obs['condition'] = adata.obs.apply(lambda x: "_".join([x.cell_type, x.pathway, x.gene]), axis=1)
adata.obs['background'] = adata.obs.apply(lambda x: "_".join([x.cell_type, x.pathway]), axis=1)
condition_counts = adata.obs['condition'].value_counts()
filtered_conditions = condition_counts[condition_counts >= 100]
adata = adata[adata.obs['condition'].isin(filtered_conditions.index)]

adata.layers["counts"] = adata.layers["counts"].astype(np.float32)
adata.X = csr_matrix(adata.layers["counts"])
del adata.layers['counts']
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

columns_to_drop = ['orig.ident', 'nCount_RNA', 'nFeature_RNA', 'sample', 'percent.mito', 'sample_ID', 'Batch_info', 'bc1_well', 'bc2_well', 'bc3_well', 'guide', 'mixscale_score', 'RNA_snn_res.0.9', 'seurat_clusters']
adata.obs.drop(columns=columns_to_drop, inplace=True)
print('Datasets prepared, running hvg analysis')

highly_var_genes = {}
for bg in tqdm(adata.obs['background'].unique()):
    temp = adata[adata.obs['background'] == bg, :]
    sc.pp.highly_variable_genes(temp, inplace=True, n_top_genes=hvg)
    temp = adata[:,temp.var["highly_variable"]==True]
    highly_var_genes[bg] = set(temp.var.index)

combined_set = set()
for key in highly_var_genes:
    combined_set.update(highly_var_genes[key])
combined_set = combined_set.union(set(genes_from_paper))
adata = adata[:, adata.var.index.isin(combined_set)]

print('HVG analysis done')

adata = get_DE_genes(adata, by='condition', covariate='background')
print('DE genes calculated')

# controls = {}
# for bg in adata.obs["background"].unique():
#     controls[bg] = adata[adata.obs["condition"]==bg+'_NT'].X.toarray()

for col in adata.obs.select_dtypes(include=["category"]):
    adata.obs[col].cat.remove_unused_categories()

# adata = get_DE_genes(adata, by='condition', covariate='background')

ood_condition = ood_cell_type + '_' + ood_pathway
filtered_conditions = adata.obs['condition'].unique() # unnecessary

perturbations = list(adata.obs[adata.obs['gene'] != 'NT']["condition"].unique())
ood_conditions = [c for c in perturbations if c.startswith(ood_condition) and c in filtered_conditions]

remaining_conditions = list(set(filtered_conditions) - set(ood_conditions))
remaining_unique = list(set([value for entry in remaining_conditions for value in entry.split('_')]))

adata.obs["ood"] = adata.obs.apply(lambda x: x["condition"] if x["condition"] in ood_conditions else False, axis=1)
adata.obs["is_ood"] = adata.obs.apply(lambda x: x["condition"] in ood_conditions, axis=1)
adata.obs.drop(columns='ood', inplace=True)
adata_train = adata[~adata.obs["is_ood"]]
adata_ood = adata[adata.obs["is_ood"]]
print(adata_ood.obs['control'].value_counts())

rng = np.random.default_rng(0)
split_dfs = []
adata_train.obs["split"] = "not_included"
for c in adata_train.obs["condition"].unique():
    n_cells = adata_train[(adata_train.obs["condition"]==c)].n_obs
    if c.endswith('_NT'):
        idx_test = rng.choice(np.arange(n_cells), 500, replace=False)
        remaining_idcs = set(np.arange(n_cells)) - set(idx_test)
        adata_train.obs.loc[adata_train.obs['condition'] == c, 'split'] = ["test" if idx in idx_test else "train" for idx in range(n_cells)]
    elif n_cells>300:
        idx_test = rng.choice(np.arange(n_cells), 100, replace=False)
        adata_train.obs.loc[adata_train.obs['condition'] == c, 'split'] = ["test" if idx in idx_test else "train" for idx in range(n_cells)]

adata_train_final = adata_train[adata_train.obs["split"]=="train"]
adata_test_final = adata_train[adata_train.obs["split"]=="test"]
adata_ood_final = anndata.concat((adata_ood, adata_test_final[adata_test_final.obs["condition"].str.endswith('_NT')]))
adata_ood_final.uns = adata_ood.uns
print(adata_ood_final.obs['control'].value_counts())

del adata

adata_train_final = adata_train_final[~((adata_train_final.obs['pathway'] == ood_pathway) & (adata_train_final.obs['cell_type'] == ood_cell_type)), :]
adata_test_final = adata_test_final[~((adata_test_final.obs['pathway'] == ood_pathway) & (adata_test_final.obs['cell_type'] == ood_cell_type)), :]
adata_ood_final = adata_ood_final[((adata_ood_final.obs['pathway'] == ood_pathway) & (adata_ood_final.obs['cell_type'] == ood_cell_type)), :]

adata_train_final.write(os.path.join(output_dir, "adata_train_" + ood_pathway + "_" + ood_cell_type + "_wo_pca.h5ad"))

cfpp.centered_pca(adata_train_final, n_comps=pca_dim)

adata_train_final.write(os.path.join(output_dir, "adata_train_" + ood_pathway + "_" + ood_cell_type + "_pca.h5ad"))

adata_train_final.layers["X_log1p"] = adata_train_final.X.copy()
adata_train_final_mean = adata_train_final.varm["X_mean"].flatten()

adata_ood_final.varm["X_mean"] = adata_train_final.varm["X_mean"]
adata_test_final.varm["X_mean"] = adata_train_final.varm["X_mean"]
adata_test_final.layers["centered_X"] = csr_matrix(adata_test_final.X.toarray() - adata_train_final_mean)
adata_ood_final.layers["centered_X"] = csr_matrix(adata_ood_final.X.toarray() - adata_train_final_mean)
adata_test_final.obsm["X_pca"] = np.matmul(adata_test_final.layers["centered_X"].toarray(), adata_train_final.varm["PCs"])
adata_ood_final.obsm["X_pca"] = np.matmul(adata_ood_final.layers["centered_X"].toarray(), adata_train_final.varm["PCs"])

adata_train_final.obs['control'] = adata_train_final.obs.apply(lambda x: x['gene'] == 'NT', axis=1)
adata_test_final.obs['control'] = adata_test_final.obs.apply(lambda x: x['gene'] == 'NT', axis=1)
adata_ood_final.obs['control'] = adata_ood_final.obs.apply(lambda x: x['gene'] == 'NT', axis=1)


path_to_embeddings = os.path.join('/lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/embeddings/perturb_emb/satijas_v2', 'gene_embeddings.pkl')
ko_embeddings = pickle.load(open(path_to_embeddings, 'rb'))
ko_embeddings = pd.DataFrame(ko_embeddings).T
ko_embeddings = ko_embeddings.astype(np.float32)
cell_embeddings = pd.read_csv('/lustre/groups/ml01/workspace/ot_perturbation/data/satija/embeddings/cell_line_embedding_full_ccle_300_normalized.csv', index_col=0)
cell_embeddings = cell_embeddings.astype(np.float32)
cell_embeddings_dict = dict(zip(cell_embeddings.index, cell_embeddings.values))
cell_embeddings_dict = {k: v for k, v in cell_embeddings_dict.items() if k in adata_train_final.obs['cell_type'].unique()}
gene_embeddings_dict = dict(zip(ko_embeddings.index, ko_embeddings.values))
gene_embeddings_dict['NT'] = np.zeros(gene_embeddings_dict['IFNG'].shape)
pathway_embeddings = {k: v for k, v in gene_embeddings_dict.items() if k in adata_train_final.obs['pathway'].unique()}

adata_train_final.uns['gene_emb'] = gene_embeddings_dict
adata_train_final.uns['cell_type_emb'] = cell_embeddings_dict
adata_train_final.uns['pathway_emb'] = pathway_embeddings

adata_test_final.uns['gene_emb'] = gene_embeddings_dict
adata_test_final.uns['cell_type_emb'] = cell_embeddings_dict
adata_test_final.uns['pathway_emb'] = pathway_embeddings

adata_ood_final.uns['gene_emb'] = gene_embeddings_dict
adata_ood_final.uns['cell_type_emb'] = cell_embeddings_dict
adata_ood_final.uns['pathway_emb'] = pathway_embeddings

adata_train_final = adata_train_final[adata_train_final.obs['cell_type'].isin(cell_embeddings_dict.keys()), :]
adata_train_final = adata_train_final[adata_train_final.obs['pathway'].isin(pathway_embeddings.keys()), :]
adata_train_final = adata_train_final[(adata_train_final.obs['gene'].isin(gene_embeddings_dict.keys()) | (adata_train_final.obs['gene'] == 'NT')), :]

adata_test_final = adata_test_final[adata_test_final.obs['cell_type'].isin(cell_embeddings_dict.keys()), :]
adata_test_final = adata_test_final[adata_test_final.obs['pathway'].isin(pathway_embeddings.keys()), :]
adata_test_final = adata_test_final[(adata_test_final.obs['gene'].isin(gene_embeddings_dict.keys()) | (adata_test_final.obs['gene'] == 'NT')), :]

adata_ood_final = adata_ood_final[adata_ood_final.obs['cell_type'].isin(cell_embeddings_dict.keys()), :]
adata_ood_final = adata_ood_final[adata_ood_final.obs['pathway'].isin(pathway_embeddings.keys()), :]
adata_ood_final = adata_ood_final[(adata_ood_final.obs['gene'].isin(gene_embeddings_dict.keys()) | (adata_ood_final.obs['gene'] == 'NT')), :]
print(adata_ood_final.obs['control'].value_counts())

adata_train_final = adata_train_final[~((adata_train_final.obs['pathway'] == ood_pathway) & (adata_train_final.obs['cell_type'] == ood_cell_type)), :]
adata_test_final = adata_test_final[~((adata_test_final.obs['pathway'] == ood_pathway) & (adata_test_final.obs['cell_type'] == ood_cell_type)), :]

print('Embeddings added to datasets')

adata_train_final.write(os.path.join(output_dir, "adata_train_" + ood_pathway + "_" + ood_cell_type + ".h5ad"))
adata_ood_final.write(os.path.join(output_dir, "adata_ood_" + ood_pathway + "_" + ood_cell_type + ".h5ad"))
adata_test_final.write(os.path.join(output_dir, "adata_test_" + ood_pathway + "_" + ood_cell_type + ".h5ad"))
print('Datasets saved')
