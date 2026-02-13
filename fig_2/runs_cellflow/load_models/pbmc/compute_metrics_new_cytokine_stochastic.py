import pandas as pd
import os
import scanpy as sc
import cellflow
import anndata as ad
import pickle
from cellflow.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca, reconstruct_pca
from cellflow.metrics import compute_r_squared, compute_e_distance
import numpy as np
import sys


def compute_metrics(adata_ref: ad.AnnData, adata_pred_all_samples: ad.AnnData, donor_deg_dict: dict, adata_ood_true: ad.AnnData, n_neighbors: int=1, cell_type_col: str = "cell_type_new", min_cells_for_dist_metrics: int = 50) -> dict:
    dict_to_log = {}
    
    compute_wknn(ref_adata=adata_ref, query_adata=adata_pred_all_samples, n_neighbors=n_neighbors, ref_rep_key="X_pca", query_rep_key="X_pca_for_ct_transfer")
    transfer_labels(query_adata=adata_pred_all_samples, ref_adata=adata_ref, label_key=cell_type_col)
    
    ood_e_distances = []
    decoded_ood_r_squareds = []
    mean_decoded_r_sq_per_cell_types = []
    mean_e_distance_per_cell_types = []
    mean_deg_r_sq_per_cell_types = []

    for i in range(10):

        adata_pred = adata_pred_all_samples[adata_pred_all_samples.obs["sample"]==i]

        ood_e_distance = compute_e_distance(adata_ood_true.obsm["X_pca"], adata_pred.obsm["X_pca"])
        decoded_ood_r_squared = compute_r_squared(adata_ood_true.X.toarray(), adata_pred.layers["X_recon"])

        ood_e_distances.append(ood_e_distance)
        decoded_ood_r_squareds.append(decoded_ood_r_squared)
        

        r_sq = {}
        e_distance = {}
        deg_r_sq = {}
        
        for ct_cyto in donor_deg_dict.keys(): 
            cell_type = ct_cyto.split("_")[1]
            adata_true_ct = adata_ood_true[(adata_ood_true.obs[f"{cell_type_col}"]==cell_type)]
            adata_pred_ct = adata_pred[adata_pred.obs[f"{cell_type_col}_transfer"]==cell_type]
            if adata_pred_ct.n_obs == 0:
                continue
            dist_true_decoded = adata_true_ct.X.toarray()
            dist_pred_decoded = adata_pred_ct.X
            dist_true = adata_true_ct.obsm["X_pca"]
            dist_pred = adata_pred_ct.obsm["X_pca"]
            r_sq[f"decoded_r_squared_{cell_type}"] = compute_r_squared(dist_true_decoded, dist_pred_decoded)
            e_distance[f"e_distance_{cell_type}"] = compute_e_distance(dist_true, dist_pred)
            
            deg_mask = [True if el in donor_deg_dict[ct_cyto] else False for el in adata_ood_true.var_names]
            deg_true_decoded = adata_true_ct[:,deg_mask].X.toarray()
            deg_pred_decoded = adata_pred_ct[:,deg_mask].X
            deg_r_sq[f"deg_decoded_r_squared_{cell_type}"] = compute_r_squared(deg_true_decoded, deg_pred_decoded)

        mean_decoded_r_sq_per_cell_types.append(np.mean(list(r_sq.values())))
        mean_e_distance_per_cell_types.append(np.mean(list(e_distance.values())))
        mean_deg_r_sq_per_cell_types.append(np.mean(list(deg_r_sq.values())))

    # performance metrics to return
    for i in range(10):
        dict_to_log[f"ood_e_distance_{i}"] = ood_e_distances[i]
        dict_to_log[f"decoded_ood_r_squared_{i}"] = decoded_ood_r_squareds[i]
    dict_to_log["mean_ood_e_distance"] = np.mean(ood_e_distances)
    dict_to_log["mean_decoded_ood_r_squared"] = np.mean(decoded_ood_r_squareds)

    dict_to_log["mean_decoded_r_sq_per_cell_type"] = np.mean(mean_decoded_r_sq_per_cell_types)
    dict_to_log["mean_e_distance_per_cell_type"] = np.mean(mean_e_distance_per_cell_types)
    dict_to_log["mean_deg_r_sq_per_cell_type"] = np.mean(mean_deg_r_sq_per_cell_types)

    # now the uncertainty metrics

    distr_dict = {}
    distr_dict_per_cell_type = {}
    gex_dict = {}
    gex_dict_per_cell_type = {}
    deg_gex_per_cell_type = {}
    for sample_idx in range(10):
        adata_pred = adata_pred_all_samples[adata_pred_all_samples.obs["sample"]==sample_idx]
        distr_dict[sample_idx] = adata_pred.obsm["X_pca_train"]
        gex_dict[sample_idx] = np.mean(adata_pred.X, axis=0)
        for ct_cyto in donor_deg_dict.keys():
            ct = ct_cyto.split("_")[1]
            adata_pred_ct = adata_pred[adata_pred.obs[f"{cell_type_col}_transfer"]==ct]
            if ct not in distr_dict_per_cell_type:
                distr_dict_per_cell_type[ct] = {}
            distr_dict_per_cell_type[ct][sample_idx] = adata_pred_ct.obsm["X_pca_train"]
            if ct not in gex_dict_per_cell_type:
                gex_dict_per_cell_type[ct] = {}
            gex_dict_per_cell_type[ct][sample_idx] = np.mean(adata_pred_ct.X, axis=0)
            if ct not in deg_gex_per_cell_type:
                deg_gex_per_cell_type[ct] = {}
            deg_mask = [True if el in donor_deg_dict[ct_cyto] else False for el in adata_ood_true.var_names]
            deg_gex_per_cell_type[ct][sample_idx] = np.mean(adata_pred_ct[:,deg_mask].X, axis=0)
            

    dict_to_log["frechet_variance"] = compute_frechet_var(distr_dict)
    dict_to_log["var_mean_gex"] = np.var(np.stack(list(gex_dict.values())), axis=0).mean()
    f_var_per_cell_types = []

    for ct, distr_ct in distr_dict_per_cell_type.items():
        f_var_ct = compute_frechet_var(distr_ct)
        f_var_per_cell_types.append(f_var_ct)
        dict_to_log[f"frechet_variance_{ct}"] = f_var_ct

    dict_to_log["mean_frechet_variance_per_cell_type"] = np.mean(f_var_per_cell_types)
    
    var_mean_gex_per_cell_types = []
    var_mean_deg_gex_per_cell_types = []
    for ct, gex_ct in gex_dict_per_cell_type.items():
        var_mean_gex_ct = np.var(np.stack(list(gex_ct.values())), axis=0).mean()
        var_mean_gex_per_cell_types.append(var_mean_gex_ct)
        dict_to_log[f"var_mean_gex_{ct}"] = var_mean_gex_ct
        deg_gex_ct = deg_gex_per_cell_type[ct]
        var_mean_deg_gex_ct = np.var(np.stack(list(deg_gex_ct.values())), axis=0).mean()
        var_mean_deg_gex_per_cell_types.append(var_mean_deg_gex_ct)
        dict_to_log[f"var_mean_deg_gex_{ct}"] = var_mean_deg_gex_ct

    dict_to_log["mean_var_mean_gex_per_cell_type"] = np.mean(var_mean_gex_per_cell_types)
    dict_to_log["mean_var_mean_deg_gex_per_cell_type"] = np.mean(var_mean_deg_gex_per_cell_types)

    
    return dict_to_log

def compute_frechet_var(x: dict[int, np.ndarray]):
    pairwise_e_dists = []
    denominator = 2 * len(x)**2
    for s1 in x.values():
        for s2 in x.values():
            pairwise_e_dists.append(compute_e_distance(s1, s2)**2)
    return np.sum(np.array(pairwise_e_dists))/denominator



if __name__ == "__main__":
    pred_file = sys.argv[1]
    complete_pred_file = os.path.join("/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_cytokine_stochastic_2", pred_file)
    adata_pred = sc.read_h5ad(complete_pred_file)
    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/idcs_to_keep.pkl", "rb") as pickle_file:
        idcs_to_keep = pickle.load(pickle_file)

    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/degs.pkl", "rb") as pickle_file:
            deg_genes = pickle.load(pickle_file)


    adata_full = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_with_pca.h5ad")
    adata_ref = adata_full[adata_full.obs_names.isin(idcs_to_keep)]
        

    out_dicts = {}
    dfs = []
    for condition in adata_pred.obs["condition"].unique():
        print(condition)
        adata_pred_one_cond = adata_pred[adata_pred.obs["condition"]==condition]
        donor=condition.split("_")[0]
        cytokine = condition.split("_")[1]
        adata_ood_true_cond = adata_full[(adata_full.obs["donor"] == donor) & (adata_full.obs["cytokine"]==cytokine)]
        donor_deg_dict = {k: v for k, v in deg_genes.items() if (k.startswith(donor) and k.endswith(f"_{cytokine}"))}
        out = compute_metrics(adata_ref=adata_ref, adata_pred_all_samples=adata_pred_one_cond, donor_deg_dict=donor_deg_dict, adata_ood_true=adata_ood_true_cond)
        out_dicts[condition] = out
        pd.DataFrame(out, index=[condition])
        #pd.DataFrame.from_dict(out, columns=[condition], orient="index").to_csv(os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_{condition}.csv"))
            
            
        dfs.append(pd.DataFrame(out, index=[condition]))

    df = pd.concat(dfs)
    filename = pred_file.strip("_all_preds.h5ad")+".csv"
    df.to_csv(f"/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_cytokine_stochastic_metrics/{filename}")

        