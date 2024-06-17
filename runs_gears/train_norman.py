import os, sys
from gears import PertData, GEARS
import scanpy as sc
import numpy as np
import pickle
import jax
import jax.tree_util as jtu
from scipy.sparse import csr_matrix
#from metrics import compute_mean_metrics, compute_metrics_fast, compute_metrics
from functools import partial
from datetime import datetime
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from typing import Dict
import traceback
import torch

from ott.geometry import costs, pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.metrics.pairwise import rbf_kernel

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

def setup_logger(cfg):
    """Initialize and return a Weights & Biases logger."""
    wandb.login()
    return wandb.init(
        project=cfg.dataset.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        #dir="/home/icb/lea.zimmermann/projects/pertot/ot_pert_reproducibility/runs_gears/bash_scripts",
        settings=wandb.Settings(start_method="thread"),
    )

def compute_r_squared(x: np.ndarray, y: np.ndarray) -> float:
    return r2_score(np.mean(x, axis=0), np.mean(y, axis=0))


def compute_sinkhorn_div(x: np.ndarray, y: np.ndarray, epsilon: float) -> float:
    return float(
        sinkhorn_divergence(
            pointcloud.PointCloud,
            x=x,
            y=y,
            cost_fn=costs.SqEuclidean(),
            epsilon=epsilon,
            scale_cost=1.0,
        ).divergence
    )


def compute_e_distance(x: np.ndarray, y: np.ndarray) -> float:
    sigma_X = pairwise_distances(x, x, metric="sqeuclidean").mean()
    sigma_Y = pairwise_distances(y, y, metric="sqeuclidean").mean()
    delta = pairwise_distances(x, y, metric="sqeuclidean").mean()
    return 2 * delta - sigma_X - sigma_Y


def compute_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    metrics = {}
    metrics["r_squared"] = compute_r_squared(x, y)
    metrics["sinkhorn_div_1"] = compute_sinkhorn_div(x, y, epsilon=1.0)
    metrics["sinkhorn_div_10"] = compute_sinkhorn_div(x, y, epsilon=10.0)
    metrics["sinkhorn_div_100"] = compute_sinkhorn_div(x, y, epsilon=100.0)
    metrics["e_distance"] = compute_e_distance(x, y)
    metrics["mmd"] = compute_scalar_mmd(x, y)
    return metrics


def compute_mean_metrics(metrics: Dict[str, Dict[str, float]], prefix: str = ""):
    metric_names = list(list(metrics.values())[0].keys())
    metric_dict = {prefix + met_name: [] for met_name in metric_names}
    for met in metric_names:
        stat = 0.0
        for vals in metrics.values():
            stat += vals[met]
        metric_dict[prefix + met] = stat / len(metrics)
    return metric_dict


def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_scalar_mmd(target, transport, gammas=None):  # from CellOT repo
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))


def compute_metrics_fast(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    metrics = {}
    metrics["r_squared"] = compute_r_squared(x, y)
    metrics["e_distance"] = compute_e_distance(x, y)
    metrics["mmd_distance"] = compute_scalar_mmd(x, y)
    return metrics

def get_mask(x, y, var_names):
    return x[:, [gene in y for gene in var_names]]

def transform_condition(input_str):
    # Check if the input string contains a '+'
    if '+' in input_str:
        # Split the input string by '+'
        parts = input_str.split('+')
        # Construct the output string for the case with two strings
        output_str = f'K562_{parts[0]}+{parts[1]}_1+1'
    else:
        # Construct the output string for the case with one string
        output_str = f'K562_{input_str}+ctrl_1+1'
    
    return output_str

def load_data(adata, cfg, *, skip_cond, deg):
    """Loads data and preprocesses it based on configuration."""
    data_source = {}
    data_target = {}
    data_source_decoded = {}
    data_target_decoded = {}
    data_conditions = {}
    data_conditions_deg = {}
    for cond in adata.obs["condition_name"].cat.categories:
        if cond!= 'K562_ctrl+1' and cond not in skip_cond:
            src_str_unique = list(adata[adata.obs["condition_name"] == cond].obs["cell_type"].unique())
            assert len(src_str_unique) == 1
            src_str = 'K562_ctrl+1'
            source = adata[adata.obs["condition_name"] == src_str].obsm[cfg.dataset.obsm_key_data]
            source_decoded = adata[adata.obs["condition_name"] == src_str].X.A
            target = adata[adata.obs["condition_name"] == cond].obsm[cfg.dataset.obsm_key_data]
            target_decoded = adata[adata.obs["condition_name"] == cond].X.A
            data_source[cond] = source
            data_target[cond] = target
            data_source_decoded[cond] = source_decoded
            data_target_decoded[cond] = target_decoded
            data_conditions[cond] = cond
            data_conditions_deg[cond.split('_')[1].replace('+ctrl', '')] = cond.split('_')[1].replace('+ctrl', '')
       
    print("deg: ", deg)
    print("data_conditions: ", data_conditions)     
    print("data_conditions_deg: ", data_conditions_deg)

    # GEARS outputs predictions in which key is 'cellline_gene1+gene2_1+1', so we need deg to follow that format
    deg_dict = {transform_condition(k): v for k, v in deg.items() if k in data_conditions_deg.keys()}

    print("deg_dict: ", deg_dict)

    return {
        "source": data_source,
        "target": data_target,
        "source_decoded": data_source_decoded,
        "target_decoded": data_target_decoded,
        "conditions": data_conditions,
        "deg_dict": deg_dict,
    }

def eval_step(cfg, model, data, log_metrics, comp_metrics_fn, mask_fn, PCs, train_mean):
    for split, dat in data.items():
        print("SPLIT: ", split)
        if split == "test":
            n_samples = cfg.training.n_test_samples
        if split == "ood":
            n_samples = cfg.training.n_ood_samples

        if split == "ood":
            print(n_samples)
        if n_samples != 0:
            if n_samples > 0:
                idcs = np.random.choice(list(list(dat.values())[0]), n_samples)
                dat_conditions = {k: v for k, v in dat["conditions"].items() if k in idcs}
                dat_deg_dict = {k: v for k, v in dat["deg_dict"].items() if k in idcs}
                dat_target = {k: v for k, v in dat["target"].items() if k in idcs}
                dat_target_decoded = {k: v for k, v in dat["target_decoded"].items() if k in idcs}
            else:
                dat_conditions = dat["conditions"]
                dat_deg_dict = dat["deg_dict"]
                dat_target = dat["target"]
                dat_target_decoded = dat["target_decoded"]
            
            print("data deg dict:", dat_deg_dict)
            
            predictions = {}
            predictions_pca = {}
            for k, v in dat_target_decoded.items():
                cond = dat_conditions[k]
                # cond is originally 'K562_gene1+gene2_1+1'
                genes = []
                gene1 = cond.split('+')[0].split('_')[1]
                gene2 = cond.split('+')[1].split('_')[0]
                #gene = cond.split('+')[0]
                #gene = cond.split('_')[1] # added this line cause cond is originally 'K562_gene1+gene2'
                if gene1 != 'ctrl':
                    genes.append(gene1)
                if gene2 != 'ctrl':
                    genes.append(gene2)
                samples = np.zeros((v.shape[0], v.shape[1]))
                for i in range(samples.shape[0]):
                    samples[i] = model.predict([genes])['_'.join(genes)]#[genes]
                    #samples[i] = samples[i][genes]
                    model.saved_pred = {}
                predictions[k] = samples
                samples_centered = csr_matrix(samples - train_mean)
                predictions_pca[k] = np.matmul(samples_centered.A, PCs)
                
            print("Predictions: ", predictions)

            metrics = jtu.tree_map(comp_metrics_fn, dat_target, predictions_pca)
            mean_metrics = compute_mean_metrics(metrics, prefix=f"{split}_")
            log_metrics.update(mean_metrics)

            metrics_decoded = jtu.tree_map(comp_metrics_fn, dat_target_decoded, predictions)
            mean_metrics_decoded = compute_mean_metrics(metrics_decoded, prefix=f"decoded_{split}_")
            log_metrics.update(mean_metrics_decoded)
            
            print(mean_metrics_decoded)

            print("predictions: ", predictions.keys())
            print("dat_deg_dict: ", dat_deg_dict.keys())
            prediction_decoded_deg = jtu.tree_map(mask_fn, predictions, dat_deg_dict)
            target_decoded_deg = jax.tree_util.tree_map(mask_fn, dat_target_decoded, dat_deg_dict)
            metrics_deg = jtu.tree_map(comp_metrics_fn, target_decoded_deg, prediction_decoded_deg)
            mean_metrics_deg = compute_mean_metrics(metrics_deg, prefix=f"deg_{split}_")
            log_metrics.update(mean_metrics_deg)
    
    wandb.log(log_metrics)
            
@hydra.main(config_path="conf", config_name="train_norman")
def run(cfg: DictConfig):
    print('run started')
    logger = setup_logger(cfg)
    print('logger setup')

    num_gpus = torch.cuda.device_count()
    print("gpus available: ", num_gpus)

    print('load data')
    
    adata = sc.read_h5ad(cfg.dataset.gears_train)
    adata.obs["cell_type"] = adata.obs["cell_line"]
    
    pert_data = PertData(cfg.dataset.base_path)
    # this should be done just once
    pert_data.new_data_process(dataset_name = cfg.dataset.name, adata = adata)
    pert_data.load(data_path = f"{cfg.dataset.base_path}/{cfg.dataset.name}")
    #pert_data.load(data_path=cfg.dataset.train_data)

    pert_data.prepare_split(split = 'custom', split_dict_path=cfg.dataset.custom_split, seed = cfg.training.seed) # get data split with seed
    pert_data.get_dataloader(batch_size = cfg.training.batch_size, test_batch_size = cfg.training.batch_size) # prepare data loader 
    
    print('model setup')
    gears_model = GEARS(pert_data, device='cuda',
                        weight_bias_track = cfg.logger.track_gears, )
                        #wandb_logger=logger)
                            
    gears_model.model_initialize(hidden_size = cfg.model.hidden_size,
                                num_go_gnn_layers = cfg.model.num_go_gnn_layers,
                                num_gene_gnn_layers = cfg.model.num_gene_gnn_layers,
                                decoder_hidden_size = cfg.model.decoder_hidden_size,
                                num_similar_genes_go_graph = cfg.model.num_similar_genes_go_graph,
                                num_similar_genes_co_express_graph = cfg.model.num_similar_genes_co_express_graph,                    
                                coexpress_threshold = cfg.model.coexpress_threshold,
                                uncertainty = cfg.model.uncertainty,
                                uncertainty_reg = cfg.model.uncertainty_reg,
                                direction_lambda = cfg.model.direction_lambda,
                                G_go = None if cfg.model.G_go == 'None' else cfg.model.G_go,
                                G_go_weight = None if cfg.model.G_go_weight == 'None' else cfg.model.G_go_weight,
                                G_coexpress = None if cfg.model.G_coexpress == 'None' else cfg.model.G_coexpress,
                                G_coexpress_weight = None if cfg.model.G_coexpress_weight == 'None' else cfg.model.G_coexpress_weight,
                                no_perturb = cfg.model.no_perturb)
    
    num_batches = len(pert_data.dataloader['train_loader'])
    num_epochs = cfg.training.num_iterations // num_batches
    valid_freq_in_epochs = cfg.training.valid_freq // num_batches
    training_blocks = num_epochs // valid_freq_in_epochs
    # optimizer = optim.Adam(gears_model.model.parameters(), lr=cfg.training.learning_rate, weight_decay = cfg.training.weight_decay)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    test_data = None

    print('start training')
    for i in range(training_blocks):
        gears_model.train(epochs = valid_freq_in_epochs,
                        lr = cfg.training.learning_rate,
                        weight_decay = cfg.training.weight_decay,
                        # optimizer=optimizer,
                        # scheduler=scheduler
                        )
        
        if test_data == None:
            #data_path = cfg.dataset.test_data
            adata_train = sc.read_h5ad(cfg.dataset.adata_train_path)
            adata_train.obs["cell_type"] = adata_train.obs["cell_line"]
            #adata_train.obs.condition_name = adata_train.obs['condition_name'].str.replace('K562_ctrl+ctrl_1+1', 'K562_ctrl_1')
            adata_train = adata_train[~adata_train.obs.condition.isin(cfg.dataset.remove_perturbations)]
            
            train_mean = adata_train.varm["X_train_mean"].T
            PCs = adata_train.varm["PCs"]
            mask_fn = partial(get_mask, var_names=adata_train.var_names)
            deg = adata_train.uns["rank_genes_groups_cov_all"]
            #deg = {k.split('_')[1]+'+ctrl': v  for k, v in adata_train.uns["rank_genes_groups_cov_all"].items()}

            adata_test = sc.read_h5ad(cfg.dataset.adata_test_path)
            #adata_test.obs['condition'] = adata_test.obs['condition'].apply(lambda x: x.split('_')[-1] + '+ctrl')
            adata_test.obs["cell_type"] = adata_test.obs["cell_line"]
            #adata_test.obs.condition_name = adata_test.obs['condition_name'].str.replace('K562_ctrl+ctrl_1+1', 'K562_ctrl_1')
            adata_test = adata_test[~adata_test.obs.condition.isin(cfg.dataset.remove_perturbations)]
            
            adata_ood = sc.read_h5ad(cfg.dataset.adata_ood_path)
            #adata_ood.obs['condition'] = adata_ood.obs['condition'].apply(lambda x: x.split('_')[-1] + '+ctrl')
            adata_ood.obs["cell_type"] = adata_ood.obs["cell_line"]
            #adata_ood.obs.condition_name = adata_ood.obs['condition_name'].str.replace('K562_ctrl+ctrl_1+1', 'K562_ctrl_1')
            adata_ood = adata_ood[~adata_ood.obs.condition.isin(cfg.dataset.remove_perturbations)]
            deg2 = adata_ood.uns["rank_genes_groups_cov_all"]
            
            deg = deg | deg2

            print("deg: ", deg)
            testset = load_data(adata_test, cfg, deg=deg, skip_cond=[])
            oodset = load_data(adata_ood, cfg, deg=deg, skip_cond=[])
            test_data = {
                'test': testset,
                'ood': oodset
            }

            comp_metrics_fn = compute_metrics_fast if cfg.training.fast_metrics == True else compute_metrics
        
        log_metrics = {}
        eval_step(cfg, gears_model, test_data, log_metrics, comp_metrics_fn, mask_fn, PCs, train_mean)

    print('training finished')
    if cfg.training.save_model:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")
        gears_model.save_model(os.path.join(cfg.base_path.save_model_path, timestamp))

    print('at the end')
    return 1.0

if __name__ == "__main__":
    try:
        print('script started')
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)