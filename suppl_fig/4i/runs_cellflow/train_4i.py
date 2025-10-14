import scanpy as sc
import pandas as pd 
import cellflow
import rapids_singlecell as rsc
import numpy as np
import pandas as pd
import seaborn as sns
import jax
import functools
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import flax.linen as nn
import optax
import cellflow
from cellflow.model import CellFlow
import cellflow.preprocessing as cfpp
from cellflow.utils import match_linear
from cellflow.plotting import plot_condition_embedding
from cellflow.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca, reconstruct_pca
from cellflow.metrics import compute_r_squared, compute_e_distance, compute_scalar_mmd
import anndata
from omegaconf import OmegaConf
import hydra
import os
import anndata as ad
import sys

def compute_sinkhorn_div(x: np.ndarray, y: np.ndarray, epsilon: float) -> float:
    from ott.tools.sinkhorn_divergence import sinkhorn_divergence
    from ott.geometry import costs, pointcloud
    return float(sinkhorn_divergence(
            pointcloud.PointCloud,
            x=x,
            y=y,
            cost_fn=costs.SqEuclidean(),
            epsilon=epsilon,
            scale_cost=1.0,
        )[0])
    

@hydra.main(config_path="conf", config_name="train")
def run(config):
    config_dict  = OmegaConf.to_container(config, resolve=True)
    adata = sc.read_h5ad("/lustre/groups/ml01/projects/CellOT_comparison/adata_processed.h5ad")
    drug = config_dict["dataset"]["drug"]
    adata = adata[adata.obs["drug"].isin(("control", drug))]
    dummy_rep = {drug: np.array([0.0])}
    adata.uns["dummy_drug_rep"] = dummy_rep
    adata.obs["is_control"] = adata.obs.apply(lambda x: True if x["drug"]=="control" else False, axis=1)
    adata.obsm["X"] = adata.X.copy()
    adata_train = adata[adata.obs["split"]=="train"]
    adata_test = adata[adata.obs["split"]=="test"]

    cf = cellflow.model.CellFlow(adata_train, solver="otfm")
    cf.prepare_data(
        sample_rep="X",
        control_key="is_control",
        perturbation_covariates={"drugs": ("drug",)},
        perturbation_covariate_reps={"drugs": "dummy_drug_rep"},
    )
    match_fn = functools.partial(match_linear, epsilon=config["model"]["epsilon"], tau_a=config["model"]["tau_a"], tau_b=config["model"]["tau_b"])

    cf.prepare_model(
        encode_conditions=True,
        pooling="mean",
        layers_before_pool= {"drugs": {"layer_type": "mlp", "dims": [2], "dropout_rate": 0.0}},
        layers_after_pool={"layer_type": "mlp", "dims": [2], "dropout_rate": 0.0},
        condition_embedding_dim=2,
        cond_output_dropout=0.0,
        time_freqs=1024,
        time_encoder_dims=[1024, 1024, 1024],
        time_encoder_dropout=0.0,
        hidden_dims=[1024, 1024, 1024],
        hidden_dropout=0.2,
        decoder_dims=[2048, 2048, 2048],
        decoder_dropout=0.2,
        flow={"constant_noise": config["model"]["flow_noise"]},
        match_fn=match_fn,
        optimizer=optax.MultiSteps(optax.adam(config["model"]["lr"]), 1),
        layer_norm_before_concatenation=False,
        linear_projection_before_concatenation=False,
    )


    cf.prepare_validation_data(
        adata_test,
        name="test",
        n_conditions_on_log_iteration=None,
        n_conditions_on_train_end=None,
    )

    metrics_callback = cellflow.training.Metrics(metrics=["r_squared", "mmd", "e_distance"])
    wandb_callback = cellflow.training.WandbLogger(project="4i_data", out_dir="/home/icb/dominik.klein/tmp", config=config_dict)
    
    callbacks = [metrics_callback, wandb_callback]
    cf.train(num_iterations=50_000,  batch_size=1024,  callbacks=callbacks,  valid_freq=500_000)

    adata_ctrl_for_prediction = adata_test[adata_test.obs["is_control"].to_numpy()]
    covariate_data = adata_test.obs[adata_test.obs["drug"]==drug].iloc[:1,:]
    preds = cf.predict(adata=adata_ctrl_for_prediction, sample_rep="X_pca", covariate_data=covariate_data)
    prediction = next(iter(preds.values()))
    true = adata_test[adata_test.obs["drug"]==drug].X
    r_squared = compute_r_squared(true, prediction)
    e_distance = compute_e_distance(true, prediction)
    mmd = compute_scalar_mmd(true, prediction)
    sinkhorn_div_1 = compute_sinkhorn_div(true, prediction, 1.0)
    sinkhorn_div_10 = compute_sinkhorn_div(true, prediction, 10.0)
    sinkhorn_div_100 = compute_sinkhorn_div(true, prediction, 100.0)
    metrics = {
        "r_squared": r_squared,
        "e_distance": e_distance,
        "mmd": mmd,
        "sinkhorn_div_1": sinkhorn_div_1,
        "sinkhorn_div_10": sinkhorn_div_10,
        "sinkhorn_div_100": sinkhorn_div_100
    }
    metrics_df = pd.DataFrame(metrics, columns=["values"])
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=[drug]).T
    metrics_df["drug"] = metrics_df.index
    metrics_df["method"] = "CellFlow"
    adata = ad.AnnData(
        X=np.asarray(prediction),
        obs=[drug] * len(prediction),
    )
    output_dir = "/lustre/groups/ml01/projects/CellOT_comparison/4i"
    metrics_df.to_csv(os.path.join(output_dir, f"metrics_{drug}.csv"))
    adata.write_h5ad(os.path.join(output_dir, f"predictions_{drug}.h5ad"))






    

    
    

    
if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
