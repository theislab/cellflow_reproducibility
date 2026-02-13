from memory_profiler import memory_usage
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)

import numpy as np
import pandas as pd
import seaborn as sns
import jax
import functools
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import rapids_singlecell as rsc
import flax.linen as nn
import optax
import cellflow
from cellflow.model import CellFlow
import cellflow.preprocessing as cfpp
from cellflow.utils import match_linear
from cellflow.plotting import plot_condition_embedding
from cellflow.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca, reconstruct_pca
from cellflow.metrics import compute_r_squared, compute_e_distance

adata = cellflow.datasets.pbmc_cytokines()
adata.obs["condition"] = adata.obs.apply(lambda x: x["donor"] + "_" + x["cytokine"], axis=1)
adata.obs["is_control"] = adata.obs.apply(lambda x: True if x["cytokine"]=="PBS" else False, axis=1)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

adata_train = adata[(adata.obs["cytokine"]!="IL-15") | (adata.obs["donor"]=="Donor8")].copy()
adata_test = adata[((adata.obs["cytokine"]=="IL-15") & (adata.obs["donor"]!="Donor8")) | (adata.obs["cytokine"]=="PBS")].copy()
adata_train.n_obs, adata_test.n_obs

cfpp.centered_pca(adata_train, n_comps=100, method="rapids", keep_centered_data=False)
cfpp.project_pca(query_adata=adata_test, ref_adata=adata_train)




def run_cellflow():
    cf = CellFlow(adata_train, solver="otfm")
    cf.prepare_data(
        sample_rep = "X_pca",
        control_key = "is_control",
        perturbation_covariates = {"cytokine_treatment": ("cytokine",)},
        perturbation_covariate_reps = {"cytokine_treatment": "esm2_embeddings"},
        sample_covariates = ["donor"],
        sample_covariate_reps = {"donor": "donor_embeddings"},
        split_covariates = ["donor"],
        max_combination_length = 1,
        null_value = 0.0,
    )
    layers_before_pool = {
        "cytokine_treatment": {"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.5},
        "donor": {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0},
    }

    layers_after_pool = {
        "layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.0,
    }

    match_fn = functools.partial(match_linear, epsilon=0.5, tau_a=1.0, tau_b=1.0)
    cf.prepare_model(
        condition_mode="deterministic",
        regularization=0.0,
        pooling="attention_token",
        pooling_kwargs={},
        layers_before_pool=layers_before_pool,
        layers_after_pool=layers_after_pool,
        condition_embedding_dim=256,
        cond_output_dropout=0.9,
        condition_encoder_kwargs={},
        pool_sample_covariates=True,
        time_freqs=1024,
        time_encoder_dims=[1024, 1024, 1024],
        time_encoder_dropout=0.0,
        hidden_dims=[2048, 2048, 2048],
        hidden_dropout=0.0,
        conditioning="concatenation"
        decoder_dims=[4096, 4096, 4096],
        vf_act_fn=nn.silu,
        vf_kwargs=None,
        probability_path={"constant_noise": 0.5},
        match_fn=match_fn,
        optimizer=optax.MultiSteps(optax.adam(5e-5), 20),
        solver_kwargs={},
        layer_norm_before_concatenation=False,
        linear_projection_before_concatenation=False,
    )

    cf.train(
        num_iterations=10,
        batch_size=1024,
        callbacks=[],
        valid_freq=20_000,
    )


mem = memory_usage(run_cellflow, max_usage=True)
print(f"Peak memory: {mem:.2f} MiB")