import warnings

warnings.filterwarnings("ignore")

import os
import sys

sys.path.append("/home/fleckj/projects/atlas/repo/atlas_utils")

import numpy as np
import pandas as pd

import scvi
from scvi.model.utils import mde
import matplotlib.pyplot as plt

import scanpy as sc
import cloudpickle

from hnoca.map import AtlasMapper
from hnoca.map.utils import prepare_features


# PATHS
REF_PATH = "/home/fleckj/scratch/data/public_datasets/primary_brain/BraunLinnarsson2022/braun_2022_fetal_brain_v3cleaned_common_hv2k_wknn.h5ad"
QUERY_PATH = "/projects/site/pred/organoid-atlas/data/local/perturbation/MidbrainPatterningNadya/midbrain_patterning_v1.h5ad"
MODEL_DIR = "/home/fleckj/projects/atlas/data/results/atlasv3_scanvi_scarches/v7/subregion_class/query_scarches_nolabs"

RESULTS_DIR = "/home/fleckj/projects/cellflow/results/midbrain_braun_mapping/"
PLOT_DIR = "/home/fleckj/projects/cellflow/plots/midbrain_braun_mapping/"

# Load model and data
print("Loading reference & query data")
adata_ref = sc.read(REF_PATH)
print(adata_ref)

adata_q = sc.read(QUERY_PATH)
print(adata_q)

adata_q.varm = dict()
adata_q.obs["batch"] = adata_q.obs["sample"].astype(str).copy()

adata_ref.varm = dict()
adata_ref.obs["batch"] = adata_ref.obs["Donor"].astype(str).copy()


print("Loading finetuned reference model")
ref_vae = scvi.model.SCANVI.load(
    os.path.join(MODEL_DIR, "model.pt"),
    adata=adata_ref,
)


# Map query to ref with 200 epochs
mapper = AtlasMapper(ref_vae)
mapper.map_query(
    adata_q,
    retrain="partial",
    max_epochs=200,
    batch_size=1024,
    plan_kwargs=dict(weight_decay=0.0),
)

X_scanvi = mapper.get_latent_representation()
adata_q.obsm["X_scanvi_braun"] = X_scanvi

adata_q.write(
    f"/projects/site/pred/organoid-atlas/data/local/perturbation/MidbrainPatterningNadya/midbrain_patterning_v2_common_hv2k.h5ad"
)

os.makedirs(f"{RESULTS_DIR}/midbrain_braun_scanvi", exist_ok=True)
mapper.save(f"{RESULTS_DIR}/midbrain_braun_scanvi/mapper.pkl")


sc.pp.neighbors(adata_q, use_rep="X_scanvi_braun", n_neighbors=30, method="rapids")
sc.tl.umap(adata_q, min_dist=0.3, method="rapids")


#### Get region predictions with logreg and wknn ###

with open(
    f"/home/fleckj/projects/cellflow/results/braun_region_pred/models/braun_region_l1_logreg.pkl",
    "rb",
) as f:
    reg = cloudpickle.load(f)

pred_region = reg["label_encoder"].inverse_transform(
    reg["model"].predict(adata_q.obsm["X_scanvi_braun"])
)

adata_q.obs["region_pred_logreg"] = pred_region


#### Get wknn transfer ####
mapper.compute_wknn(adata_ref)
region_scores = mapper.transfer_labels("Region")
subregion_scores = mapper.transfer_labels("Subregion")
class_scores = mapper.transfer_labels("CellClass")

adata_q.obs["region_pred_wknn"] = region_scores["best_label"]
adata_q.obs["subregion_pred_wknn"] = subregion_scores["best_label"]
adata_q.obs["class_pred_wknn"] = class_scores["best_label"]

p = sc.pl.scatter(
    adata_q,
    color=[
        "region_pred_wknn",
        "region_pred_logreg",
        "subregion_pred_wknn",
        "class_pred_wknn",
    ],
    basis="umap",
    show=False,
)
plt.savefig(f"{PLOT_DIR}/midbrain_braun_scanvi_umap_label_transfer.png")

adata_q.write(
    f"/projects/site/pred/organoid-atlas/data/local/perturbation/MidbrainPatterningNadya/midbrain_patterning_v2_common_hv2k_wknn.h5ad"
)

mapper.save(f"{RESULTS_DIR}/midbrain_braun_scanvi/mapper.pkl")


#### Highres clustering ####
from cugraph import leiden, Graph

sc.pp.neighbors(adata_q, use_rep="X_scanvi_braun", n_neighbors=30, method="rapids")
sc.tl.umap(adata_q, min_dist=0.2, method="rapids")

con_graph = Graph()
con_graph.from_numpy_array(adata_q.obsp["connectivities"].toarray())

leiden_clustering = leiden(con_graph, resolution=50)
clusters_sorted = leiden_clustering[0]["partition"].values.get()[
    np.argsort(leiden_clustering[0]["vertex"].values.get())
]

adata_q.obs["leiden_50"] = clusters_sorted
adata_q.obs["leiden_50"] = adata_q.obs["leiden_50"].astype("category")

p = sc.pl.scatter(adata_q, color="leiden_50", basis="umap", show=False)
plt.savefig(f"{PLOT_DIR}/midbrain_braun_scanvi_umap_leiden10.png")

adata_q.write(
    f"/projects/site/pred/organoid-atlas/data/local/perturbation/MidbrainPatterningNadya/midbrain_patterning_v2_common_hv2k_wknn.h5ad"
)

dims_to_plot = [(d, d + 1) for d in range(0, 20, 2)]
for dims in dims_to_plot:
    p = sc.pl.scatter(
        adata_q,
        color="region_pred_logreg",
        basis="scanvi_braun",
        show=False,
        components=dims,
    )
    plt.savefig(
        f"{PLOT_DIR}/scanvi_latent/midbrain_braun_scanvi_umap_region_dims_{dims[0]}_{dims[1]}.png"
    )
