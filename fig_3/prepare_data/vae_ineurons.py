import scanpy as sc
import numpy as np

from cfp.external._scvi import CFJaxSCVI

SMALL_DIR = "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/results/vae_ineurons/single"
FULL_DIR = (
    "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/results/vae_ineurons/full"
)
FINAL_DIR = (
    "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/results/vae_ineurons/"
)
test_combs = [
    "XAV+BMP4",
    "CHIR+BMP4",
    "RA+BMP4",
    "FGF8+BMP4",
    "XAV+SHH",
    "CHIR+SHH",
    "RA+SHH",
    "FGF8+SHH",
    "RA+CHIR+ctrl",
    "FGF8+CHIR+ctrl",
    "RA+CHIR+BMP4",
    "FGF8+CHIR+BMP4",
    "RA+CHIR+SHH",
    "FGF8+CHIR+SHH",
]

adata = sc.read(
    "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/data/morphogens/ineurons_glutpost.h5ad"
)
adata_all = adata.copy()

# filter out test combinations
test_filt = np.zeros(adata.n_obs, dtype=bool)
for test_comb in test_combs:
    mols = test_comb.split("+") if test_comb != "ctrl+ctrl" else [test_comb]
    mols = mols + ["ctrl"] if len(mols) == 1 and mols[0] != "ctrl+ctrl" else mols
    test_filt = test_filt | adata.obs["comb"].apply(
        lambda x: all([mol in x for mol in mols])
        and all([mol in mols for mol in x.split("+")])
    )
adata = adata[~test_filt].copy()

EPOCHS = 500
BATCH_SIZE = 1024

#### Train FactorVI model with counts ####
CFJaxSCVI.setup_anndata(adata, layer="counts")
vae = CFJaxSCVI(adata, n_hidden=512, n_layers=2, n_latent=20, gene_likelihood="nb")

vae.train(
    max_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    plan_kwargs=dict(n_epochs_kl_warmup=int(EPOCHS / 2)),
    early_stopping=True,
)

vae.save(f"{SMALL_DIR}/vae/factorvi_counts/", overwrite=True)

#### Get latent representation and reconstructed counts ####
vae = CFJaxSCVI.load(f"{SMALL_DIR}/vae/factorvi_counts/", adata=adata)
adata.obsm["X_latent_counts"] = vae.get_latent_representation()
X_recon = vae.get_reconstructed_expression(adata, use_rep="X_latent_counts")
adata.layers["counts_recon"] = X_recon
adata.write(f"{SMALL_DIR}/adata_hv4k_vae.h5ad")


#### Train on all combinations ####
adata_all = sc.read(
    "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/data/morphogens/ineurons_glutpost.h5ad"
)
CFJaxSCVI.setup_anndata(adata_all, layer="counts")
vae_all = CFJaxSCVI(
    adata_all, n_hidden=512, n_layers=2, n_latent=20, gene_likelihood="nb"
)

vae_all.train(
    max_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    plan_kwargs=dict(n_epochs_kl_warmup=int(EPOCHS / 2)),
    early_stopping=True,
)

vae_all.save(f"{FULL_DIR}/vae/factorvi_counts/", overwrite=True)

#### Get latent representation and reconstructed counts ####
vae_all = CFJaxSCVI.load(f"{FULL_DIR}/vae/factorvi_counts/", adata=adata_all)
adata_all.obsm["X_latent_counts"] = vae_all.get_latent_representation()
X_recon = vae_all.get_reconstructed_expression(adata_all, use_rep="X_latent_counts")
adata_all.layers["counts_recon"] = X_recon
adata_all.write(f"{FULL_DIR}/adata_hv4k_vae.h5ad")


#### Project all cells using VAE on single ####
adata_all.obsm["X_latent_single"] = vae.get_latent_representation(adata_all)
adata_all.write(f"{FINAL_DIR}/adata_hv4k_vae.h5ad")
