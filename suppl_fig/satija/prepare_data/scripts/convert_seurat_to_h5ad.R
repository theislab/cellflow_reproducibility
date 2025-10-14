library(reticulate)
use_condaenv(conda_env_name, required = TRUE)

library(Seurat)
library(sceasy)
library(anndata)
anndata <- reticulate::import('anndata')
ins <- readRDS(file='./data/satija_raw/Seurat_object_INS_Perturb_seq.rds')
ins[["RNA"]]$data <- ins[["RNA"]]$counts
sceasy::convertFormat(ins, from="seurat", to="anndata", outFile='data/INS_Perturb_seq_raw.h5ad')

ifng <- readRDS(file='./data/satija_raw/Seurat_object_IFNG_Perturb_seq.rds')
ifng[["RNA"]]$data <- ifng[["RNA"]]$counts
sceasy::convertFormat(ifng, from="seurat", to="anndata", outFile='data/IFNG_Perturb_seq_raw.h5ad')

tgfb <- readRDS(file='./data/satija_raw/Seurat_object_TGFB_Perturb_seq.rds')
tgfb[["RNA"]]$data <- tgfb[["RNA"]]$counts
sceasy::convertFormat(tgfb, from="seurat", to="anndata", outFile='data/TGFB_Perturb_seq_raw.h5ad')

tnfa <- readRDS(file='./data/satija_raw/Seurat_object_TNFA_Perturb_seq.rds')
tnfa[["RNA"]]$data <- tnfa[["RNA"]]$counts
sceasy::convertFormat(tnfa, from="seurat", to="anndata", outFile='data/TNFA_Perturb_seq_raw.h5ad')

ifnb <- readRDS(file='./data/satija_raw/Seurat_object_IFNB_Perturb_seq.rds')
ifnb[["RNA"]]$data <- ifnb[["RNA"]]$counts
sceasy::convertFormat(ifnb, from="seurat", to="anndata", outFile='data/IFNB_Perturb_seq_raw.h5ad')
