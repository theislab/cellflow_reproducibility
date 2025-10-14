source('~/scripts/R/master.R')
source('~/scripts/R/colors.R')

select <- dplyr::select

setwd('~/projects/cellflow/')

model_colors <- c(
    "mean" = "#BDBDBD",
    "barycenter" = "#BDBDBD",
    "union" = "#8f97a8",
    "background" = "#8f97a8",
    "train_dataset" = "#8f97a8",
    "individual" = "#566573",
    "single" = "#566573",
    "closest" = "#566573",
    "cpa" = "#26B0B6",
    "biolord" = "#BD6DED",
    "cellflow" = "#B12F8C"
)

cond_colors <- c(
    "RA" = "#69B79A",
    "BMP4" = "#FFC94F",
    "RA+BMP4" = "#4570B4",
    "other" = "lightgrey"
)

cond_colors2 <- c(
    "RA_gt" = "#69B79A",
    "BMP4_gt" = "#FFC94F",
    "RA+BMP4_gt" = "#4570B4",
    "RA+BMP4_pred" = "#B12F8C",
    "RA+BMP4_cpa" = "#26B0B6",
    "RA+BMP4_biolord" = "#BD6DED",
    "RA+BMP4_mean" = "#BDBDBD",
    "RA+BMP4_union" = "#8f97a8",
    "other" = "lightgrey"
)


region_colors <- c(
    "Spinal cord"="#FFC01E", "ENS"="#a1887f", "Forebrain"="#f06292",
    "Hindbrain"="#5c6bc0", "Midbrain"="#af7ac5", "SYM"="#D2927D", 
    "TG"="#4db6ac", "DRG"="#00bcd4"
)
type_colors <- c("GLUT"="#EB984E", "CHO"="#F4D03F", "NOR"="#AF7AC5")


#### Load data ####
PLOT_DIR = "/home/fleckj/projects/cellflow/plots/paper/figure_5/"
dir.create(PLOT_DIR, showWarnings = FALSE)

# Baselines
BASELINE_DIR = "/projects/site/pred/organoid-atlas/USERS/bobrovsd/results/baselines/combined/"
baseline_files <- list.files(BASELINE_DIR, pattern = "*.csv", recursive = TRUE, full.names = TRUE)
names(baseline_files) <- baseline_files

dist_baseline_files <- baseline_files[!str_detect(baseline_files, "_cluster_")]
cluster_baseline_files <- baseline_files[str_detect(baseline_files, "_cluster_")]

dist_baseline_df <- map_dfr(dist_baseline_files, read_csv, .id = "file")
cluster_baseline_df <- map_dfr(cluster_baseline_files, read_csv, .id = "file")

cluster_baseline_df %>% filter(is.na(method))  %>% pull(file) %>% head()
cluster_baseline_df %>% filter(!is.na(method))  %>% pull(file) %>% head()

dist_baseline_df$model <- dist_baseline_df$method
dist_baseline_df$comb <- str_replace_all(dist_baseline_df$comb, "_", "+")
cluster_baseline_df$model <- cluster_baseline_df$method
cluster_baseline_df$comb <- str_replace(cluster_baseline_df$file, ".+_(.+).csv", "\\1")

# Metrics
DATA_DIR <- "/projects/site/pred/organoid-atlas/USERS/bobrovsd/results/runs/bestsw/"
metrics_files <- list.files(DATA_DIR, pattern = "*.csv", recursive = TRUE, full.names = TRUE)
metrics_files <- metrics_files[str_detect(metrics_files, "metrics")]
metrics_files <- metrics_files[!str_detect(metrics_files, "new_results")]
names(metrics_files) <- metrics_files

dist_metrics_files <- metrics_files[str_detect(metrics_files, "_cfp.csv")]
cluster_metrics_files <- metrics_files[str_detect(metrics_files, "_cluster.csv")]

dist_metrics_df <- map_dfr(dist_metrics_files, read_csv, .id = "file") %>% 
    mutate(model=case_when(
        str_detect(file, "20241209_034509_262") ~ "biolord",
        str_detect(file, "20241130_101236_868") ~ "cellflow",
        str_detect(file, "20250110_191253_264") ~ "cpa",
        T ~ "NA"
    )) %>% 
    filter(model!="NA") %>%
    select(-file)

cluster_metrics_df <- map_dfr(cluster_metrics_files, read_csv, .id = "file") %>% 
    mutate(model=case_when(
        str_detect(file, "20241209_034509_262") ~ "biolord",
        str_detect(file, "20241130_101236_868") ~ "cellflow",
        str_detect(file, "20250110_191253_264") ~ "cpa",
        T ~ "NA"
    )) %>% 
    filter(model!="NA") %>%
    select(-file)

adata_pred <- anndata::read_h5ad(str_c(DATA_DIR, "20241130_101236_868/RA+BMP4/RA+BMP4_pred.h5ad"))
adata_pred_cpa <- anndata::read_h5ad(str_c(DATA_DIR, "20250110_191253_264/RA+BMP4/RA+BMP4_pred.h5ad"))
adata_pred_biolord <- anndata::read_h5ad(str_c(DATA_DIR, "20241209_034509_262/RA+BMP4/RA+BMP4_pred.h5ad"))
adata_gt <- anndata::read_h5ad("/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/data/ineurons_glutpost.h5ad")
ineuron_meta <- read_tsv("/home/fleckj/projects/juniper/data/ineuron/iNeurons_dr_clustered_raw_merged_meta.tsv") %>% 
    rename("cell"=1) %>% 
    filter(str_detect(cell, "iGlut_post_")) %>%
    mutate(cell=str_replace(cell, "iGlut_post_p\\d_", ""))


apdv_meta <- ineuron_meta %>% 
    select(cell, AP_axis, DV_axis)

umap_df <- adata_pred$obsm$X_umap %>% 
    as_tibble() %>% 
    set_names(c("UMAP1", "UMAP2"))

rownames(adata_pred$obs) <- 1:nrow(adata_pred$obs)
pred_meta <- adata_pred$obs %>% 
    as_tibble(rownames="cell") %>% 
    bind_cols(umap_df) 

umap_df <- adata_gt$obsm$X_umap %>% 
    as_tibble() %>% 
    set_names(c("UMAP1", "UMAP2"))

gt_meta <- adata_gt$obs %>% 
    as_tibble(rownames='cell') %>% 
    bind_cols(umap_df) %>% 
    # inner_join(ineuron_meta, by=c("cell"="cell")) %>%
    mutate(status="gt")

cluster_meta <- gt_meta %>% 
    select(leiden_4, Neuron_type, Region) %>% 
    group_by(leiden_4) %>%
    summarize(
        Neuron_type = mode(Neuron_type),
        Region = mode(Region)
    )


#### Plot metrics for figure ####
model_order <- c(
    "closest"="Closest train\ncondition",
    "single"="Closest single\nmorphogen",
    "cellflow"="CellFlow",
    "biolord"="biolord",
    "cpa"="CPA",
    "union"="Union",
    "barycenter"="Barycenter"
)

dist_metrics_df %>% View()

plot_df <- dist_metrics_df %>% 
    mutate(
        mmd=mmd_pca_reproj,
        sinkhorn_div_1=sinkhorn_div_1_pca_reproj,
        e_distance=e_distance_pca_reproj,
        comb=str_replace_all(condition, "_\\d+", ""),
    ) %>% 
    bind_rows(dist_baseline_df) %>%
    mutate(
        model=factor(model, levels=names(model_order)),
    ) %>% 
    filter(!comb%in%c("CHIR+BMP4", "FGF8+SHH")) %>%
    filter(!comb%in%c("FGF8+CHIR", "RA+CHIR")) %>%
    filter(!model%in%c("single", "closest"), space=="reproj" | is.na(space))

# plot_df %>% filter(model=="closest") %>% arrange(condition) %>% View

plot_df %>% filter(model=="cellflow") %>% pull(e_distance) %>% mean()
plot_df %>% filter(model=="biolord") %>% pull(e_distance) %>% mean()


ggplot(plot_df, aes(x=e_distance, y=model, fill=model)) +
    geom_boxplot(size=0.2,  width=0.8, outlier.size=0.2, outlier.shape=16) +
    scale_fill_manual(values=model_colors) +
    scale_y_discrete(labels=model_order) +
    scale_x_log10(breaks=c(1,10,100), limits=c(1,NA)) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank()
    ) +
    labs(y="Model", x="Energy distance") 
ggsave(paste0(PLOT_DIR, "benchmark_energy_distance.pdf"), unit="cm", width=4.5, height=2.4)


ggplot(plot_df, aes(x=sinkhorn_div_1, y=model, fill=model)) +
    geom_boxplot(size=0.2,  width=0.8, outlier.size=0.2, outlier.shape=16) +
    scale_fill_manual(values=model_colors) +
    scale_y_discrete(labels=model_order) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank()
    ) +
    labs(y="Model", x="Energy distance") 
ggsave(paste0(PLOT_DIR, "benchmark_sinkhorn_div.pdf"), unit="cm", width=4.5, height=2.4)


ggplot(plot_df, aes(x=mmd, y=model, fill=model)) +
    geom_boxplot(size=0.2,  width=0.8, outlier.size=0.2, outlier.shape=16) +
    scale_fill_manual(values=model_colors) +
    scale_y_discrete(labels=model_order) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank()
    ) +
    labs(y="Model", x="Energy distance") 
ggsave(paste0(PLOT_DIR, "benchmark_mmd.pdf"), unit="cm", width=4.5, height=2.4)


plot_df <- cluster_metrics_df %>% 
    mutate(
        comb=str_replace_all(condition, "_\\d+", ""),
    ) %>% 
    bind_rows(cluster_baseline_df) %>%
    mutate(
        model=factor(model, levels=names(model_order)),
    ) %>% 
    filter(!comb%in%c("CHIR+BMP4", "FGF8+SHH")) %>%
    filter(!comb%in%c("FGF8+CHIR", "RA+CHIR")) %>%
    filter(!model%in%c("single", "closest"), !is.na(model), label_key=="leiden_4", n_neighbors==30)

plot_df %>% filter(model=="cellflow") %>% pull(cosine) %>% mean()
plot_df %>% filter(model=="union") %>% pull(cosine) %>% mean()

(1-0.297)/(1-0.097)

ggplot(plot_df, aes(x=1-cosine, y=model, fill=model)) +
    geom_boxplot(size=0.2,  width=0.8, outlier.size=0.2, outlier.shape=16) +
    scale_fill_manual(values=model_colors) +
    scale_y_discrete(labels=model_order) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank()
    ) +
    labs(y="Model", x="Cosine similarity") 
ggsave(paste0(PLOT_DIR, "benchmark_cluster_cosine.pdf"), unit="cm", width=4.5, height=2.4)



#### Plots for supplement with comparison to single/closest conditions and more metrics ####
plot_df <- dist_metrics_df %>% 
    mutate(
        mmd=mmd_pca_reproj,
        sinkhorn_div_1=sinkhorn_div_1_pca_reproj,
        e_distance=e_distance_pca_reproj,
        comb=str_replace_all(condition, "_\\d+", ""),
    ) %>% 
    bind_rows(dist_baseline_df) %>%
    mutate(
        model=factor(model, levels=names(model_order)),
    ) %>% 
    filter(!comb%in%c("CHIR+BMP4", "FGF8+SHH")) %>%
    filter(!comb%in%c("FGF8+CHIR", "RA+CHIR")) %>%
    filter(model%in%c("single", "closest", "cellflow"), space=="reproj" | is.na(space))

ggplot(plot_df, aes(x=e_distance, y=model, fill=model)) +
    geom_boxplot(size=0.2,  width=0.8, outlier.size=0.2, outlier.shape=16) +
    scale_fill_manual(values=model_colors) +
    scale_y_discrete(labels=model_order) +
    scale_x_log10() +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank()
    ) +
    labs(y="Model", x="Energy distance") 
ggsave(paste0(PLOT_DIR, "supp_energy_distance.pdf"), unit="cm", width=4.5, height=2.3)


plot_df <- cluster_metrics_df %>% 
    mutate(
        comb=str_replace_all(condition, "_\\d+", ""),
    ) %>% 
    filter(!comb%in%c("CHIR+BMP4", "FGF8+SHH")) %>%
    bind_rows(cluster_baseline_df) %>%
    mutate(
        model=factor(model, levels=names(model_order)),
    ) %>% 
    filter(model%in%c("single", "closest", "cellflow"), !is.na(model), label_key=="leiden_4", n_neighbors==30)

# plot_df %>% filter(model=="closest") %>% arrange(condition) %>% View

ggplot(plot_df, aes(x=1-cosine, y=model, fill=model)) +
    geom_boxplot(size=0.2,  width=0.8, outlier.size=0.2, outlier.shape=16) +
    scale_fill_manual(values=model_colors) +
    scale_y_discrete(labels=model_order) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank()
    ) +
    labs(y="Model", x="Cosine similarity") 
ggsave(paste0(PLOT_DIR, "supp_cluster_cosine.pdf"), unit="cm", width=4.5, height=2.3)



#### Metrics per condition ####

plot_df <- dist_metrics_df %>% 
    mutate(
        mmd=mmd_pca_reproj,
        sinkhorn_div_1=sinkhorn_div_1_pca_reproj,
        e_distance=e_distance_pca_reproj,
        comb=str_replace_all(condition, "_\\d+", ""),
    ) %>% 
    bind_rows(dist_baseline_df) %>%
    # filter(!comb%in%c("CHIR+BMP4", "FGF8+SHH")) %>%
    filter(!comb%in%c("FGF8+CHIR", "RA+CHIR")) %>%
    mutate(
        model=factor(model, levels=names(model_order)),
    ) %>% 
    filter(!model%in%c("single", "closest"), space=="reproj" | is.na(space))

# plot_df %>% filter(model=="closest") %>% arrange(condition) %>% View

ggplot(plot_df, aes(x=e_distance, y=model, fill=model)) +
    geom_boxplot(size=0.2,  width=0.8, outlier.size=0.2, outlier.shape=16) +
    scale_fill_manual(values=model_colors) +
    scale_y_discrete(labels=model_order) +
    scale_x_log10() +
    facet_wrap(~comb) +
    article_text() +
    no_legend() +
    theme(
        panel.border = element_rect(color = "darkgrey", fill = NA, linewidth = 0.5),
        panel.spacing = unit(0.1, "lines"),
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(),
        strip.text = element_text(size=6)
    ) +
    labs(y="Model", x="Energy distance") 
ggsave(paste0(PLOT_DIR, "benchmark_per_comb_energy_distance.pdf"), unit="cm", width=10, height=6)



#### Predictions and UMAP coords ####
# Check RA+BMP4
plot_df <- pred_meta %>% 
    filter(comb=="RA+BMP4", condition%in%c("RA_3+BMP4_3")) %>% 
    mutate(status="pred") %>%
    bind_rows(sample_n(gt_meta, 50000)) %>% 
    arrange(status)

p2 <- ggplot(plot_df, aes(x=UMAP1, y=UMAP2, color=status)) +
    geom_point(size=1.5) +
    scale_color_manual(values=c("pred"=model_colors[["cellflow"]], "gt"="lightgrey")) +
    theme_dr()


plot_df <- sample_n(gt_meta, 50000) %>% 
    mutate(plot_col=case_when(
        comb%in%c("RA", "BMP4") ~ comb,
        condition%in%c("RA_3+BMP4_3") ~ comb,
        TRUE ~ "other"
    )) %>% 
    arrange(plot_col!="other")

p1 <- ggplot(plot_df, aes(x=UMAP1, y=UMAP2, color=plot_col)) +
    geom_point(size=1.5) +
    scale_color_manual(values=cond_colors) +
    theme_dr()

p1 | p2
ggsave(str_c(PLOT_DIR, "RA+BMP4_pred_gt.png"), width=16, height=6)


plot_df <- gt_meta %>% 
    mutate(training_set=!comb%in%c("RA+BMP4", "RA+CHIR+BMP4")) %>% 
    arrange(training_set)

p <- ggplot(plot_df, aes(x=UMAP1, y=UMAP2, color=training_set)) +
    geom_point() +
    scale_color_manual(values=c("TRUE"="black", "FALSE"="lightgray")) +
    theme_dr() +
    no_legend()
ggsave(p, filename=str_c(PLOT_DIR, "training_set.png"), width=10, height=8)



#### Plot cluster distributions ####
plot_conds <- c("RA_3+BMP4_3")
plot_combs <- c("RA", "BMP4")

gt_df <- gt_meta %>% 
    filter((condition %in% plot_conds) | (comb %in% plot_combs))

pred_df <- pred_meta %>% 
    filter(condition %in% plot_conds) %>% 
    mutate(leiden_4=leiden_4_transfer)

plot_df %>% filter(is.na(Neuron_type))
plot_df %>% filter(is.na(Neuron_type))

region_prop_df <- bind_rows("GT"=gt_df, "pred"=pred_df, .id="source") %>% 
    select(-c(Neuron_type, Region)) %>%
    inner_join(cluster_meta, by="leiden_4") %>%
    group_by(leiden_4) %>% 
    group_by(comb, source) %>% 
    mutate(
        total_count=n()
    ) %>%
    group_by(comb, leiden_4, source) %>%
    summarize(
        Neuron_type = mode(Neuron_type),
        Region = mode(Region),
        clust_frac = n()/first(total_count),
        clust_count = n(),
    ) %>%
    arrange(Neuron_type)  %>% 
    mutate(
        leiden_4 = factor(leiden_4, levels=unique(.$leiden_4)),
        cond = paste0(comb, "_", source)
    )

region_prop_mat <- region_prop_df %>% 
    ungroup() %>%
    select(cond, leiden_4, clust_frac) %>%
    pivot_wider(names_from=leiden_4, values_from=clust_frac, values_fill=0) %>%
    column_to_rownames("cond") %>% as.matrix()

cluster_order <- region_prop_mat %>% t() %>% dist() %>% hclust() %>% {.$labels[.$order]}

plot_df <- region_prop_df %>% 
    mutate(leiden_4=factor(leiden_4, levels=cluster_order)) %>% 
    arrange(Neuron_type, leiden_4) %>% 
    mutate(leiden_4=factor(leiden_4, levels=unique(.$leiden_4))) 

ggplot(plot_df, aes(x=leiden_4, y=clust_frac, fill=Neuron_type)) +
    geom_bar(stat="identity") +
    scale_fill_manual(values=type_colors) +
    scale_y_continuous(breaks=c(0,1)) +
    facet_grid(cond~., scales="free") +
    article_text() +
    no_x_text() +
    no_y_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2)
    ) +
    labs(x="Leiden cluster", y="Fraction of cells", fill="Neuron type")
ggsave(str_c(PLOT_DIR, "gt_pred_neuron_type_bar.pdf"), unit="cm", height=2.5, width=6.5)


plot_df <- region_prop_df %>% 
    mutate(leiden_4=factor(leiden_4, levels=cluster_order))

p1 <- ggplot(plot_df, aes(x=leiden_4, y=cond, fill=clust_frac)) +
    geom_tile() +
    article_text() +
    no_x_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2)
    ) +
    labs(x="Leiden cluster", y="log10(# cells)", fill="Neuron type")
p1
ggsave(str_c(PLOT_DIR, "gt_pred_neuron_type_heatmap.pdf"), unit="cm", height=2.5, width=6.5)


#### Plot marginals ####
adata_pred <- anndata::read_h5ad(str_c(DATA_DIR, "20241130_101236_868/RA+BMP4/RA+BMP4_pred.h5ad"))
adata_pred_cpa <- anndata::read_h5ad(str_c(DATA_DIR, "20250110_191253_264/RA+BMP4/RA+BMP4_pred.h5ad"))
adata_pred_biolord <- anndata::read_h5ad(str_c(DATA_DIR, "20241209_034509_262/RA+BMP4/RA+BMP4_pred.h5ad"))
adata_pred_mean <- anndata::read_h5ad("/projects/site/pred/organoid-atlas/USERS/bobrovsd/results/baselines/adata_mean_RA+BMP4.h5ad")
adata_pred_union <- anndata::read_h5ad("/projects/site/pred/organoid-atlas/USERS/bobrovsd/results/baselines/adata_union_RA+BMP4.h5ad")
adata_gt <- anndata::read_h5ad("/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/data/ineurons_glutpost.h5ad")
ineuron_meta <- read_tsv("/home/fleckj/projects/juniper/data/ineuron/iNeurons_dr_clustered_raw_merged_meta.tsv") %>% 
    rename("cell"=1) %>% 
    filter(str_detect(cell, "iGlut_post_")) %>%
    mutate(cell=str_replace(cell, "iGlut_post_p\\d_", ""))

plot_conds <- c("RA_3+BMP4_3")
plot_combs <- c("RA", "BMP4")

gt_df <- gt_meta %>% 
    filter((condition %in% plot_conds) | (comb %in% plot_combs))

pred_df <- pred_meta %>% 
    filter(condition %in% plot_conds) %>% 
    mutate(leiden_4=leiden_4_transfer)

latent_pred_use <- adata_pred$obsm$X_pca_reproj[pred_meta$cell%in%pred_df$cell,]
rownames(latent_pred_use) <- pred_df$cell
colnames(latent_pred_use) <- 1:ncol(latent_pred_use)
latent_gt_use <- adata_gt$obsm$X_pca[gt_meta$cell%in%gt_df$cell,]
rownames(latent_gt_use) <- gt_df$cell
colnames(latent_gt_use) <- 1:ncol(latent_gt_use)

latent_pred_df <- latent_pred_use %>% 
    as_tibble(rownames="cell") %>% 
    pivot_longer(-cell, names_to="PC", values_to="value") %>% 
    mutate(source="pred") %>% 
    inner_join(pred_df)

latent_gt_df <- latent_gt_use %>%
    as_tibble(rownames="cell") %>% 
    pivot_longer(-cell, names_to="PC", values_to="value") %>% 
    mutate(source="gt") %>% 
    inner_join(gt_df)

latent_all <- bind_rows("pred"=latent_pred_df, "gt"=latent_gt_df, .id="source") %>% 
    filter(as.numeric(PC)<=5) %>% 
    mutate(
        cond_plot=factor(str_c(comb, "_", source), levels = rev(c("RA_gt", "BMP4_gt", "RA+BMP4_gt", "RA+BMP4_pred"))),
        PC=factor(PC, levels=1:20)
    ) 


ggplot(latent_all, aes(y=cond_plot, x=value, fill=cond_plot)) +
    geom_density_ridges(alpha=0.5, scale=10, linewidth=0.2) +
    facet_wrap(~PC, scales="free_x") +
    scale_fill_manual(values=cond_colors2) +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2)
    ) +
    article_text() +
    no_legend() +
    no_margin() +
    no_y_text()
ggsave(str_c(PLOT_DIR, "latent_marginals.pdf"), unit="cm", height=4, width=7)


# Compare with cpa, biolord and baselines
rownames(adata_pred_cpa$obs) <- 1:nrow(adata_pred_cpa$obs)
cpa_df <- adata_pred_cpa$obs %>% 
    as_tibble(rownames="cell") %>% 
    filter(condition %in% plot_conds) %>% 
    mutate(comb=str_replace_all(condition, "_\\d+", ""))

latent_cpa_use <- adata_pred_cpa$obsm$X_pca_reproj[rownames(adata_pred_cpa$obs)%in%cpa_df$cell,]
rownames(latent_cpa_use) <- cpa_df$cell
colnames(latent_cpa_use) <- 1:ncol(latent_cpa_use)

rownames(adata_pred_biolord$obs) <- 1:nrow(adata_pred_biolord$obs)
biolord_df <- adata_pred_biolord$obs %>% 
    as_tibble(rownames="cell") %>% 
    filter(condition %in% plot_conds)

latent_biolord_use <- adata_pred_biolord$obsm$X_pca_reproj[rownames(adata_pred_biolord$obs)%in%biolord_df$cell,]
rownames(latent_biolord_use) <- biolord_df$cell
colnames(latent_biolord_use) <- 1:ncol(latent_biolord_use)

rownames(adata_pred_mean$obs) <- 1:nrow(adata_pred_mean$obs)
mean_df <- adata_pred_mean$obs %>% 
    as_tibble(rownames="cell") %>% 
    filter(condition %in% plot_conds) %>% 
    mutate(comb=str_replace_all(condition, "_\\d+", ""))

latent_mean_use <- adata_pred_mean$obsm$X_pca_reproj[rownames(adata_pred_mean$obs)%in%mean_df$cell,]
rownames(latent_mean_use) <- mean_df$cell
colnames(latent_mean_use) <- 1:ncol(latent_mean_use)

rownames(adata_pred_union$obs) <- 1:nrow(adata_pred_union$obs)
union_df <- adata_pred_union$obs %>% 
    as_tibble(rownames="cell") %>% 
    filter(condition %in% plot_conds) %>% 
    mutate(comb=str_replace_all(condition, "_\\d+", ""))

latent_union_use <- adata_pred_union$obsm$X_pca_reproj[rownames(adata_pred_union$obs)%in%union_df$cell,]
rownames(latent_union_use) <- union_df$cell
colnames(latent_union_use) <- 1:ncol(latent_union_use)

latent_cpa_df <- latent_cpa_use %>% 
    as_tibble(rownames="cell") %>% 
    pivot_longer(-cell, names_to="PC", values_to="value") %>% 
    mutate(source="cpa") %>% 
    inner_join(cpa_df)

latent_biolord_df <- latent_biolord_use %>%
    as_tibble(rownames="cell") %>% 
    pivot_longer(-cell, names_to="PC", values_to="value") %>% 
    mutate(source="biolord") %>% 
    inner_join(biolord_df)

latent_mean_df <- latent_mean_use %>%
    as_tibble(rownames="cell") %>% 
    pivot_longer(-cell, names_to="PC", values_to="value") %>% 
    mutate(source="mean") %>% 
    inner_join(mean_df)

latent_union_df <- latent_union_use %>%
    as_tibble(rownames="cell") %>% 
    pivot_longer(-cell, names_to="PC", values_to="value") %>% 
    mutate(source="union") %>% 
    inner_join(union_df)
    
latent_comp <- bind_rows("pred"=latent_pred_df, "gt"=latent_gt_df, "cpa"=latent_cpa_df, "biolord"=latent_biolord_df, "mean"=latent_mean_df, "union"=latent_union_df) %>%
    # filter(as.numeric(PC)<=5) %>% 
    # filter(condition %in% plot_conds) %>% 
    mutate(
        cond_plot=factor(str_c(comb, "_", source), levels = rev(c("RA_gt", "BMP4_gt", "RA+BMP4_gt", "RA+BMP4_pred", "RA+BMP4_cpa", "RA+BMP4_biolord", "RA+BMP4_mean", "RA+BMP4_union"))),
        PC=factor(str_c("PC", PC), levels=str_c("PC", 1:20))
    )
latent_comp$source %>% unique()

ggplot(latent_comp, aes(y=cond_plot, x=value, fill=cond_plot)) +
    geom_density_ridges(alpha=0.5, scale=2, linewidth=0.2) +
    facet_wrap(~PC, scales="free_x") +
    scale_fill_manual(values=cond_colors2) +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2),
        strip.text = element_text(size=6),
        panel.spacing = unit(0, "lines")
    ) +
    article_text() +
    no_legend() +
    no_margin() +
    no_y_text() +
    labs(x="Principal component", y="Condition")
ggsave(str_c(PLOT_DIR, "latent_marginals_comp.pdf"), unit="cm", height=9.5, width=12)


#### Ground truth UMAPs ####
p <- ggplot(gt_meta, aes(x=UMAP1, y=UMAP2, color=Neuron_type, fill=Neuron_type)) +
    geom_point(size=1.2, shape=21, color="black") +
    geom_point(size=1.2, shape=21, stroke=0) +
    scale_color_manual(values=type_colors) +
    scale_fill_manual(values=type_colors) +
    theme_dr() +
    no_legend()
ggsave(p, filename=str_c(PLOT_DIR, "gt_neuron_type.png"), width=10, height=8)

p <- ggplot(gt_meta, aes(x=UMAP1, y=UMAP2, color=Region, fill=Region)) +
    geom_point(size=1.2, shape=21, color="black") +
    geom_point(size=1.2, shape=21, stroke=0) +
    scale_color_manual(values=region_colors) +
    scale_fill_manual(values=region_colors) +
    theme_dr() +
    no_legend()
ggsave(p, filename=str_c(PLOT_DIR, "gt_region.png"), width=10, height=8)

# Sort "ctrl" first
ap_levels <- c(
    "ctrl", 
    "CHIR_1", "CHIR_2", "CHIR_3", "CHIR_4",
    "XAV_1", "XAV_2", "XAV_3", "XAV_4",
    "FGF8_1", "FGF8_2", "FGF8_3", "FGF8_4",
    "FGF8_1_CHIR", "FGF8_2_CHIR", "FGF8_3_CHIR", "FGF8_4_CHIR",
    "FGF8_1+CHIR_4", "FGF8_2+CHIR_4", "FGF8_3+CHIR_4", "FGF8_4+CHIR_4",
    "RA_1", "RA_2", "RA_3", "RA_4",
    "RA_1_CHIR", "RA_2_CHIR", "RA_3_CHIR", "RA_4_CHIR",
    "RA_1+CHIR_4", "RA_2+CHIR_4", "RA_3+CHIR_4", "RA_4+CHIR_4"
)
dv_levels <- c(
    "BMP4_1", "BMP4_2", "BMP4_3", 
    "SHH_1", "SHH_2", "SHH_3", "SHH_4",
    "ctrl"
)

gt_meta$AP_axis %>% unique()

plot_df <- gt_meta %>% 
    mutate(
        AP_axis=factor(AP_axis, levels=ap_levels),
        DV_axis=factor(DV_axis, levels=dv_levels)
    )

ggplot(plot_df, aes(AP_axis, fill=Region)) +
    geom_bar(position="fill") +
    facet_grid(DV_axis~.) +
    scale_fill_manual(values=region_colors) +
    scale_y_continuous(breaks=c(0,1)) 

ggplot(plot_df, aes(AP_axis, fill=Region)) +
    geom_bar(position="fill") +
    facet_grid(DV_axis~.) +
    scale_fill_manual(values=region_colors) +
    scale_y_continuous(breaks=c(0,1)) +
    article_text() +
    rotate_x_text(90) +
    no_x_text() +
    no_legend() +
    no_label() +
    theme(
        strip.text = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.spacing = unit(0, "lines")
    )
    
ggsave(str_c(PLOT_DIR, "gt_region_ap_dv.pdf"), unit="cm", height=4.4, width=4.4)




#### Analyze covariance in latent space ####
adata_pred <- anndata::read_h5ad(str_c(DATA_DIR, "20241130_101236_868/CHIR+BMP4/CHIR+BMP4_pred.h5ad"))
adata_pred_cpa <- anndata::read_h5ad(str_c(DATA_DIR, "20250110_191253_264/CHIR+BMP4/CHIR+BMP4_pred.h5ad"))
adata_pred_biolord <- anndata::read_h5ad(str_c(DATA_DIR, "20241209_034509_262/CHIR+BMP4/CHIR+BMP4_pred.h5ad"))

latent_pred <- adata_pred$obsm$X_pca_reproj
latent_gt <- adata_gt$obsm$X_pca_reproj
latent_cpa <- adata_pred_cpa$obsm$X_pca_reproj
latent_biolord <- adata_pred_biolord$obsm$X_pca_reproj
adata_pred_cpa$obs$comb <- str_replace_all(adata_pred_cpa$obs$condition, "_\\d+", "")

adata_pred$obs$comb %>% unique()
combs_use <- intersect(adata_pred$obs$comb, adata_gt$obs$comb)

latent_pred_use <- adata_pred$obsm$X_pca_reproj[adata_pred$obs$comb=="CHIR+BMP4",]
latent_gt_use <- adata_gt$obsm$X_pca[adata_gt$obs$comb=="CHIR+BMP4",]
latent_biolord_use <- adata_pred_biolord$obsm$X_pca_reproj[adata_pred_biolord$obs$comb=="CHIR+BMP4",]
latent_cpa_use <- adata_pred_cpa$obsm$X_pca_reproj[adata_pred_cpa$obs$comb=="CHIR+BMP4",]

# latent_pred_use <- adata_pred$obsm$X_pca_reproj[adata_pred$obs$comb%in%combs_use,]
# latent_gt_use <- adata_gt$obsm$X_pca[adata_gt$obs$comb%in%combs_use,]
# latent_biolord_use <- adata_pred_biolord$obsm$X_pca_reproj[adata_pred_biolord$obs$comb%in%combs_use,]
# latent_cpa_use <- adata_pred_cpa$obsm$X_pca_reproj[adata_pred_cpa$obs$comb%in%combs_use,]

colnames(latent_pred_use) <- paste0(1:ncol(latent_pred_use))
colnames(latent_gt_use) <- paste0(1:ncol(latent_gt_use))
colnames(latent_biolord_use) <- paste0(1:ncol(latent_biolord_use))
colnames(latent_cpa_use) <- paste0(1:ncol(latent_cpa_use))

latent_cov_pred <- cov(latent_pred_use)
latent_cov_gt <- cov(latent_gt_use)
latent_cov_biolord <- cov(latent_biolord_use)
latent_cov_cpa <- cov(latent_cpa_use)

# latent_gt_order <- latent_cov_gt %>% dist() %>% hclust() %>% {.$labels[.$order]}
latent_gt_order <- rev(colnames(latent_pred_use))

latent_cov_pred_df <- latent_cov_pred[1:10, 1:10] %>% 
    as_tibble(rownames="row") %>% 
    pivot_longer(-row, names_to="col", values_to="value") %>% 
    mutate(
        row=factor(row, levels=latent_gt_order),
        col=factor(col, levels=rev(latent_gt_order))
    ) 

latent_cov_gt_df <- latent_cov_gt[1:10, 1:10] %>% 
    as_tibble(rownames="row") %>% 
    pivot_longer(-row, names_to="col", values_to="value") %>% 
    mutate(
        row=factor(row, levels=latent_gt_order),
        col=factor(col, levels=rev(latent_gt_order))
    ) 

latent_cov_biolord_df <- latent_cov_biolord[1:10, 1:10] %>%
    as_tibble(rownames="row") %>% 
    pivot_longer(-row, names_to="col", values_to="value") %>% 
    mutate(
        row=factor(row, levels=latent_gt_order),
        col=factor(col, levels=rev(latent_gt_order))
    )

latent_cov_cpa_df <- latent_cov_cpa[1:10, 1:10] %>%
    as_tibble(rownames="row") %>% 
    pivot_longer(-row, names_to="col", values_to="value") %>% 
    mutate(
        row=factor(row, levels=latent_gt_order),
        col=factor(col, levels=rev(latent_gt_order))
    )



clim <- 95
# clim <- max(abs(latent_cov_pred_df$value))
p1 <- ggplot(latent_cov_pred_df, aes(x=col, y=row, fill=value)) +
    geom_tile() +
    scale_fill_gradientn(colors=pals::ocean.balance(100), limits=c(-clim, clim)) +
    ggtitle("CellFlow")

# clim <- max(abs(latent_cov_gt_df$value))
p2 <- ggplot(latent_cov_gt_df, aes(x=col, y=row, fill=value)) +
    geom_tile() +
    scale_fill_gradientn(colors=pals::ocean.balance(100), limits=c(-clim, clim)) +
    ggtitle("Ground truth")

# clim <- max(abs(latent_cov_biolord_df$value))
p3 <- ggplot(latent_cov_biolord_df, aes(x=col, y=row, fill=value)) +
    geom_tile() +
    scale_fill_gradientn(colors=pals::ocean.balance(100), limits=c(-clim, clim)) +
    ggtitle("biolord")

# clim <- max(abs(latent_cov_cpa_df$value))
p4 <- ggplot(latent_cov_cpa_df, aes(x=col, y=row, fill=value)) +
    geom_tile() +
    scale_fill_gradientn(colors=pals::ocean.balance(100), limits=c(-clim, clim)) +
    ggtitle("CPA")

p1 + p2 + p3 + p4 &
    article_text() &
    no_legend() &
    labs(x="PC", y="PC") &
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2)
    )
ggsave(str_c(PLOT_DIR, "latent_covariance_CHIR+BMP4.pdf"), unit="cm", height=7, width=6)



#### Morphogen interaction scores ####
interaction_df <- read_csv("/home/fleckj/projects/cellflow/results/ineuron_interactions/interactions_from_single.csv")

interaction_pred <- interaction_df %>% 
    select(ap_cond, dv_cond, mmd_ap=mmd_ap_pred, mmd_dv=mmd_dv_pred, mmd_both=mmd_both_pred) %>% 
    mutate(type="pred")

interaction_pred$ap_cond %>% unique()

interaction_plot <- interaction_df %>% 
    select(ap_cond, dv_cond, mmd_ap, mmd_dv, mmd_both) %>%
    mutate(type="true") %>% 
    bind_rows(interaction_pred) %>% 
    mutate(
        ap_cond=factor(ap_cond, levels=ap_levels),
        dv_cond=factor(dv_cond, levels=rev(dv_levels))
    )


ggplot(interaction_plot, aes(ap_cond, dv_cond, fill=mmd_both)) +
    geom_tile() +
    facet_grid(type~.) +
    scale_fill_gradientn(colors=pals::brewer.blues(100), limits=c(0,0.063)) +
    article_text() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2)
    ) 
ggsave(str_c(PLOT_DIR, "interaction_scores.pdf"), unit="cm", height=4, width=6)
