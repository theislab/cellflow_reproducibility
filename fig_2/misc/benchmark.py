"""Unified timing and memory benchmark for CellFlow.

Measures:
  - Training: wall-clock per iteration, peak GPU memory, peak CPU memory
  - Inference: wall-clock per condition (varying number of cells), peak GPU memory

Outputs a single CSV with all results plus hardware metadata.

Usage
-----
  python benchmark.py                          # default: batch_size=1024
  python benchmark.py --batch_size 2048
  python benchmark.py --skip_training          # inference only (needs a trained model)
  python benchmark.py --n_cells 1000 5000 10000 50000
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
import platform
import threading
import time
import warnings

import jax
import numpy as np
import pandas as pd
import pynvml
import psutil

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def _to_str(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def get_hardware_info() -> dict:
    """Collect hardware metadata."""
    info = {
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "jax_backend": jax.default_backend(),
        "jax_devices": str(jax.devices()),
    }
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        info["gpu_name"] = _to_str(pynvml.nvmlDeviceGetName(h))
        info["gpu_memory_total_mb"] = round(mem.total / 1024**2)
    except Exception:
        info["gpu_name"] = "N/A"
        info["gpu_memory_total_mb"] = 0
    return info


# ---------------------------------------------------------------------------
# GPU memory sampler (reuse from existing code)
# ---------------------------------------------------------------------------

class GPUMemorySampler:
    """Background thread that polls GPU memory via NVML."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.01):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.interval_s = interval_s
        self._peak = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _used_mb(self):
        return pynvml.nvmlDeviceGetMemoryInfo(self.h).used / 1024**2

    def start(self):
        self._peak = self._used_mb()
        self._stop.clear()

        def _loop():
            while not self._stop.is_set():
                mb = self._used_mb()
                if mb > self._peak:
                    self._peak = mb
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return self._peak

    @property
    def current_mb(self):
        return self._used_mb()


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(params) -> int:
    """Count total trainable parameters in a JAX pytree."""
    leaves = jax.tree_util.tree_leaves(params)
    return sum(x.size for x in leaves)


# ---------------------------------------------------------------------------
# Data setup (PBMC dataset — same as existing benchmarks)
# ---------------------------------------------------------------------------

def setup_cellflow(batch_size: int = 1024):
    """Set up the CellFlow model on PBMC data, matching existing benchmark config."""
    import functools

    import flax.linen as nn
    import optax
    import scanpy as sc
    from cellflow.model import CellFlow
    from cellflow.utils import match_linear

    adata = sc.read_h5ad(
        "/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_with_pca.h5ad"
    )
    adata.obsm["X_pca_simulated"] = adata.X[:, :100].copy()
    adata.uns["donor_embeddings"] = {
        f"Donor{i}": np.random.normal(size=(2000,)) for i in range(1, 13)
    }
    adata.obs["condition"] = adata.obs.apply(
        lambda x: x["donor"] + "_" + x["cytokine"], axis=1
    )
    adata.obs["is_control"] = adata.obs.apply(
        lambda x: x["cytokine"] == "PBS", axis=1
    )

    cf = CellFlow(adata, solver="otfm")
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={"cytokine_treatment": ("cytokine",)},
        perturbation_covariate_reps={"cytokine_treatment": "esm2_embeddings"},
        sample_covariates=["donor"],
        sample_covariate_reps={"donor": "donor_embeddings"},
        split_covariates=["donor"],
        max_combination_length=1,
        null_value=0.0,
    )

    layers_before_pool = {
        "cytokine_treatment": {
            "layer_type": "mlp",
            "dims": [1024, 1024],
            "dropout_rate": 0.5,
        },
        "donor": {
            "layer_type": "mlp",
            "dims": [256, 256],
            "dropout_rate": 0.0,
        },
    }
    layers_after_pool = {
        "layer_type": "mlp",
        "dims": [1024, 1024],
        "dropout_rate": 0.0,
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
        conditioning="concatenation",
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

    return cf, adata


# ---------------------------------------------------------------------------
# Training benchmark
# ---------------------------------------------------------------------------

def benchmark_training(
    cf,
    batch_size: int,
    warmup_iters: int = 5,
    bench_iters: int = 100,
    gpu_sample_interval: float = 0.005,
) -> dict:
    """Benchmark training iterations.

    Returns dict with timing and memory results.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING BENCHMARK  (batch_size={batch_size}, iters={bench_iters})")
    print(f"{'='*60}")

    # Set the actual batch size
    cf.dataloader.batch_size = batch_size
    rng_jax = jax.random.PRNGKey(42)
    rng_np = np.random.default_rng(42)
    step_fn = cf.solver.step_fn
    sampler = cf.dataloader

    # Warmup at target batch size
    print(f"  Warmup ({warmup_iters} iters at batch_size={batch_size})...")
    for _ in range(warmup_iters):
        rng_jax, rng_step = jax.random.split(rng_jax)
        batch = sampler.sample(rng_np)
        loss = step_fn(rng_step, batch)
        jax.block_until_ready(loss)

    # Measure CPU memory baseline
    proc = psutil.Process()
    cpu_mem_before_mb = proc.memory_info().rss / 1024**2

    # Measure
    print(f"  Benchmarking {bench_iters} iterations...")
    gpu_sampler = GPUMemorySampler(interval_s=gpu_sample_interval)
    gpu_baseline_mb = gpu_sampler.current_mb

    gpu_sampler.start()
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        rng_jax, rng_step = jax.random.split(rng_jax)
        batch = sampler.sample(rng_np)
        loss = step_fn(rng_step, batch)
        jax.block_until_ready(loss)
    t1 = time.perf_counter()
    gpu_peak_mb = gpu_sampler.stop()

    cpu_mem_after_mb = proc.memory_info().rss / 1024**2

    elapsed = t1 - t0
    per_iter_ms = elapsed * 1000 / bench_iters

    result = {
        "phase": "training",
        "batch_size": batch_size,
        "n_cells": None,
        "n_conditions": None,
        "iters": bench_iters,
        "total_elapsed_s": round(elapsed, 3),
        "per_iter_ms": round(per_iter_ms, 2),
        "per_10k_cells_per_cond_s": None,
        "std_s": None,
        "gpu_baseline_mb": round(gpu_baseline_mb, 1),
        "gpu_peak_mb": round(gpu_peak_mb, 1),
        "gpu_incremental_mb": round(gpu_peak_mb - gpu_baseline_mb, 1),
        "cpu_mem_mb": round(cpu_mem_after_mb, 1),
    }

    print(f"  -> {per_iter_ms:.1f} ms/iter, GPU peak {gpu_peak_mb:.0f} MB")
    return result


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------

def benchmark_inference(
    cf,
    adata,
    n_cells_list: list[int],
    warmup_runs: int = 2,
    bench_runs: int = 5,
    gpu_sample_interval: float = 0.005,
) -> list[dict]:
    """Benchmark inference (predict) for varying numbers of cells.

    Returns list of dicts with timing and memory results.
    """
    import scanpy as sc

    print(f"\n{'='*60}")
    print(f"INFERENCE BENCHMARK")
    print(f"{'='*60}")

    # Build control adata and covariate_data for one condition
    adata_ctrl = adata[adata.obs["is_control"].to_numpy()].copy()
    covariate_data_all = (
        adata[~adata.obs["is_control"].to_numpy()]
        .obs.drop_duplicates(subset=["condition"])
    )

    # Pick one condition for the single-condition benchmark
    single_cov = covariate_data_all.iloc[[0]]
    # Pick 10 conditions for multi-condition benchmark
    n_multi = min(10, len(covariate_data_all))
    multi_cov = covariate_data_all.iloc[:n_multi]

    results = []

    for n_cells in n_cells_list:
        print(f"\n  --- n_cells={n_cells} ---")

        # Subsample control cells
        if n_cells >= adata_ctrl.n_obs:
            adata_sub = adata_ctrl.copy()
            actual_n_cells = adata_ctrl.n_obs
            print(f"  (requested {n_cells}, using all {actual_n_cells} control cells)")
        else:
            adata_sub = adata_ctrl.copy()
            sc.pp.subsample(adata_sub, n_obs=n_cells)
            actual_n_cells = n_cells

        for label, cov_data in [("1_condition", single_cov), (f"{n_multi}_conditions", multi_cov)]:
            n_conds = len(cov_data)
            print(f"    {label} ({n_conds} conditions, {actual_n_cells} cells)...")

            # Warmup
            for _ in range(warmup_runs):
                _ = cf.predict(
                    adata=adata_sub,
                    sample_rep="X_pca",
                    condition_id_key="condition",
                    covariate_data=cov_data,
                )

            # Measure CPU memory baseline
            proc = psutil.Process()
            cpu_mem_before_mb = proc.memory_info().rss / 1024**2

            # Benchmark
            gpu_sampler = GPUMemorySampler(interval_s=gpu_sample_interval)
            gpu_baseline_mb = gpu_sampler.current_mb

            gpu_sampler.start()
            timings = []
            for _ in range(bench_runs):
                t0 = time.perf_counter()
                preds = cf.predict(
                    adata=adata_sub,
                    sample_rep="X_pca",
                    condition_id_key="condition",
                    covariate_data=cov_data,
                )
                # Ensure JAX computation is complete
                for v in preds.values():
                    jax.block_until_ready(v)
                t1 = time.perf_counter()
                timings.append(t1 - t0)
            gpu_peak_mb = gpu_sampler.stop()

            cpu_mem_after_mb = proc.memory_info().rss / 1024**2

            mean_s = np.mean(timings)
            std_s = np.std(timings)
            per_10k_s = mean_s * 10_000 / (actual_n_cells * n_conds)

            result = {
                "phase": f"inference_{label}",
                "batch_size": None,
                "n_cells": actual_n_cells,
                "n_conditions": n_conds,
                "iters": bench_runs,
                "total_elapsed_s": round(mean_s, 3),
                "per_iter_ms": round(mean_s * 1000, 2),
                "per_10k_cells_per_cond_s": round(per_10k_s, 3),
                "std_s": round(std_s, 4),
                "gpu_baseline_mb": round(gpu_baseline_mb, 1),
                "gpu_peak_mb": round(gpu_peak_mb, 1),
                "gpu_incremental_mb": round(gpu_peak_mb - gpu_baseline_mb, 1),
                "cpu_mem_mb": round(cpu_mem_after_mb, 1),
            }
            results.append(result)
            print(f"      -> {mean_s:.3f} s (±{std_s:.3f}), "
                  f"per 10k cells/cond: {per_10k_s:.3f} s, "
                  f"GPU peak {gpu_peak_mb:.0f} MB")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CellFlow unified benchmark")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Training batch size (default: 1024)")
    parser.add_argument("--train_iters", type=int, default=100,
                        help="Number of training iterations to benchmark (default: 100)")
    parser.add_argument("--n_cells", type=int, nargs="+",
                        default=[1000, 5000, 10000, 50000],
                        help="Number of cells for inference benchmark")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training benchmark")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference benchmark")
    parser.add_argument("--out_dir", type=str,
                        default="/lustre/groups/ml01/workspace/ot_perturbation/results/profiling",
                        help="Output directory for results")
    parser.add_argument("--out_prefix", type=str, default="benchmark",
                        help="Prefix for output files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect hardware info
    hw_info = get_hardware_info()
    print("Hardware info:")
    for k, v in hw_info.items():
        print(f"  {k}: {v}")

    # Setup
    print("\nSetting up CellFlow model...")
    cf, adata = setup_cellflow(batch_size=args.batch_size)

    all_results = []

    # Initial compilation run (needed for both training and inference benchmarks)
    print("\nCompiling model (2 warmup iterations)...")
    cf.train(num_iterations=2, batch_size=1, callbacks=[], valid_freq=100_000)

    # Count parameters (after train initializes the solver state)
    n_params = count_parameters(cf.solver.vf_state.params)
    print(f"Model parameters: {n_params:,}")
    hw_info["n_parameters"] = n_params

    # Training benchmark
    if not args.skip_training:
        train_result = benchmark_training(
            cf,
            batch_size=args.batch_size,
            bench_iters=args.train_iters,
        )
        all_results.append(train_result)

        # Additional batch sizes for training scaling
        for bs in [64, 256, 1024, 4096, 16384]:
            if bs == args.batch_size:
                continue
            try:
                r = benchmark_training(cf, batch_size=bs, bench_iters=args.train_iters)
                all_results.append(r)
            except Exception as e:
                print(f"  Batch size {bs} failed: {e}")

    # Inference benchmark
    if not args.skip_inference:
        inf_results = benchmark_inference(
            cf, adata, n_cells_list=args.n_cells,
        )
        all_results.extend(inf_results)

    # Assemble output
    df = pd.DataFrame(all_results)

    # Add hardware metadata columns
    for k, v in hw_info.items():
        df[k] = v

    # Save
    out_csv = os.path.join(args.out_dir, f"{args.out_prefix}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    # Also save hardware info as JSON
    out_json = os.path.join(args.out_dir, f"{args.out_prefix}_hardware.json")
    with open(out_json, "w") as f:
        json.dump(hw_info, f, indent=2, default=str)
    print(f"Hardware info saved to {out_json}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
