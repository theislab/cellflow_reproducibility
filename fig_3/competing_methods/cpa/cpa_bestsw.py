import sys
import os
import time
import subprocess as sp
import warnings
import yaml
import random
import numpy as np

warnings.filterwarnings("ignore")

OUT_DIR = (
    "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/results/runs/bestsw/"
)
CPA_SUB = "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/scripts/bnchmrk/cpa_sub.sh"
EVAL_SUB = "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/scripts/bnchmrk/bnchmrk_eval_sub.sh"


def poll_bsub_completion(job_id):
    completed = False
    while not completed:
        status_cmd = f"bjobs {job_id}"
        status_process = sp.Popen(
            status_cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE
        )
        status_output, _ = status_process.communicate()
        status_output = status_output.split(b"\n")[1]
        if not (b"PEND" in status_output or b"RUN" in status_output):
            completed = True
        return completed


if __name__ == "__main__":
    # load config
    sweep_config = sys.argv[1]
    run_prefix = sys.argv[2]
    test_combs = sys.argv[3]
    with open(sweep_config, "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)

    run_config = os.path.join(sweep_config["out_dir"], "configs", f"{run_prefix}.yml")
    with open(run_config, "r") as f:
        run_config = yaml.load(f, Loader=yaml.FullLoader)

    script = sweep_config.pop("script")
    bsub_params = {
        key.split("_")[1]: sweep_config.pop(key)
        for key in list(sweep_config.keys())
        if key.startswith("bsub")
    }

    out_prefix = run_config["out_prefix"]
    test_combs = test_combs.split(",")
    ap_mols = ["FGF8", "XAV", "RA", "CHIR"]
    dv_mols = ["SHH", "BMP4"]
    double_combs = ["+".join([a, b]) for a in ap_mols for b in dv_mols]
    triple_ap = ["RA+CHIR", "FGF8+CHIR"]
    double_combs_chir = triple_ap
    triple_dv = ["SHH", "BMP4"]
    triple_combs = ["+".join([a, b]) for a in triple_ap for b in triple_dv]
    processed_combs = []
    left_combs = test_combs
    cmds_cpa = []
    cmds_eval = []
    for test_comb in test_combs:
        exclude_combs = []
        if test_comb in processed_combs:
            continue

        random.seed(212)
        exclude_combs.append(test_comb)

        # standard double combinations
        if test_comb in double_combs:
            ap_mol, dv_mol = test_comb.split("+")
            ap_rest = [mol for mol in ap_mols if mol != ap_mol]
            dv_rest = [mol for mol in dv_mols if mol != dv_mol]
            dv_rest.append(dv_rest[0])
            dv_rest.append(dv_mol)
        else:
            ap_rest = ap_mols.copy()
            dv_rest = dv_mols.copy() * 2
        random.shuffle(ap_rest)
        exclude_combs.extend(["+".join([a, b]) for a, b in zip(ap_rest, dv_rest)])

        # CHIR double combinations
        if not test_comb in double_combs_chir:
            exclude_combs.append(random.choice(double_combs_chir))

        # triple combinations
        if test_comb in triple_combs:
            ap_mol, _, dv_mol = test_comb.split("+")
            ap_rest = "RA" if ap_mol == "FGF8" else "FGF8"
            dv_rest = "SHH" if dv_mol == "BMP4" else "BMP4"
            exclude_combs.append(f"{ap_rest}+CHIR+{dv_rest}")
        elif test_comb in double_combs:
            ap_mol, dv_mol = test_comb.split("+")
            if ap_mol == "CHIR":
                exclude_combs.append(f"RA+{ap_mol}+{dv_mol}")
                exclude_combs.append(f"FGF8+{ap_mol}+{dv_mol}")
            elif ap_mol in ["RA", "FGF8"]:
                exclude_combs.append(f"{ap_mol}+CHIR+{dv_mol}")
                ap_rest = "RA" if ap_mol == "FGF8" else "FGF8"
                dv_rest = "SHH" if dv_mol == "BMP4" else "BMP4"
                exclude_combs.append(f"{ap_rest}+CHIR+{dv_rest}")
            else:
                random.shuffle(triple_ap)
                exclude_combs.extend(
                    ["+".join([a, b]) for a, b in zip(triple_ap, triple_dv)]
                )
        elif test_comb in double_combs_chir:
            ap_mol = test_comb.split("+")[0]
            exclude_combs.append(f"{ap_mol}+CHIR+SHH")
            exclude_combs.append(f"{ap_mol}+CHIR+BMP4")
        else:
            random.shuffle(triple_ap)
            exclude_combs.extend(
                ["+".join([a, b]) for a, b in zip(triple_ap, triple_dv)]
            )

        # combinations to evaluate
        eval_combs = []
        for comb in left_combs:
            if comb in exclude_combs:
                if comb in double_combs:
                    ap_mol, dv_mol = comb.split("+")
                    if ap_mol == "CHIR":
                        if not (
                            f"RA+{ap_mol}+{dv_mol}" in exclude_combs
                            and f"FGF8+{ap_mol}+{dv_mol}" in exclude_combs
                        ):
                            continue
                    elif ap_mol in ["RA", "FGF8"]:
                        ap_rest = "RA" if ap_mol == "FGF8" else "FGF8"
                        dv_rest = "SHH" if dv_mol == "BMP4" else "BMP4"
                        if not (
                            f"{ap_mol}+CHIR+{dv_mol}" in exclude_combs
                            and f"{ap_rest}+CHIR+{dv_rest}" in exclude_combs
                        ):
                            continue
                elif comb in double_combs_chir:
                    ap_mol = comb.split("+")[0]
                    if not (
                        f"{ap_mol}+CHIR+SHH" in exclude_combs
                        and f"{ap_mol}+CHIR+BMP4" in exclude_combs
                    ):
                        continue
                elif comb in triple_combs:
                    ap_mol, _, dv_mol = comb.split("+")
                    ap_rest = "RA" if ap_mol == "FGF8" else "FGF8"
                    dv_rest = "SHH" if dv_mol == "BMP4" else "BMP4"
                    if not f"{ap_rest}+CHIR+{dv_rest}" in exclude_combs:
                        continue
                eval_combs.append(comb)

        processed_combs.extend(eval_combs)
        left_combs = [comb for comb in left_combs if comb not in processed_combs]
        run_config["exclude_combs"] = exclude_combs
        run_config["eval_combs"] = eval_combs
        run_config["plot_umap_combs"] = eval_combs.copy()
        run_config["plot_heatmap_combs"] = eval_combs.copy()
        run_config["sweep_combs"] = []
        if "RA+BMP4" in eval_combs:
            run_config["plot_umap_conds"] = [
                "RA_2+BMP4_2",
                "RA_2+BMP4_3",
                "RA_3+BMP4_2",
                "RA_3+BMP4_3",
                "RA_4+BMP4_2",
                "RA_4+BMP4_3",
                "RA_3+CHIR_4+BMP4_3",
                "RA_4+CHIR_4+BMP4_3",
            ]
        else:
            run_config["plot_umap_conds"] = []

        # CPA command
        out_dir = os.path.join(OUT_DIR, out_prefix, test_comb)
        os.makedirs(out_dir, exist_ok=True)
        run_config["out_dir"] = out_dir
        run_config["out_prefix"] = test_comb
        run_config["save_adata"] = True
        run_config["save_model"] = True
        run_config["minimal"] = False
        # Overwrite h5ad in case of changed paths
        run_config["h5ad"] = sweep_config["h5ad"]
        config_path = os.path.join(out_dir, "config.yml")
        with open(config_path, "w") as f:
            f.write(yaml.dump(run_config, sort_keys=False))
        bsub_params["J"] = "cf_bestsw"
        bsub_params["o"] = os.path.join(run_config["out_dir"], "log.out")
        bsub_params["e"] = os.path.join(run_config["out_dir"], "log.err")
        cmd = [f"bash {CPA_SUB}"]
        cmd.append(config_path)
        bsub_str = " ".join(
            [f"-{key} {str(value)}" for key, value in bsub_params.items()]
        )
        cmd.append(f'"{bsub_str}"')
        cmd = " ".join(cmd)
        cmds_cpa.append(cmd)

        # evaluation command
        cmd = [f"bash {EVAL_SUB}"]
        cmd.append(config_path)
        cmd.append(f'"{bsub_str}"')
        cmd = " ".join(cmd)
        cmds_eval.append(cmd)

    job_ids = []
    for cmd in cmds_cpa:
        process = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, _ = process.communicate()
        # Extract job ID from BSUB output
        stdout = stdout.decode("utf-8")
        print(stdout)
        job_id = int(stdout.split()[1].strip("<>"))
        job_ids.append(job_id)

    done = [False] * len(job_ids)
    while not all(done):
        time.sleep(60)
        for i, job_id in enumerate(job_ids):
            if not done[i]:
                completed = poll_bsub_completion(job_id)
                if completed:
                    done[i] = True
                    os.system(cmds_eval[i])
