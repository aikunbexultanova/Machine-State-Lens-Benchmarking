import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import re
import json
import os

FNAME_RE = re.compile(r'^(?P<activity>\w+)_(?P<file_number>\d+)_seed(?P<seed>\d+)\.csv$')

def save_config(config: dict, file_path: str):
    with open(file_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config file to: {file_path}")    


def compute_metrics(gt, pred):
    acc = accuracy_score(gt, pred)
    f1  = f1_score(gt, pred, average="macro")
    return acc, f1


def gather_one_dir(shot_dir: Path, shot: int, labels: list):
    FNAME_RE = re.compile(r'^(?P<activity>\w+)_(?P<file_number>\d+)_seed(?P<seed>\d+)\.csv$')
    rows = []
    for a in labels:
        for p in sorted((shot_dir).glob(f"{a}_*.csv")):
            m = FNAME_RE.match(p.name)
            if not m:
                continue
            activity = m.group("activity")
            file_number = int(m.group("file_number"))
            seed = int(m.group("seed"))

            df = pd.read_csv(p)

            acc, f1 = compute_metrics(df["gt_label"], df["prediction"])
            rows.append({
                "shot": shot,
                "activity": activity,
                "file_number": file_number,
                "seed": seed,
                "accuracy": acc,
                "f1": f1,
                "path": str(p),
            })
    return pd.DataFrame(rows)


def eval_results(args, config):
            
    save_config(config, f"{args.base_output_dir}/{args.dataset_name}/config.json")      
        
    per_seed_rows = []
    for shot in range(1, config["N_max"]+1):  # lens_results_1shot ... lens_results_10shot
        shot_dir = Path(f"{args.base_output_dir}/{args.dataset_name}/lens_results_{shot}shot")
        if not shot_dir.exists():
            continue
        df_dir = gather_one_dir(shot_dir, shot, config["labels"])
        per_seed_rows.append(df_dir)

    df_per_seed = pd.concat(per_seed_rows, ignore_index=True)

    df_per_file = (
        df_per_seed
        .groupby(["shot","activity","file_number"], as_index=False)[["accuracy","f1"]]
        .mean()
        .rename(columns={"accuracy":"accuracy_mean_seed", "f1":"f1_mean_seed"})
    )

    df_shot_activity = (
        df_per_file
        .groupby(["shot","activity"], as_index=False)[["accuracy_mean_seed","f1_mean_seed"]]
        .mean()
        .rename(columns={
            "accuracy_mean_seed":"accuracy",
            "f1_mean_seed":"f1"
        })
        .round(3)
    )

    by_shot = (
        df_shot_activity.groupby("shot", as_index=False)[["accuracy","f1"]]
        .mean()
        .round(3)
    )
    by_activity = (
        df_shot_activity.groupby("activity", as_index=False)[["accuracy","f1"]]
        .mean()
        .round(3)
    )

    out_metrics = Path(f"{args.base_output_dir}/{args.dataset_name}/metrics")
    out_metrics.mkdir(exist_ok=True, parents=True)
    df_per_seed.to_csv(out_metrics / "metrics_all_seeds.csv", index=False)
    df_per_file.to_csv(out_metrics / "metrics_all_files.csv", index=False)
    df_shot_activity.to_csv(out_metrics / "metrics_seedavg_per_shot_activity.csv", index=False)
    by_shot.to_csv(out_metrics / "metrics_shot.csv", index=False)
    print(by_shot)
    by_activity.to_csv(out_metrics / "metrics_activity.csv", index=False)
    
    print(f"metrics saved to {str(out_metrics)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_output_dir", default="data_processed", type=str)
    parser.add_argument("--dataset_name", default="ACSF1", type=str)
    parser.add_argument("--base_output_dir", type=str, default="tsc_1/lens_results")
    parser.add_argument("--base_config_dir", default="tsc_1/data_configs", type=str)
    
    args = parser.parse_args()
    
    with open(f"{args.base_config_dir}/{args.dataset_name}_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    # args.base_output_dir = f"{args.base_output_dir}/{args.dataset_name}" 
    eval_results(args, config)    