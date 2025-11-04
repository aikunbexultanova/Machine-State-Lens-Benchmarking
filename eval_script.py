import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import re
import json
import os

FNAME_RE = re.compile(r'^(?P<activity>\w+)_(?P<file_number>\d+)_seed(?P<seed>\d+)\.csv$')

def save_metadata(metadata: dict, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(metadata, f, indent=2)


def compute_metrics(gt, pred):
    acc = accuracy_score(gt, pred)
    f1  = f1_score(gt, pred, average="macro")
    return acc, f1


def gather_one_dir(shot_dir: Path, shot: int, labels: list):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="Heartbeat", type=str)
    parser.add_argument("--base_output_path", type=str, default="out")
    parser.add_argument("--commit", type=str, default="ace1946e6")
    
    args = parser.parse_args()

    with open(f"data_processed/{args.dataset_name}/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
         
    metadata = {"commit": args.commit}  
    save_metadata(metadata, f"{args.base_output_path}/metrics/metadata.json")        
        
    per_seed_rows = []
    for shot in range(1, config["N_max"]):  # lens_results_1shot ... lens_results_10shot
        shot_dir = Path(f"{args.base_output_path}/lens_results_{shot}shot")
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

    out_metrics = Path(f"{args.base_output_path}/metrics")
    out_metrics.mkdir(exist_ok=True, parents=True)
    df_per_seed.to_csv(out_metrics / "metrics_all_seeds.csv", index=False)
    df_per_file.to_csv(out_metrics / "metrics_all_files.csv", index=False)
    df_shot_activity.to_csv(out_metrics / "metrics_seedavg_per_shot_activity.csv", index=False)
    by_shot.to_csv(out_metrics / "metrics_shot.csv", index=False)
    by_activity.to_csv(out_metrics / "metrics_activity.csv", index=False)