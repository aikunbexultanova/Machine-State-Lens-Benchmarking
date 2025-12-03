import os
import re
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import argparse
import sys
sys.path.insert(0, "tsc_1/")
import random

from atai.training.engines.engine_omega_encoder import OmegaEncoderInferenceEngine
from utils import get_csv_chunks, get_timepoint_mask
from preprocessing import preprocess_dataset
from eval_script import eval_results

# source atai_core/common/atai_py/.venv/bin/activate


def compute_embeddings(args, train_or_test, config, omega_engine, normalize_embeds=False):
    window_size = config["window_size"]
    step_size = config["step_size"]
    base = Path(f"{args.dataset_output_dir}/{args.dataset_name}/prepared_{train_or_test}")
    files_for_emb = []
    for label in config["labels"]:
        if train_or_test =="train":
            i_max = config['N_max']
        else:
            i_max = config["test_file_idx_max"]
        for i in range(i_max):
            p = base / f"{label}_{train_or_test}_{i}.csv"
            if p.exists():
                files_for_emb.append((label, str(p)))
            else:
                logging.warning(f"Missing file: {p}")
                
    emb_list: List[np.ndarray] = []
    label_list: List[str] = []
    meta_list: List[Dict[str, Any]] = []

    for label, file_path in files_for_emb:
        logging.info(f"Processing {file_path}")

        chunks = get_csv_chunks(
            file_path,
            data_columns=config["data_columns"],
            time_variate=config["timestamp_column"],
            label_columns=[],  
            window_size=window_size,
            step_size=step_size,
        )

        for chunk_idx, chunk in enumerate(chunks):
            if chunk.isna().any().any():
                continue

            input_data = []
            for col in config["data_columns"]:
                ts = chunk[col].tolist()
                padded_data, timepoint_mask = get_timepoint_mask(ts)
                sample = {"timeseries": padded_data, "timepoint_mask": timepoint_mask}
                input_data.append(sample)

            output = omega_engine.generate(input_data, normalize=config["normalize_input"])
            tokens = output["embeddings"].embeds.cpu()   # [n_channels, n_tokens, d_model]

            cls_tokens = tokens[:, -1, :]                # [n_channels, d_model]

            if normalize_embeds:
                cls_tokens = F.normalize(cls_tokens, dim=-1)

            cls_tokens_np = cls_tokens.numpy()
            emb_vec = cls_tokens_np.reshape(-1)          # flatten channels -> 1D

            emb_list.append(emb_vec)
            label_list.append(label)

            meta_list.append({
                "file": str(file_path),
                "split": train_or_test,
                "window_index": chunk_idx,
                "window_start_index": chunk_idx * step_size,
                "window_size": window_size,
            })

    embeddings = np.stack(emb_list, axis=0)
    labels = np.array(label_list)
    logging.info(
        f"Embeddings {train_or_test} generated: {embeddings.shape[0]} windows, dim={embeddings.shape[1]}"
    )
    return embeddings, labels, meta_list


def select_nshot_train_subset(train_emb, train_labels, train_meta, config, args, n, seed):

    base_train = Path(f"{args.dataset_output_dir}/{args.dataset_name}/prepared_train")
    N = config["N_max"]
    k = min(n, N)

    rng_local = random.Random(seed)

    files_nshot = []

    for label in config["labels"]:
        idxs = rng_local.sample(range(N), k=k)
        for i in idxs:
            p = base_train / f"{label}_train_{i}.csv"
            if p.exists():
                files_nshot.append((label, str(p)))
            else:
                logging.warning(f"[n-shot] Missing file: {p}")

    chosen_files = {fp for _, fp in files_nshot}
    mask = np.array([m["file"] in chosen_files for m in train_meta])

    sub_train_emb = train_emb[mask]
    sub_train_labels = train_labels[mask]

    logging.info(
        f"[n-shot] n={n}, seed={seed}: selected {len(chosen_files)} files "
        f"-> {sub_train_emb.shape[0]} windows for KNN"
    )

    return sub_train_emb, sub_train_labels, files_nshot


def run_offline_inference_for_file(args, config, train_emb, train_labels, train_meta, test_emb, test_meta,
                                    gt_class, file_number, n, seed):
    
    lens_output_dir = f"{args.base_output_dir}/{args.dataset_name}/lens_results_{n}shot"
    os.makedirs(lens_output_dir, exist_ok=True)
    eval_file = os.path.join(lens_output_dir, f"{gt_class}_{file_number}.csv")

    p = Path(eval_file)
    out_csv = p.with_name(f"{p.stem}_seed{seed}{p.suffix}")
    files_nshot_json = out_csv.with_suffix(".json")

    if out_csv.exists() and files_nshot_json.exists():
        logging.info(f"[skip] already have {out_csv.name} and {files_nshot_json.name}, n={n}, seed={seed}")
        return
    
    input_file = f"{args.dataset_output_dir}/{args.dataset_name}/prepared_test/{gt_class}_test_{file_number}.csv"
    assert os.path.exists(input_file), f"Test file not found: {input_file}"

    sub_train_emb, sub_train_labels, files_nshot = select_nshot_train_subset(train_emb, train_labels, train_meta, config, args, n, seed)

    if sub_train_emb.shape[0] == 0:
        logging.warning(f"[skip] No n-shot windows available for n={n}, seed={seed}")
        return

    n_neighbors = min(args.knn_k, sub_train_emb.shape[0])
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean", weights="uniform", algorithm="brute")
    knn.fit(sub_train_emb, sub_train_labels)
    logging.info(f"[KNN] Fitted KNN with {sub_train_emb.shape[0]} samples, "
                f"n_neighbors={n_neighbors}, classes={np.unique(sub_train_labels)}")

    test_mask = np.array([m["file"] == input_file for m in test_meta])
    X_test = test_emb[test_mask]
    test_meta_subset = [m for m in test_meta if m["file"] == input_file]

    if X_test.shape[0] == 0:
        logging.warning(f"[skip] No test windows for file {input_file}")
        return

    logging.info(f"[inference] {gt_class}, file {file_number}, n={n}, seed={seed}: "
                f"{X_test.shape[0]} windows to classify")

    eval_outputs: List[Dict[str, Any]] = []
    for emb_vec, meta in zip(X_test, test_meta_subset):
        pred = knn.predict([emb_vec])[0]
        distances, neighbor_indices = knn.kneighbors([emb_vec])
        neighbor_labels = sub_train_labels[neighbor_indices[0]]
                
        class_counts = {str(lbl): 0 for lbl in np.unique(sub_train_labels)}
        for lbl in neighbor_labels:
            class_counts[str(lbl)] += 1
        
        window_start_index = meta["window_start_index"]
        window_size = meta["window_size"]
        window_end_index = window_start_index + window_size - 1

        eval_outputs.append({
            "prediction": pred,
            "gt_label": gt_class,
            "prediction_index": window_end_index,
            "window_start_index": window_start_index,
            "window_size": window_size,
            "metrics": class_counts,
        })

    if not eval_outputs:
        logging.warning(f"[skip] No eval outputs for {input_file}; n={n}, seed={seed}")
        return

    df_outputs = pd.DataFrame(eval_outputs).sort_values("window_start_index")
    df_outputs.to_csv(out_csv, index=False)
    logging.info(f"[inference] Eval results exported to {out_csv}")

    with open(files_nshot_json, "w") as f:
        json.dump(
            {
                "seed": seed,
                "n": n,
                "files_nshot": files_nshot,
                "label_file_counts": {
                    lbl: sum(1 for l, _ in files_nshot if l == lbl)
                    for lbl in config["labels"]
                },
            },
            f,
            indent=2,
        )
    logging.info(f"[inference] Saved files_nshot: {files_nshot_json}")
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    parser = argparse.ArgumentParser()
    parser.add_argument("--omega_config", default="atai_core/configs/demos/newton_demo/inference_omega.yml")
    parser.add_argument("--dataset_name", default="BasicMotions", type=str)
    parser.add_argument("--dataset_type", default="Multivariate", type=str)
    parser.add_argument("--dataset_output_dir", default="tsc_1/data_processed", type=str)
    parser.add_argument("--base_config_dir", default="tsc_1/data_configs", type=str)
    parser.add_argument("--base_output_dir", default="tsc_1/lens_results", type=str)
    parser.add_argument("--frequency", default=0.1, type=float)
    
    parser.add_argument("--repeats", type=int, default=3, help="How many randomizations per setting")
    parser.add_argument("--seed_base", type=int, default=0, help="Base for deterministic seeds")
    parser.add_argument("--knn_k", type=int, default=5, help="k for kNN")
    
    args = parser.parse_args()

    if not os.path.exists(f"{args.dataset_output_dir}/{args.dataset_name}"):
        preprocess_dataset(args)
    
    with open(f"{args.base_config_dir}/{args.dataset_name}_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
          
    logging.info(f"{args.dataset_name}") 

    logging.info("Loading OmegaEncoderInferenceEngine...")
    omega_engine = OmegaEncoderInferenceEngine.from_config(args.omega_config, gpu_devices="auto", num_gpu_nodes=1)

    # Generate embeddings
    train_emb, train_labels, train_meta = compute_embeddings(args, "train", config, omega_engine)
    test_emb, test_labels, test_meta = compute_embeddings(args, "test", config, omega_engine)

    for gt_class in config["labels"]:
        for file_number in range(0, config["test_file_idx_max"]):
            for n in range(1, config["N_max"] + 1): 
                for r in range(args.repeats):
                    seed = args.seed_base + r
                    run_offline_inference_for_file(args, config, train_emb, train_labels, train_meta,
                                                test_emb, test_meta, gt_class, file_number, n, seed)

    eval_results(args, config)