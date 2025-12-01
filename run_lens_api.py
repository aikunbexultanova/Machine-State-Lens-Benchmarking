import numpy as np
from archetypeai.api_client import ArchetypeAI
import time
from pprint import pprint
import logging
import sys
import argparse
import asyncio 
import csv
import json
import os
import pandas as pd
from pathlib import Path
import re
import random
from collections import defaultdict
from tqdm import tqdm
import sys
from preprocessing import preprocess_dataset
from eval_script import eval_results, save_config


async def main(args, config, input_file, eval_file, gt_class, n):
    eval_outputs = []
        
    client = ArchetypeAI(args.api_key, args.api_endpoint)

    lens_metadata = client.lens.get_metadata(lens_id=args.lens_id)
    assert len(lens_metadata) == 1, lens_metadata
    logging.info(f"\nlens_metadata: {lens_metadata}")

    # Create and connect session
    session_id, session_endpoint = client.lens.create_session(lens_id=args.lens_id)
    client.lens.sessions.connect(session_id=session_id, session_endpoint=session_endpoint)
    
    client.lens.sessions.process_event(session_id, {
        "type": "output_stream.set",
        "event_data": {
            "stream_type": "server_side_events_writer",
            "stream_config": {}
        }
    })
    
    # Upload n-shot example files
    print("--- Uploading n-shot examples ---")
    
    if getattr(args, "seed", None) is not None:
        random.seed(args.seed)
    
    base = Path(f"{args.dataset_output_dir}/{args.dataset_name}/prepared_train")
    N = config["N_max"]
    k = min(n, N) 

    files_nshot = []

    for label in config["labels"]:
        idxs = random.sample(range(N), k=k)
        for i in idxs:
            p = base / f"{label}_train_{i}.csv"
            if p.exists():
                files_nshot.append((label, str(p)))
            else:
                logging.warning(f"Missing file: {p}")
                
    label_file_dict = defaultdict(list)

    for label, file_path in files_nshot:
        logging.info(f"Uploading example: ({label}) {file_path}")
        resp = client.files.local.upload(file_path)
        label_file_dict[label].append(resp["file_id"])

    label_file_dict = dict(label_file_dict) 
    logging.info({k: len(v) for k, v in label_file_dict.items()})
    
    files_nshot_json = eval_file.replace(".csv", ".json")
    os.makedirs(os.path.dirname(files_nshot_json), exist_ok=True)
    with open(files_nshot_json, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "n": n,
                "files_nshot": files_nshot,
                "label_file_counts": {k: len(v) for k, v in label_file_dict.items()},
            },
            f,
            indent=2
        )
    logging.info(f"Saved files_nshot: {files_nshot_json}")

    # Set n-shot examples as session parameters
    print("--- Modify lens session parameters ---")
    event_message = {
        "type": "session.modify",
        "event_data": {
            "input_n_shot": label_file_dict,
            "csv_configs" : { "timestamp_column": config["timestamp_column"],
                            "data_columns": config["data_columns"],
                            "window_size": config["window_size"],
                            "step_size": config["step_size"]},
            "normalize_input": config["normalize_input"]
        }
    }
    response = client.lens.sessions.write(session_id, event_message)
    
    # Create SSE reader
    sse_reader = client.lens.sessions.create_sse_consumer(session_id)
    # time.sleep(5) # Wait for the session to be ready
    await asyncio.sleep(2)

    # Upload input file
    upload_response = client.files.local.upload(input_file)

    # Set lens input and output stream
    client.lens.sessions.process_event(session_id, {
        "type": "input_stream.set",
        "event_data": {
            "stream_type": "csv_file_reader",
            "stream_config": {
                "file_id": upload_response["file_id"],
                "window_size": config["window_size"],
                "step_size": config["step_size"],
                "loop_recording": False,
                "output_format": "",
            }
        }
    })

    
    # Read lens outputs
    print("--- Read outputs ---")

    try:
        for event in sse_reader.read(block=True):
            t = event.get("type")
            if t == "inference.result":  
                print(t)
                query_metadata = event["event_data"]["query_metadata"]["query_metadata"]
                read_index = query_metadata["read_index"]   # The first index in the window
                window_size = query_metadata["window_size"]
                response = event["event_data"]["response"]
                query_time_sec = event["event_data"]["query_time_sec"]
                print(f'Query metadata: {query_metadata}')
                print(f'Query time:     {query_time_sec}')
                print(f'Response:       {response}') 

                # Get the ground truth label: the state at the last time point in the window
                if gt_class is not None:
                    window_end_index = read_index + window_size - 1
                    print(f'Ground truth:   {gt_class}\n')
                    eval_outputs.append({
                        "prediction": response[0],
                        "gt_label": gt_class,
                        "prediction_index": window_end_index,
                        "window_start_index": read_index,
                        "window_size": window_size,
                        # "nshot_files":files_nshot,
                        "metrics": response[1],
                    })
                    
    finally:
        sse_reader.close()
        client.lens.sessions.destroy(session_id=session_id)

        if eval_file and gt_class:
            if not eval_outputs:
                logging.warning(f"No inference results for {input_file}; skipping write. n={n}, seed={args.seed}")
            else:
                df_outputs = pd.DataFrame(eval_outputs).sort_values("window_start_index")
                os.makedirs(os.path.dirname(eval_file), exist_ok=True)
                df_outputs.to_csv(eval_file, index=False)
                print(f"Eval results exported to {eval_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    parser = argparse.ArgumentParser()
    parser.add_argument("--lens_id", default="lns-1d519091822706e2-bc108andqxf8b4os", type=str)
    parser.add_argument("--api_key", default="ai048b7-9a61874c48", type=str)
    parser.add_argument("--api_endpoint", default="https://api.u1.archetypeai.app/v0.5", type=str)
    
    parser.add_argument("--dataset_name", default="Coffee", type=str)
    parser.add_argument("--dataset_type", default="Univariate", type=str)
    parser.add_argument("--dataset_output_dir", default="tsc_1/data_processed", type=str)
    parser.add_argument("--base_config_dir", default="tsc_1/data_configs", type=str)
    parser.add_argument("--base_output_dir", default="tsc_1/lens_results", type=str)
    parser.add_argument("--frequency", default=0.1, type=float)
    
    parser.add_argument("--repeats", type=int, default=3, help="How many randomizations per setting")
    parser.add_argument("--seed_base", type=int, default=0, help="Base for deterministic seeds")
    
    args = parser.parse_args()
    
    if not os.path.exists(f"{args.dataset_output_dir}/{args.dataset_name}"):
        preprocess_dataset(args)
    
    with open(f"{args.base_config_dir}/{args.dataset_name}_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
          
    print(args.dataset_name) 
    
    rng = random.Random(42)
    reduced_test_file_numbers = sorted(rng.sample(range(1, config["test_file_idx_max"]), min(config["test_file_idx_max"], 10)))
    print(reduced_test_file_numbers)
    
    reduced_n_shots = sorted(rng.sample(range(1, config["N_max"] + 1), min(config["N_max"], 7)))
    print(reduced_n_shots) 
    
    reduced_labels = sorted(rng.sample(config["labels"], min(len(config["labels"]), 4)))
    print(reduced_labels)

    for gt_class in reduced_labels:
        for file_number in reduced_test_file_numbers:
            input_file = f"{args.dataset_output_dir}/{args.dataset_name}/prepared_test/{gt_class}_test_{file_number}.csv"
            assert os.path.exists(input_file)
            for n in reduced_n_shots:  
                lens_output_dir = f"{args.base_output_dir}/{args.dataset_name}/lens_results_{n}shot"
                os.makedirs(lens_output_dir, exist_ok=True)
                
                eval_file = os.path.join(lens_output_dir, f"{gt_class}_{file_number}.csv")
                
                for r in range(args.repeats):
                    args.seed = args.seed_base + r

                    p = Path(eval_file)  
                    out_csv = p.with_name(f"{p.stem}_seed{args.seed}{p.suffix}")   
                    files_nshot = out_csv.with_suffix(".json")                        
                    
                    if out_csv.exists() and files_nshot.exists():
                        print(f"[skip] already have {out_csv.name} and {files_nshot.name}, n={n}")
                        continue 
                    
                    asyncio.run(main(args, config, input_file, str(out_csv), gt_class, n))
    
    eval_results(args, config)                