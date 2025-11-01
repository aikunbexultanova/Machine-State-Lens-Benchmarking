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
# python3 tsc_1/lens_ts.py --api_key ai048b7-9a61874c48 --input_file prepared_test/Running/imu_0.csv 

ACTIVITIES = ["Badminton", "Walking", "Standing", "Running"]
async def main(args):
    # Get eval labels
    # if args.gt_class is not None:
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
    
    base = Path("prepared_train")
    n = args.n
    MAX_IDX = 9
    k = min(n, MAX_IDX + 1) 

    files_nshot = []  # list of (label, path)

    for act in ACTIVITIES:
        idxs = random.sample(range(MAX_IDX + 1), k=k)
        for i in idxs:
            p = base / f"{act}_imu_{i}.csv"
            if p.exists():
                files_nshot.append((act, str(p)))
            else:
                logging.warning(f"Missing file: {p}")
                
    label_file_dict = defaultdict(list)

    for label, file_path in files_nshot:
        logging.info(f"Uploading example: ({label}) {file_path}")
        resp = client.files.local.upload(file_path)
        label_file_dict[label].append(resp["file_id"])

    label_file_dict = dict(label_file_dict) 
    logging.info({k: len(v) for k, v in label_file_dict.items()})
    
    files_nshot_json = args.eval_file.replace(".csv", ".json")
    os.makedirs(os.path.dirname(files_nshot_json), exist_ok=True)
    with open(files_nshot_json, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "n": args.n,
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
            "csv_configs" : { "timestamp_column": "timestamp",
                            "data_columns": ["a_x", "a_y", "a_z"],
                            "window_size": 64,
                            "step_size": 8 },
            "normalize_input": False
        }
    }
    response = client.lens.sessions.write(session_id, event_message)
    # print(response)
    

    # Create SSE reader
    sse_reader = client.lens.sessions.create_sse_consumer(session_id)
    # time.sleep(5) # Wait for the session to be ready
    await asyncio.sleep(2)

    # Upload input file
    upload_response = client.files.local.upload(args.input_file)

    # Set lens input and output stream
    client.lens.sessions.process_event(session_id, {
        "type": "input_stream.set",
        "event_data": {
            "stream_type": "csv_file_reader",
            "stream_config": {
                "file_id": upload_response["file_id"],
                "window_size": 64,
                "step_size": 8,
                "loop_recording": False,
                "output_format": "",
            }
        }
    })

    
    # Read lens outputs
    print("--- Read outputs ---")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!")

    try:
        for event in sse_reader.read(block=True):
            t = event.get("type")
            print(t)
            # pprint(event)
            # if t == "session.modify.result":
            #     pprint(event)
                # print(t)
            if t == "inference.result":
                # pprint(event)
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
                if args.gt_class is not None:
                    gt_class = args.gt_class
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

        if args.eval_file and args.gt_class:
            if not eval_outputs:
                logging.warning(f"No inference results for {args.input_file}; skipping write. n={args.n}, seed={args.seed}")
            else:
                df_outputs = pd.DataFrame(eval_outputs).sort_values("window_start_index")
                os.makedirs(os.path.dirname(args.eval_file), exist_ok=True)
                df_outputs.to_csv(args.eval_file, index=False)
                print(f"Eval results exported to {args.eval_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # build the parser ONCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens_id", default="lns-1d519091822706e2-bc108andqxf8b4os", type=str)
    parser.add_argument("--api_key", default="ai048b7-9a61874c48", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--api_endpoint", default="https://api.archetypeai.dev/v0.5", type=str)
    parser.add_argument("--gt_class", type=str)
    parser.add_argument("--n", type=int)
    
    parser.add_argument("--repeats", type=int, default=3, help="How many randomizations per setting")
    parser.add_argument("--seed_base", type=int, default=0, help="Base for deterministic seeds")
    
    reduced_test_file_numbers = [1, 5, 7, 9]

    for gt_class in ACTIVITIES:
        for file_number in reduced_test_file_numbers:
            input_file = f"prepared_test/{gt_class}_test_imu_{file_number}.csv"
            assert os.path.exists(input_file)
            for n in range(1, 11):  
                # base output path
                base_dir = f"tsc_1/lens_results_{n}shot"
                os.makedirs(base_dir, exist_ok=True)

                args = parser.parse_args(args=[
                    "--lens_id", "lns-1d519091822706e2-bc108andqxf8b4os",
                    "--api_key", "ai048b7-9a61874c48",
                    "--input_file", input_file,
                    "--eval_file", os.path.join(base_dir, f"{gt_class}_{file_number}.csv"),
                    "--api_endpoint", "https://api.archetypeai.dev/v0.5",
                    "--gt_class", gt_class,
                    "--n", str(n),
                    # repeats/seed_base
                ])
                
                base_eval_file = os.path.join(base_dir, f"{gt_class}_{file_number}.csv")
                for r in range(args.repeats):
                    args.seed = args.seed_base + r

                    p = Path(base_eval_file)  # tsc_1/lens_results_1shot/Walking_0.csv
                    out_csv = p.with_name(f"{p.stem}_seed{args.seed}{p.suffix}")   # Walking_0_seed3.csv
                    files_nshot = out_csv.with_suffix(".json")                         # Walking_0_seed3.json
                    
                    if out_csv.exists() and files_nshot.exists():
                    # if files_nshot.exists():
                        print(f"[skip] already have {out_csv.name} and {files_nshot.name}, n={n}")
                        continue 
                    
                    args.eval_file = str(out_csv)
                    asyncio.run(main(args))