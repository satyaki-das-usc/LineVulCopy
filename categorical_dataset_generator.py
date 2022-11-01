import os
import json
import pandas as pd
import numpy as np
import logging

from os.path import isdir, isfile, join
from argparse import ArgumentParser

logger = logging.getLogger(__name__)

SRC_DATA_DIR = "data/big-vul_dataset"
DST_DATA_DIR = "data/cat"

filenames = {
    "train": {
        "input": "train.csv",
        "output": "cat_train.csv"
    },
    "eval": {
        "input": "val.csv",
        "output": "cat_val.csv"
    },
    "test": {
        "input": "test.csv",
        "output": "cat_test.csv"
    },
    "full": {
        "input": "processed_data.csv",
        "output": "cat_processed_data.csv"
    }
}

def generate(dataset_type: str):
    SRC_PATH = join(SRC_DATA_DIR, filenames[dataset_type]["input"])
    
    logger.info(f"Reading from \"{SRC_PATH}\"...")

    if not isfile(SRC_PATH):
        print("Source file does not exist")
        return
    
    df = pd.read_csv(SRC_PATH)

    logger.info(f"Reading completed.")

    df.loc[df["CWE ID"].isnull(),'cwe_is_NaN'] = 1
    df.loc[df["CWE ID"].notnull(), 'cwe_is_NaN'] = 0

    df_nonan = df.query("cwe_is_NaN != 1 or target != 1")

    df_nonan.drop("cwe_is_NaN", axis=1, inplace=True)

    cwes = df_nonan["CWE ID"].tolist()

    unq_cwes = list(set(cwes))

    unq_cwes = list(filter(lambda x: isinstance(x, str), unq_cwes))

    vul_sample_counts = dict()

    for cwe in unq_cwes:
        vul_sample_counts[cwe] = len(df.query("`CWE ID` == @cwe and target == 1").index)

    for key in vul_sample_counts:
        if vul_sample_counts[key] == 0:
            unq_cwes.remove(key)
    
    unq_cwes.insert(0, "safe")

    categories = np.array(unq_cwes)

    targets = df_nonan["target"].tolist()

    cat_targets = []

    for idx, target in enumerate(targets):
        if target == 0:
            cat_targets.append("safe")
        else:
            cat_targets.append(cwes[idx])

    cat_series = pd.Series(cat_targets)

    df_nonan["target"] = pd.Categorical(cat_series, categories=categories)

    df_nonan["target"].replace(unq_cwes, list(range(len(unq_cwes))), inplace=True)

    DST_PATH = join(DST_DATA_DIR, filenames[dataset_type]["output"])

    logger.info(f"Writing to \"{DST_PATH}\"...")

    if not isfile(DST_PATH):
        with open(DST_PATH, "w"):
            pass

    df_nonan.to_csv(DST_PATH, index=False)

    logger.info(f"Writing completed.")

def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset-type", type=str, default="train")
    return arg_parser    

if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    generate(__args.dataset_type)