import os

import pandas as pd
import numpy as np

from os.path import join, isdir, isfile
from typing import cast
from omegaconf import OmegaConf, DictConfig
from math import ceil
from argparse import ArgumentParser
from tqdm import tqdm

def main(dataset_filepath: str = None, batch_size: int = None):
	BATCH_SIZE = 10000

	if batch_size is not None:
		BATCH_SIZE = batch_size
	
	df = pd.read_parquet(dataset_filepath)

	BATCHES = [df[i:i+BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]

	dataset_dir = "/".join(dataset_filepath.split("/")[:-1])

	for idx, batch_df in enumerate(BATCHES):
		parquet_filepath = join(dataset_dir, f"batch{idx + 1}.parquet.gzip")
		batch_df.to_parquet(parquet_filepath, compression="gzip")

		cmd = f"PYTHONPATH=\".\" python linevul_main.py --model_name=12heads_linevul_model.bin --output_dir=./saved_models --model_type=roberta --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --do_test --test_data_file={parquet_filepath} --is_parquet --block_size 512 --eval_batch_size 256 --write_raw_preds"
		code = os.system(cmd)
		if code != 0:
			continue

		os.system(f"rm {parquet_filepath}")

def configure_arg_parser() -> ArgumentParser:
	arg_parser = ArgumentParser()
	arg_parser.add_argument("--dataset_filepath", type=str, default=None)
	arg_parser.add_argument("--batch_size", type=int, default=None)
	return arg_parser

if __name__ == "__main__":
	__arg_parser = configure_arg_parser()
	__args = __arg_parser.parse_args()
	main(__args.dataset_filepath, __args.batch_size)
