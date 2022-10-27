#!/bin/bash

PYTHONPATH="." python data_visualizer.py

PYTHONPATH="." python categorical_dataset_generator.py --dataset-type train &&
PYTHONPATH="." python categorical_dataset_generator.py --dataset-type eval &&
PYTHONPATH="." python categorical_dataset_generator.py --dataset-type test


mv data/cat/* ../to_aws_instance/

rm devign_test.csv devign_train.csv devign_val.csv
