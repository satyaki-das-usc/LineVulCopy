# PYTHONPATH="." python linevul_main.py \
#     --model_name=12heads_linevul_model.bin \
#     --output_dir=./saved_models \
#     --model_type=roberta \
#     --tokenizer_name=microsoft/codebert-base \
#     --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --test_data_file=../data/big-vul_dataset/test.csv \
#     --block_size 512 \
#     --eval_batch_size 512

# PYTHONPATH="." python linevul_main.py \
#     --model_name=12heads_linevul_model.bin \
#     --output_dir=./saved_models \
#     --model_type=roberta \
#     --tokenizer_name=microsoft/codebert-base \
#     --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --test_data_file=../data/zero_day/zero_day_vul_only.csv \
#     --block_size 512 \
#     --eval_batch_size 256 \
#     --write_raw_preds \
#     --do_sorting_by_line_scores \

# PYTHONPATH="." python linevul_main.py \
#     --model_name=12heads_linevul_model.bin \
#     --output_dir=./saved_models \
#     --model_type=roberta \
#     --tokenizer_name=microsoft/codebert-base \
#     --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --test_data_file=../data/zero_day/zero_day_vul_only.csv \
#     --block_size 512 \
#     --eval_batch_size 256 \
#     --write_raw_preds \


PYTHONPATH="." python linevul_main.py \
    --model_name=12heads_linevul_model.bin \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --test_data_file=../data/zero_day/zero_day_vul_only.csv \
    --block_size 512 \
    --eval_batch_size 256 \
    --write_raw_preds \
    --do_train \
    --train_data_file=../data/zero_day/chromium_free.csv \
    --eval_data_file=../data/zero_day/chromium.csv