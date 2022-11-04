import torch
import random
import os
import json
import logging
import argparse
import pandas as pd
import numpy as np

from cat_linevul_model import Model

from datetime import datetime

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import auc
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SequentialSampler
from tokenizers import Tokenizer

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)


logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        

class TextDataset(Dataset):
    def __init__(self, X: pd.DataFrame, Y:torch.Tensor, tokenizer, args):
        self.examples = []
        funcs = X["processed_func"].tolist()
        labels = Y
        for i in tqdm(range(len(funcs))):
            self.examples.append(convert_examples_to_features(funcs[i], labels[i], tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)


def convert_examples_to_features(func, label, tokenizer, args):
    if args.use_word_level_tokenizer:
        encoded = tokenizer.encode(func)
        encoded = encoded.ids
        if len(encoded) > 510:
            encoded = encoded[:510]
        encoded.insert(0, 0)
        encoded.append(2)
        if len(encoded) < 512:
            padding = 512 - len(encoded)
            for _ in range(padding):
                encoded.append(1)
        source_ids = encoded
        source_tokens = []
        return InputFeatures(source_tokens, source_ids, label)
    # source
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def generate_result_df(logits, y_trues, y_preds, args):
    df = pd.read_csv(args.test_data_file)
    all_num_lines = []
    all_processed_func = df["processed_func"].tolist()
    for func in all_processed_func:
        all_num_lines.append(get_num_lines(func))
    flaw_line_indices = df["flaw_line_index"].tolist()
    all_num_flaw_lines = []
    total_flaw_lines = 0
    for indices in flaw_line_indices:
        if isinstance(indices, str):
            indices = indices.split(",")
            num_flaw_lines = len(indices)
            total_flaw_lines += num_flaw_lines
        else:
            num_flaw_lines = 0
        all_num_flaw_lines.append(num_flaw_lines)
    assert len(logits) == len(y_trues) == len(y_preds) == len(all_num_flaw_lines)
    return pd.DataFrame({"logits": logits, "y_trues": y_trues, "y_preds": y_preds, 
                         "index": list(range(len(logits))), "num_flaw_lines": all_num_flaw_lines, "num_lines": all_num_lines, 
                         "flaw_line": df["flaw_line"], "processed_func": df["processed_func"]})

def write_raw_preds_csv(args, y_preds):
    df = pd.read_csv(args.test_data_file)
    df["raw_preds"] = y_preds
    df.to_csv("./results/raw_preds.csv", index=False)

def get_num_lines(func):
    func = func.split("\n")
    func = [line for line in func if len(line) > 0]
    return len(func)

def train(args, train_dataset, train_sampler, class_weights, model, tokenizer, eval_dataset, curr_timestamp):
    """ Train the model """
    # build dataloader
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)
    # evaluate the model per epoch
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step=0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1=0

    model.zero_grad()

    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids, labels) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits = model(input_ids=inputs_ids, labels=labels, class_weights=class_weights)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
                
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_dataset, curr_timestamp, eval_when_training=True)    
                    
                    # Save model checkpoint
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, eval_dataset, class_weights, curr_timestamp, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]
    y_trues=[]
    for batch in eval_dataloader:
        (inputs_ids, labels)=[x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels, class_weights=class_weights)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    
    #calculate scores
    logits = np.concatenate(logits,0)
    y_trues = np.concatenate(y_trues,0)
    best_threshold = 0.5
    best_f1 = 0
    y_preds = logits
    recall = recall_score(y_trues, y_preds, average="micro")
    precision = precision_score(y_trues, y_preds, average="micro")
    f1 = f1_score(y_trues, y_preds, average="micro")
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,
    }

    STATS_JSON_PATH = f"./results/eval_{curr_timestamp}.json"

    if not os.path.isfile(STATS_JSON_PATH):
        os.system(f"touch {STATS_JSON_PATH}")
    
    with open(STATS_JSON_PATH, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, tokenizer, test_dataset, class_weights, curr_timestamp):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Write raw predictions = %d", args.write_raw_preds)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in tqdm(test_dataloader):
        (inputs_ids, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels, class_weights=class_weights)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds, average="micro")
    precision = precision_score(y_trues, y_preds, average="micro")
    f1 = f1_score(y_trues, y_preds, average="micro")
    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold":0.5,
    }

    STATS_JSON_PATH = f"./results/test_{curr_timestamp}.json"

    if not os.path.isfile(STATS_JSON_PATH):
        os.system(f"touch {STATS_JSON_PATH}")
    
    with open(STATS_JSON_PATH, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    logits = logits
    result_df = generate_result_df(logits, y_trues, y_preds, args)

    RESULTS_FILEPATH = "./results/results.csv"
            
    if not os.path.isfile(RESULTS_FILEPATH):
        os.system(f"touch {RESULTS_FILEPATH}")
    
    result_df.to_csv(RESULTS_FILEPATH)
    
    # write raw predictions if needed
    if args.write_raw_preds:
        write_raw_preds_csv(args, y_preds)

def get_cat_parameters(args):
    train_df = pd.read_csv(args.train_data_file)
    classes = train_df["target"].tolist()

    torch_classes = torch.tensor(classes)

    class_distribution_dict = dict()

    unq_train_tgs = set(classes)

    for tg in unq_train_tgs:
        class_distribution_dict[f"label_{tg}"] = classes.count(tg)
    
    class_count = [i for i in class_distribution_dict.values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)

    class_weights_all = class_weights[torch_classes]

    weighted_smapler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all), replacement=True)

    NUM_CLASSES = len(class_distribution_dict)

    return NUM_CLASSES, weighted_smapler, class_weights, train_df, torch_classes

def main():
    now = datetime.now()
    curr_timestamp = now.strftime("%m_%d_%Y_%H_%M_%S")
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ") 
    parser.add_argument("--reasoning_method", default="all", type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    # RQ2
    parser.add_argument("--effort_at_top_k", default=0.2, type=float,
                        help="Effort@TopK%Recall: effort at catching top k percent of vulnerable lines")
    parser.add_argument("--top_k_recall_by_lines", default=0.01, type=float,
                        help="Recall@TopK percent, sorted by line scores")
    parser.add_argument("--top_k_recall_by_pred_prob", default=0.2, type=float,
                        help="Recall@TopK percent, sorted by prediction probabilities")

    parser.add_argument("--do_sorting_by_line_scores", default=False, action='store_true',
                        help="Whether to do sorting by line scores.")
    parser.add_argument("--do_sorting_by_pred_prob", default=False, action='store_true',
                        help="Whether to do sorting by prediction probabilities.")
    # RQ3 - line-level evaluation
    parser.add_argument('--top_k_constant', type=int, default=10,
                        help="Top-K Accuracy constant")
    # num of attention heads
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help="number of attention heads used in CodeBERT")
    # raw predictions
    parser.add_argument("--write_raw_preds", default=False, action='store_true',
                            help="Whether to write raw predictions on test data.")
    # word-level tokenizer
    parser.add_argument("--use_word_level_tokenizer", default=False, action='store_true',
                        help="Whether to use word-level tokenizer.")
    # bpe non-pretrained tokenizer
    parser.add_argument("--use_non_pretrained_tokenizer", default=False, action='store_true',
                        help="Whether to use non-pretrained bpe tokenizer.")
    args = parser.parse_args()
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)
    # Set seed
    set_seed(args)
    
    # Set sampler and class

    num_classes, sampler, class_weights, train_df, train_target_torch = get_cat_parameters(args)

    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = num_classes
    config.num_attention_heads = args.num_attention_heads

    if args.use_word_level_tokenizer:
        print('using wordlevel tokenizer!')
        tokenizer = Tokenizer.from_file('./word_level_tokenizer/wordlevel.json')
    elif args.use_non_pretrained_tokenizer:
        tokenizer = RobertaTokenizer(vocab_file="bpe_tokenizer/bpe_tokenizer-vocab.json",
                                     merges_file="bpe_tokenizer/bpe_tokenizer-merges.txt")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    if args.use_non_pretrained_model:
        model = RobertaForSequenceClassification(config=config)        
    else:
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)
    
    model = Model(model, config, tokenizer, args)
    
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = TextDataset(train_df, train_target_torch, tokenizer, args)

        eval_df = pd.read_csv(args.eval_data_file)
        eval_targets = eval_df["target"].tolist()
        eval_target_torch = torch.tensor(eval_targets)
        eval_dataset = TextDataset(eval_df, eval_target_torch, tokenizer, args)

        train(args, train_dataset, sampler, class_weights, model, tokenizer, eval_dataset, curr_timestamp)
    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = f'checkpoint-best-f1/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer, eval_dataset, class_weights, curr_timestamp)
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)

        test_df = pd.read_csv(args.test_data_file)
        test_targets = eval_df["target"].tolist()
        test_target_torch = torch.tensor(test_targets)

        test_dataset = TextDataset(test_df, test_target_torch, tokenizer, args)
        test(args, model, tokenizer, test_dataset, class_weights, curr_timestamp)
    return results

if __name__ == "__main__":
    main()