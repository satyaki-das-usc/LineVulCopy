import torch
import random
import pandas as pd
import numpy as np
import os

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

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
    if args["use_word_level_tokenizer"]:
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
    code_tokens = tokenizer.tokenize(str(func))[:args["block_size"]-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args["block_size"] - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label)

def set_seed(args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    if args["n_gpu"] > 0:
        torch.cuda.manual_seed_all(args["seed"])

def train(args, train_dataset, train_sampler, model, tokenizer, eval_dataset, curr_timestamp, logger):
    """ Train the model """
    # build dataloader
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args["train_batch_size"], num_workers=0)
    
    args["max_steps"] = args["epochs"] * len(train_dataloader)
    # evaluate the model per epoch
    args["save_steps"] = len(train_dataloader)
    args["warmup_steps"] = args["max_steps"] // 5
    model.to(args["device"])

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args["weight_decay"]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["adam_epsilon"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args["warmup_steps"],
                                                num_training_steps=args["max_steps"])

    # multi-gpu training
    if args["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args["epochs"])
    logger.info("  Instantaneous batch size per GPU = %d", args["train_batch_size"]//max(args["n_gpu"], 1))
    logger.info("  Total train batch size = %d",args["train_batch_size"]*args["gradient_accumulation_steps"])
    logger.info("  Gradient Accumulation steps = %d", args["gradient_accumulation_steps"])
    logger.info("  Total optimization steps = %d", args["max_steps"])
    
    global_step=0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1=0

    model.zero_grad()

    for idx in range(args["epochs"]): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids, labels) = [x.to(args["device"]) for x in batch]
            model.train()
            loss, logits = model(input_ids=inputs_ids, labels=labels)
            if args["n_gpu"] > 1:
                loss = loss.mean()
            if args["gradient_accumulation_steps"] > 1:
                loss = loss / args["gradient_accumulation_steps"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
                
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args["gradient_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args["save_steps"] == 0:
                    results = evaluate(args, model, tokenizer, eval_dataset, curr_timestamp, eval_when_training=True)    
                    
                    # Save model checkpoint
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args["output_dir"], '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args["model_name"])) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)