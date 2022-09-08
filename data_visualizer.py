import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from ripser import ripser
from persim import plot_diagrams
from ripser import Rips

from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler

BLOCK_SIZE = 512
EVAL_BATCH_SIZE = 256

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
    def __init__(self, tokenizer):
        file_path = "data/big-vul_dataset/test.csv"
        self.examples = []
        df = pd.read_csv(file_path)
        funcs = df["processed_func"].tolist()
        labels = df["target"].tolist()
        for i in tqdm(range(len(funcs))):
            self.examples.append(convert_examples_to_features(funcs[i], labels[i], tokenizer, BLOCK_SIZE))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)

def convert_examples_to_features(func, label, tokenizer, block_size):
    # source
    code_tokens = tokenizer.tokenize(str(func))[:block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label)

def main():
    tokenizer_name = "microsoft/codebert-base"
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)

    test_dataset = TextDataset(tokenizer)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=EVAL_BATCH_SIZE, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_data_np = np.empty((EVAL_BATCH_SIZE, BLOCK_SIZE))
    for batch in tqdm(test_dataloader):
        (inputs_ids, labels) = [x.to(device) for x in batch]
        input_numpy = inputs_ids.cpu().numpy()
        test_data_np = np.concatenate((test_data_np, np.nan_to_num(input_numpy)), axis=0)
    
    test_data_np = np.nan_to_num(test_data_np)
    
    if np.isnan(test_data_np).any():
        print("Has NaN")
    else:
        print("Clean")
    rips = Rips()
    diagrams = rips.fit_transform(test_data_np)
    rips.plot(diagrams)

if __name__ == "__main__":
    main()