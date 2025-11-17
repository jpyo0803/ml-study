import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import os
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
PROCESSED_DIR = "data/processed/"
MAX_SAMPLES = 40000 # 실제로는 대략적으로 4M개의 train sample 존재 

def split_src_tgt_pair(dataset):
    src_list = []
    tgt_list = []

    for pair in tqdm(dataset, desc="Splitting EN-DE pairs"):
        src_list.append(pair['en'])
        tgt_list.append(pair['de'])

    return src_list, tgt_list

def preprocess():
    dataset = load_dataset('wmt14', 'de-en', cache_dir="./data/raw")
    
    train_dataset = dataset['train'].shuffle(seed=42).select(range(MAX_SAMPLES))['translation']
    valid_dataset = dataset['validation']['translation']

    train_src, train_tgt = split_src_tgt_pair(train_dataset)

    train_src_tok = tokenizer(train_src, truncation=True, padding='max_length', max_length=128)
    train_tgt_tok = tokenizer(train_tgt, truncation=True, padding='max_length', max_length=128)
    
    valid_src, valid_tgt = split_src_tgt_pair(valid_dataset)
    
    valid_src_tok = tokenizer(valid_src, truncation=True, padding='max_length', max_length=128)
    valid_tgt_tok = tokenizer(valid_tgt, truncation=True, padding='max_length', max_length=128)

    # Store train dataset in torch tensor
    torch.save(
        {
            'src': torch.tensor(train_src_tok['input_ids'], dtype=torch.long),
            'tgt_input': torch.tensor(train_tgt_tok['input_ids'], dtype=torch.long)[:-1],
            'tgt_output': torch.tensor(train_tgt_tok['input_ids'], dtype=torch.long)[1:],
        },
        os.path.join(PROCESSED_DIR, 'wmt14_train_dataset.pt')
    )
    torch.save(
        {
            'src': torch.tensor(valid_src_tok['input_ids'], dtype=torch.long),
            'tgt_input': torch.tensor(valid_tgt_tok['input_ids'], dtype=torch.long)[:-1],
            'tgt_output': torch.tensor(valid_tgt_tok['input_ids'], dtype=torch.long)[1:],
        },
        os.path.join(PROCESSED_DIR, 'wmt14_valid_dataset.pt')
    )

if __name__ == '__main__':
    preprocess()