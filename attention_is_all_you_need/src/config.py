import torch

class Config:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        self.train_dataset_path = './data/processed/wmt14_train_dataset.pt'
        self.valid_dataset_path = './data/processed/wmt14_valid_dataset.pt'

        # 영어, 독일어 데이터를 합쳐서 Bype-Pair Encoding을 진행하기에 src, tgt vocab size가 동일
        self.src_vocab_size = 58101
        self.tgt_vocab_size = 58101
        self.pad_idx = 0
        
        self.d_model = 512
        self.num_layers = 6
        self.num_heads = 8
        self.d_ff = 2048
        self.dropout = 0.1

        self.batch_size = 32
        self.lr = 1e-4
        self.warmup_steps = 4000
        self.num_epochs = 10
