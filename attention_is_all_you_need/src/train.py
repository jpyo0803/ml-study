import torch
import torch.nn as nn
from config import Config
from datasets import load_train_dataset, load_valid_dataset
from torch.utils.data import DataLoader
from model import TransformerBase

cfg = Config()

DEVICE = cfg.device

SRC_VOCAB_SIZE = cfg.src_vocab_size
TGT_VOCAB_SIZE = cfg.tgt_vocab_size
PAD_IDX = cfg.pad_idx

D_MODEL = cfg.d_model
NUM_LAYERS = cfg.num_layers
NUM_HEADS = cfg.num_heads
D_FF = cfg.d_ff
DROPOUT = cfg.dropout

BATCH_SIZE = cfg.batch_size
LR = cfg.lr
WARMUP_STEPS = cfg.warmup_steps
NUM_EPOCHS = cfg.num_epochs

def train():
  train_dataset = load_train_dataset()
  valid_dataset = load_valid_dataset()

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

  model = TransformerBase(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    dropout=DROPOUT,
    pad_idx=PAD_IDX,
  ).to(DEVICE)

  # 만약 label이 Pad index를 갖는다면 무시
  criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    betas=(0.9, 0.98),
    eps=1e-9,
  )

  def get_lr(step, d_model, warmup_steps):
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

  class WarmupInverseSqrtScheduler(torch.optim.lr_scheduler.LambdaLR):
      def __init__(self, optimizer, d_model, warmup_steps):
          self.d_model = d_model
          self.warmup_steps = warmup_steps
          lr_lambda = lambda step: get_lr(step, d_model, warmup_steps) / get_lr(1, d_model, warmup_steps)
          super().__init__(optimizer, lr_lambda=lr_lambda)

  scheduler = WarmupInverseSqrtScheduler(optimizer, D_MODEL, WARMUP_STEPS)

  for epoch in range(NUM_EPOCHS):
    model.train()

    train_loss = 0.0
    train_total = 0

    for batch in train_loader:
      src = batch['src'].to(DEVICE)
      tgt_input = batch['tgt_input'].to(DEVICE)
      tgt_output = batch['tgt_output'].to(DEVICE)

      optimizer.zero_grad()
      outputs = model(src, tgt_input)

      loss = criterion(
        outputs.view(-1, TGT_VOCAB_SIZE), 
        tgt_output.view(-1),
      )

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      scheduler.step() # 매 스탭을 기준으로 learning rate 변화

      train_loss += loss.item() * src.shape[0]
      train_total += src.shape[0]
    
    model.eval()

    valid_loss = 0.0
    valid_total = 0.0

    for batch in valid_loader:
      src = batch['src'].to(DEVICE)
      tgt_input = batch['tgt_input'].to(DEVICE)
      tgt_output = batch['tgt_output'].to(DEVICE)

      outputs = model(src, tgt_input)

      loss = criterion(
        outputs.view(-1, TGT_VOCAB_SIZE), 
        tgt_output.view(-1),
      )

      valid_loss += loss.item() * src.shape[0]
      valid_total += src.shape[0]

  print(f'Epoch: {epoch + 1} / {NUM_EPOCHS}, Train loss: {train_loss / train_total: .4f}, Valid loss: {valid_loss / valid_total: .4f}')


if __name__ == '__main__':
  train()