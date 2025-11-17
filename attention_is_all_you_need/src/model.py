import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model) # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)

        # 논문상 원본 버전 
        # div_term = 10000.0**(torch.arange(0, d_model, 2, dtype=torch.float) / d_model)

        # 실제로는 수치안정을 위해 아래 div_term 사용
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 

        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, X):
        # X: (batch_size, seq_len, d_model)

        seq_len = X.shape[1]
        X = X + self.pe[:, :seq_len, :]
        return X
    
def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    '''
        Q, K, V: (batch_size, n_heads, seq_len, d_k)

        Attention(Q, K, V) = softmax(QK^T/sqrt(dk))*V
    '''

    batch_size, n_heads, seq_len, d_k = q.shape

    # batch_size와 n_head의 dimension을 합쳐 bmm하면 더 효율적
    q = q.reshape(-1, seq_len, d_k) # (batch_size * n_heads, seq_len, d_k)
    k = k.transpose(-2, -1).reshape(-1, d_k, seq_len) # (batch_size * n_heads, seq_len, d_k)
    v = v.reshape(-1, seq_len, d_k) # (batch_size * n_heads, seq_len, d_k)

    scores = torch.bmm(q, k) / math.sqrt(d_k) #  # (batch_size * n_heads, seq_len, seq_len)
    # 다시 원래 dimension으로 되돌림
    scores = scores.reshape(batch_size, n_heads, seq_len, seq_len)  # (batch_size, n_heads, seq_len, seq_len)

    if mask is not None:
        # Not valid부분을 -inf로 채움. softmax 계산시 제외  
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    
    attn = attn.reshape(-1, seq_len, seq_len)
    output = torch.matmul(attn, v)
    output = output.reshape(batch_size, n_heads, seq_len, d_k)
    return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V, O projection
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, X):
        '''
            가장 마지막 dim인 d_model을 num_heads로 쪼개기 

            X: (batch_size, seq_len, d_model)
            out: (batch_size, num_heads, seq_len, d_k)
        '''
        batch_size, seq_len, _ = X.shape
        out = X.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        out = out.permute(0, 2, 1, 3)
        return out

    def _combine_heads(self, X):
        '''
            Multihead를 다시 하나로 합침

            X: (batch_size, num_heads, seq_len, d_k)
            out: (batch_size, seq_len, d_model)
        '''
        batch_size, _, seq_len, _ = X.shape
        out = X.permute(0, 2, 1, 3)
        out = out.reshape(batch_size, seq_len, self.d_model)
        return out
    
    def forward(self, q, k, v, mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attn_output, attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)

        attn_output = self._combine_heads(attn_output)
        output = self.w_o(attn_output)

        return output, attn
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = self.fc1(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.fc2(X)
        return X

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X, src_mask=None):
        attn_out, _ = self.self_attn(X, X, X, src_mask)
        X = X + self.dropout1(attn_out)
        X = self.norm1(X)

        ffn_out = self.ffn(X)
        X = X + self.dropout2(ffn_out)
        X = self.norm2(X)

        return X
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, X, memory, tgt_mask=None, memory_mask=None):
        # Masked self attention
        attn_out, _ = self.self_attn(X, X, X, tgt_mask)
        X = X + self.dropout1(attn_out)
        X = self.norm1(X)

        # encoder-decoder attnetion
        attn_out, _ = self.enc_dec_attn(X, memory, memory, memory_mask)
        X = X + self.dropout2(attn_out)
        X = self.norm2(X)

        # FFN
        ffn_out = self.ffn(X)
        X = X + self.dropout3(ffn_out)
        X = self.norm3(X)

        return X
    

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, 
                 num_heads: int, d_ff: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model) # Vocab size -> d_model
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        '''
            src: (batch_size, seq_len)

            각 토큰은 ID로 구분. 여기서 embedding vector로 변환해야함 
        '''

        X = self.embed(src) * math.sqrt(self.d_model)
        X = self.pos_encoding(X)
        X = self.dropout(X)

        for layer in self.layers:
            X = layer(X, src_mask=src_mask)
        return X

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int,
                 num_heads: int, d_ff: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        X = self.embed(tgt) * math.sqrt(self.d_model)
        X = self.pos_encoding(X)
        X = self.dropout(X)

        for layer in self.layers:
            X = layer(X, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return X
    
class TransformerBase(nn.Module):
    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model: int = 512,
            num_layers: int = 6,
            num_heads: int = 8,
            d_ff: int = 2048,
            dropout: float = 0.1,
            max_len: int = 5000,
            pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.encoder = Encoder(src_vocab_size, d_model, num_layers,
                               num_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers,
                               num_heads, d_ff, dropout, max_len)

        # Softmax는 CrossEntropyLoss내부에서 처리되므로 여기서는 logit까지만 생성
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        '''
            Weight sharing

            encoder, encoder, pre-softmax linear transformation의 embedding layer는 공유됨
        '''
        if src_vocab_size == tgt_vocab_size:
            self.encoder.embed.weight = self.decoder.embed.weight
        self.generator.weight = self.decoder.embed.weight

    def make_src_mask(self, src):
        '''
            source sequence의 패드 토큰 부분 제외를 위한 마스크 생성
        '''
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch_size, 1, 1, seq_len)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        _, tgt_len = tgt.shape

        # Pad token를 무시하기 위한 mask
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # Forward-looking 방지용 마스크 
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & subsequent_mask
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        memory = self.encoder(src, src_mask=src_mask) # (batch_size, seq_len, d_model)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        logits = self.generator(out)
        return logits

if __name__ == '__main__':
    src = torch.tensor([[1,2,3,0,0]])
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    print(src_mask)




