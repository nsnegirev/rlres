import torch
import torch.nn as nn
from transformers import AutoTokenizer
import copy
import math
import os
import pandas as pd

class PositionalEncoding(nn.Module):
    """ Implements the sinusoidal positional encoding.
    """

    def __init__(self, hid_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.hid_dim = hid_dim
        self.max_len = max_len

        # compute positional encodings
        pe = torch.zeros(max_len, hid_dim)  # (max_len, hid_dim)
        
        pe.require_grad = False
        
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, hid_dim, 2).float() * (-math.log(10000.0) / hid_dim))  # (hid_dim,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, hid_dim)
        pe = pe.unsqueeze(0)       #(1, max_len, hid_dim)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x: (batch_size, seq_len, hid_dim)
        """
        
        return self.pe[:, :x.size(1)] # (1, seq_len, hid_dim)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, num_embeddings, embedding_dim, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.embedding_dim=embedding_dim
        self.token = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0)
        self.position = PositionalEncoding(hid_dim=self.token.embedding_dim, max_len=512)
        self.segment = nn.Embedding(num_embeddings=3, embedding_dim=embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0, "hid_dim should be a multiple of n_heads"

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        Q = self.fc_q(query)  #Q = [batch size, query len, hid dim]
        K = self.fc_k(key)    #K = [batch size, key len, hid dim]
        V = self.fc_v(value)  #V = [batch size, value len, hid dim]

        #head split
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  #Q = [batch size, n heads, query len, head dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  #K = [batch size, n heads, key len, head dim]
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  #V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale        #energy = [batch size, n heads, query len, key len]

        if mask is not None:
            mask = mask[:, None, None, :]
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim = -1)        #attention = [batch size, n heads, query len, key len]
        x = torch.matmul(self.dropout(attention), V)        #x = [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()        #x = [batch size, query len, n heads, head dim]

        #head join
        x = x.view(batch_size, -1, self.hid_dim)        #x = [batch size, query len, hid dim]

        x = self.fc_o(x)        #x = [batch size, query len, hid dim]

        return x, attention

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, device):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super().__init__()
        
        self.attention_layer = MultiHeadAttention(hid_dim=hidden, n_heads=attn_heads, dropout=dropout, device=device)
        self.attention_layer_norm = nn.LayerNorm(hidden, eps=1e-6)
        
        self.feed_forward_layer = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.feed_forward_layer_norm = nn.LayerNorm(hidden, eps=1e-6)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # apply self-attention
        x1, attention = self.attention_layer(x, x, x, mask)  # (batch_size, source_seq_len, d_model)

        # apply residual connection followed by layer normalization
        x = self.attention_layer_norm(x + self.dropout(x1))  # (batch_size, source_seq_len, d_model)
        
        # apply position-wise feed-forward
        x1 = self.feed_forward_layer(x)  # (batch_size, source_seq_len, d_model)

        # apply residual connection followed by layer normalization
        x = self.feed_forward_layer_norm(x + self.dropout(x1))  # (batch_size, source_seq_len, d_model)
        
        return x, attention


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, vocab_size=14000, hidden=768, n_layers=3, attn_heads=8, dropout=0.1,  device='cpu'):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(num_embeddings=vocab_size, embedding_dim=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, device) for _ in range(n_layers)])

    def forward(self, x, segment_info, mask):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        #mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x, attention = transformer.forward(x, mask)

        return x, attention


class BERT_for_Classification(nn.Module):
    def __init__(self, bert=BERT(), hidden=768, n_classes=2893, device='cpu'):
        super().__init__()
        self.bert = bert
        self.n_classes = n_classes
        self.hidden = hidden

        self.classificaton = Classification(self.hidden, self.n_classes)

    def forward(self, x, segment_info, attention_mask):
        with torch.no_grad():
            bert_output, _ = self.bert(x, segment_info, attention_mask)
        #embeddings = bert_output[:, 0, :]
        embeddings = bert_output.mean(dim=1)
        #embeddings = bert_output.max(dim=1)[0]
        embeddings = torch.nn.functional.normalize(embeddings)

        x = self.classificaton(embeddings)
        return x

class Classification(nn.Module):
    def __init__(self, hidden, n_classes):
        super().__init__()
        self.linear_1 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = self.linear_1(x)
        return x


def load_model(device='cpu'):

    model_classification = BERT_for_Classification(device='cpu')
    model_classification.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Cl_on_myBert_allres_768_50_train_20240621_080807')))
    model = copy.deepcopy(model_classification)
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BBPE_tokenizer_KSR_allres'))

    return model, tokenizer