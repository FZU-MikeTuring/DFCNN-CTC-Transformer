import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)

    if d_model % 2 == 0:
        pe[:, 1::2] = torch.cos(position * div_term)
    else:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])

    return pe.unsqueeze(0)


def attention(query, key, value, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, value)

    return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.d_model)

        return self.linear_out(x)


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super(FeedForwardModule, self).__init__()

        hidden_dim = d_model * expansion_factor

        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super(ConformerConvModule, self).__init__()

        assert kernel_size % 2 == 1

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return x


class ConformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        ffn_expansion_factor=4,
        conv_kernel_size=31,
        dropout=0.1,
    ):
        super(ConformerEncoderBlock, self).__init__()

        self.ffn1 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=ffn_expansion_factor,
            dropout=dropout,
        )
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.ffn2 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=ffn_expansion_factor,
            dropout=dropout,
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        x = x + 0.5 * self.ffn1(x)

        attn_input = self.self_attn_norm(x)
        attn_output = self.self_attention(
            attn_input,
            attn_input,
            attn_input,
            mask=src_mask,
        )
        x = x + self.self_attn_dropout(attn_output)

        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)

        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        ffn_expansion_factor=4,
        conv_kernel_size=31,
        dropout=0.1,
    ):
        super(ConformerEncoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                ConformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)

        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        attn_output = self.self_attention(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        cross_attn_output = self.cross_attention(
            x,
            enc_output,
            enc_output,
            mask=src_mask,
        )
        x = self.norm2(x + self.dropout(cross_attn_output))

        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, src_mask=src_mask)

        return x


class ConformerTransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        num_encoder_layers=4,
        num_decoder_layers=4,
        d_model=256,
        num_heads=4,
        dropout=0.1,
        max_len=5000,
        pad_id=0,
        conv_kernel_size=31,
        ffn_expansion_factor=4,
    ):
        super(ConformerTransformerModel, self).__init__()

        self.d_model = d_model
        self.pad_id = pad_id

        self.src_embedding = nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )
        self.tgt_embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

        self.register_buffer(
            "src_positional_encoding",
            positional_encoding(max_len, d_model),
        )
        self.register_buffer(
            "tgt_positional_encoding",
            positional_encoding(max_len, d_model),
        )

        self.encoder = ConformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        _, tgt_len = tgt.size()

        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        return pad_mask & causal_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        src_len = src.size(1)
        tgt_len = tgt.size(1)

        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        src_emb = src_emb + self.src_positional_encoding[:, :src_len, :]
        tgt_emb = tgt_emb + self.tgt_positional_encoding[:, :tgt_len, :]

        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)

        enc_output = self.encoder(src_emb, src_mask=src_mask)
        dec_output = self.decoder(
            tgt_emb,
            enc_output,
            tgt_mask=tgt_mask,
            src_mask=src_mask,
        )

        return self.output_layer(dec_output)
