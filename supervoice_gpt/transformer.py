import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
from torch.cuda.amp import autocast
from .tensors import RMSNorm

class Transformer(nn.Module):
    def __init__(self, 
        n_heads,
        n_layers,
        n_dim,
        n_dim_head,
        n_dim_ffn,
        att_dropout, 
        ffn_dropout,
        position_embedding = None, # or rotary or 'alibi'
        alibi_non_bias_tokens = 0,
        casual = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_non_bias_tokens = alibi_non_bias_tokens
        self.casual = casual

        # Attention blocks
        self.layers = torch.nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(AttentionBlock(
                n_heads = n_heads, 
                n_dim = n_dim, 
                n_dim_head = n_dim_head, 
                n_dim_ffn = n_dim_ffn,
                att_dropout = att_dropout,
                ffn_dropout = ffn_dropout,
                casual = casual
            ))
        
        # Output normalization
        self.output_norm = RMSNorm(n_dim)

        # Positional embedding
        self.position_embedding = position_embedding
        if position_embedding == 'alibi':
            pass
        elif position_embedding == 'rotary':
            theta = 50000
            self.register_buffer('inv_freq', 1.0 / (theta ** (torch.arange(0, n_dim_head, 2).float() / n_dim)))
        elif position_embedding is None:
            pass
        else:
            raise ValueError(f"Unknown position embedding: {position_embedding}")


    def forward(self, x, *, mask = None):
        batch, seq_len, *_ = x.shape

        # Embeddings
        alibi = None
        rotational = None

        # Compute ALiBi
        # This computes ALiBi bias mask, excluding non-bias tokens which are expected to be appended to the end of the sequence
        # Inspired by: https://github.com/ofirpress/attention_with_linear_biases/issues/5
        if self.position_embedding == 'alibi':
            alibi = get_alibi_mask(seq_len - self.n_non_bias_tokens, self.n_heads, self.casual, x.device)
            if self.n_non_bias_tokens > 0:
                alibi = torch.nn.functional.pad(alibi, (0, self.n_non_bias_tokens, 0, self.n_non_bias_tokens), value=0)

        # Compute rotary embeddings
        if self.position_embedding == 'rotary':
            t = torch.arange(seq_len, device = self.inv_freq.device, dtype = self.inv_freq.dtype)
            freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
            rotational =  torch.cat((freqs, freqs), dim = -1)

        # Run through attention blocks
        for decoder in self.layers:
            x = decoder(x, alibi = alibi, rotational = rotational, mask = mask)

        # Output normalization
        x = self.output_norm(x)

        # Result
        return x

class TransformerAdvanced(nn.Module):
    def __init__(self, 

        # Architecture
        n_heads,
        n_layers,
        n_dim,
        n_dim_head,
        n_dim_ffn,
        att_dropout, 
        ffn_dropout,

        # Positional embedding
        position_embedding = None # or rotary or 'alibi'
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Attention blocks
        self.layers_enc = torch.nn.ModuleList([])
        self.layers_dec = torch.nn.ModuleList([])
        for i in range(n_layers):
            self.layers_enc.append(AttentionBlock(
                n_heads = n_heads, 
                n_dim = n_dim, 
                n_dim_head = n_dim_head, 
                n_dim_ffn = n_dim_ffn,
                att_dropout = att_dropout,
                ffn_dropout = ffn_dropout,
                casual = False
            ))
            self.layers_dec.append(AttentionBlockAdvanced(
                n_heads = n_heads, 
                n_dim = n_dim, 
                n_dim_head = n_dim_head, 
                n_dim_ffn = n_dim_ffn,
                att_dropout = att_dropout,
                ffn_dropout = ffn_dropout
            ))
        
        # Output normalization
        self.output_norm = RMSNorm(n_dim)

        # Positional embedding
        self.position_embedding = position_embedding
        if position_embedding == 'alibi':
            pass
        elif position_embedding == 'rotary':
            theta = 50000
            self.register_buffer('inv_freq', 1.0 / (theta ** (torch.arange(0, n_dim_head, 2).float() / n_dim)))
        elif position_embedding is None:
            pass
        else:
            raise ValueError(f"Unknown position embedding: {position_embedding}")


    def forward(self, x, y, x_mask = None, y_mask = None, xy_mask = None):
        batch, seq_len_x, *_ = x.shape
        _, seq_len_y, *_ = y.shape

        # Embeddings
        alibi_x = None
        alibi_y = None
        rotational_x = None
        rotational_y = None

        # Compute ALiBi
        # This computes ALiBi bias mask, excluding non-bias tokens which are expected to be appended to the end of the sequence
        # Inspired by: https://github.com/ofirpress/attention_with_linear_biases/issues/5
        if self.position_embedding == 'alibi':
            alibi_x = get_alibi_mask(seq_len_x, self.n_heads, False, x.device)
            alibi_y = get_alibi_mask(seq_len_y, self.n_heads, False, y.device)

        # Compute rotary embeddings
        if self.position_embedding == 'rotary':
            t_x = torch.arange(seq_len_x, device = self.inv_freq.device, dtype = self.inv_freq.dtype)
            freqs_x = torch.einsum('i , j -> i j', t_x, self.inv_freq)
            rotational_x =  torch.cat((freqs_x, freqs_x), dim = -1)
            t_y = torch.arange(seq_len_y, device = self.inv_freq.device, dtype = self.inv_freq.dtype)
            freqs_y = torch.einsum('i , j -> i j', t_y, self.inv_freq)
            rotational_y =  torch.cat((freqs_y, freqs_y), dim = -1)

        # Run through encoder attention blocks
        encoder_output = x
        for encoder in self.layers_enc:
            encoder_output = encoder(encoder_output, alibi = alibi_x, rotational = rotational_x, mask = x_mask)

        # Run through decoder attention blocks
        decoder_output = y
        for decoder in self.layers_dec:
            decoder_output = decoder(

                # Self atention for decoder outputs
                x = decoder_output,
                self_alibi = alibi_y, 
                self_rotational = rotational_y, 
                self_mask = y_mask, 

                # Cross attention between encoder and decoder outputs                
                y = encoder_output,
                cross_mask = xy_mask,
                # NOTE: No positional encodings are used in cross attention
            )

        # Output normalization
        decoder_output = self.output_norm(decoder_output)

        # Result
        return decoder_output

#
# Attention Block
#

class AttentionBlock(torch.nn.Module):
    def __init__(self, n_heads, n_dim, n_dim_head, n_dim_ffn, att_dropout, ffn_dropout, casual):
        super().__init__()

        self.n_heads = n_heads
        self.n_dim_head = n_dim_head
        self.att_dropout = att_dropout
        self.casual = casual

        # Attention
        self.attention_ln = RMSNorm(n_dim)
        self.attention = Attention(n_heads, n_dim, n_dim_head, att_dropout, casual)

        # MLP part
        self.mlp_ln = RMSNorm(n_dim)
        self.mlp = AttentionMLP(n_dim, n_dim_ffn, ffn_dropout)

    def forward(self, x, alibi = None, rotational = None, mask = None):

        # Attention
        residual = x
        x = self.attention_ln(x)
        x = self.attention(x_q = x, x_k = x, x_v = x, alibi = alibi, rotational = rotational, mask = mask) + residual

        # MLP
        residual = x
        x = self.mlp_ln(x)
        x = self.mlp(x) + residual

        return x

class AttentionBlockAdvanced(torch.nn.Module):
    def __init__(self, n_heads, n_dim, n_dim_head, n_dim_ffn, att_dropout, ffn_dropout):
        super().__init__()

        self.n_heads = n_heads
        self.n_dim_head = n_dim_head
        self.att_dropout = att_dropout

        # Attention
        self.attention_self_ln = RMSNorm(n_dim)
        self.attention_self = Attention(n_heads, n_dim, n_dim_head, att_dropout, False)
        self.attention_cross_ln_x = RMSNorm(n_dim)
        self.attention_cross_ln_y = RMSNorm(n_dim)
        self.attention_cross = Attention(n_heads, n_dim, n_dim_head, att_dropout, False)

        # MLP part
        self.mlp_ln = RMSNorm(n_dim)
        self.mlp = AttentionMLP(n_dim, n_dim_ffn, ffn_dropout)

    def forward(self, x, y, self_alibi = None, cross_alibi = None, self_rotational = None, cross_rotational = None, self_mask = None, cross_mask = None):

        # Self Attention
        residual = x
        x = self.attention_self_ln(x)
        x = self.attention_self(x_q = x, x_k = x, x_v = x, alibi = self_alibi, rotational = self_rotational, mask = self_mask)
        x = x + residual

        # Cross Attention
        residual = x
        x = self.attention_cross_ln_x(x)
        y = self.attention_cross_ln_y(y)
        x = self.attention_cross(x_q = x, x_k = y, x_v = y, alibi = cross_alibi, rotational = cross_rotational, mask = cross_mask)
        x = x + residual

        # MLP
        residual = x
        x = self.mlp_ln(x)
        x = self.mlp(x)
        x = x + residual

        return x

#
# Attention Layer
#

class Attention(torch.nn.Module):
    def __init__(self, n_heads, n_dim, n_dim_head, att_dropout, casual):
        super().__init__()
        self.n_heads = n_heads
        self.n_dim = n_dim
        self.n_dim_head = n_dim_head
        self.att_dropout = att_dropout
        self.casual = casual

        # Input -> Query/Key/Value
        self.attention_q = nn.Linear(n_dim, n_dim_head * n_heads, bias=False)
        self.attention_k = nn.Linear(n_dim, n_dim_head * n_heads, bias=False)
        self.attention_v = nn.Linear(n_dim, n_dim_head * n_heads, bias=False)
        torch.nn.init.normal_(self.attention_q.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.attention_k.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.attention_v.weight, mean=0.0, std=0.02)

        # Output flatten multiple heads into single tensor
        self.attention_output = nn.Linear(n_dim_head * n_heads, n_dim, bias=False)
        torch.nn.init.normal_(self.attention_output.weight, mean=0.0, std=0.02)

    def _split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.n_dim_head).transpose(1, 2)

    def forward(self, *, x_q, x_k, x_v, alibi = None, rotational = None, mask = None):

        # Calculation Q/K/V for each head
        q = self._split_heads(self.attention_q(x_q))
        k = self._split_heads(self.attention_k(x_k))
        v = self._split_heads(self.attention_v(x_v))

        # Rotary embedding
        if rotational is not None:
            q = apply_rotary_pos_emb(rotational, q)
            k = apply_rotary_pos_emb(rotational, k)

        # Calculate mask
        target_mask = mask
        if alibi is not None:
            if mask is not None:
                target_mask = mask + alibi
            else:
                target_mask = alibi

        # Dot product attention
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
            attn_mask = target_mask, 
            dropout_p = self.att_dropout if self.training else 0.0, 
            is_causal = self.casual
        )

        # Reassemble all head outputs side by side
        B, _, T, _ = y.size()
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.n_dim_head) # re-assemble all head outputs side by side

        # Output
        y = self.attention_output(y)

        return y

class AttentionMLP(torch.nn.Module):
    def __init__(self, n_dim, n_dim_ffn, ffn_dropout):
        super().__init__()
        self.mlp_input = nn.Linear(n_dim, n_dim_ffn)
        self.mlp_output = nn.Linear(n_dim_ffn, n_dim)
        self.mlp_output_dropout = nn.Dropout(ffn_dropout)

    def forward(self, x):
        x = self.mlp_input(x)
        x = F.gelu(x)
        x = self.mlp_output(x)
        x = self.mlp_output_dropout(x)
        return x

#
# Convolutional positional embedding
#

class ConvPositionEmbed(nn.Module):
    def __init__(self, n_dim, kernel_size):
        super().__init__()
        self.dw_conv1d = nn.Sequential(nn.Conv1d(n_dim, n_dim, kernel_size, groups = n_dim, padding = kernel_size // 2), nn.GELU())

    def forward(self, x, mask = None):

        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.)

        x = rearrange(x, 'b n c -> b c n')
        x = self.dw_conv1d(x)
        out = rearrange(x, 'b c n -> b n c')

        if mask is not None:
            out = out.masked_fill(~mask, 0.)

        return out

#
# ALiBi implementation
#

slopes_cache = {}
def get_slopes_power_of_2(n_heads, device):
    global slopes_cache
    key = str(n_heads) + "_" + str(device)
    if key not in slopes_cache:
        start = (2**(-2**-(math.log2(n_heads)-3)))
        ratio = start
        slopes_cache[key] = torch.tensor([start*ratio**i for i in range(n_heads)], requires_grad=False, device = device) * -1
    return slopes_cache[key]

alibi_do_cache = False
alibi_cache = {}
def get_alibi_mask(seq_len, n_heads, casual, device):
    global alibi_cache
    key = str(seq_len) + "_" + str(n_heads) + "_" + str(casual) + "_" + str(device)

    if key not in alibi_cache:
        slopes = get_slopes_power_of_2(n_heads, device)
        context_position = torch.arange(seq_len, device = device)[:, None]
        memory_position = torch.arange(seq_len, device = device)[None, :]
        relative_position = memory_position - context_position 
        relative_position = torch.abs(relative_position).unsqueeze(0).expand(n_heads, -1,-1)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
        alibi = alibi.view(1, n_heads, seq_len, seq_len)
        if casual:
            # Make top right triangle of the matrix to be -inf
            top_triangle_mask = torch.triu(torch.ones(seq_len, seq_len, device = device), diagonal = 1).bool()
            alibi = alibi.masked_fill(top_triangle_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        if alibi_do_cache:
            alibi_cache[key] = alibi
        return alibi

    return alibi_cache[key]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

@autocast(enabled = False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()