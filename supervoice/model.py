import torch
from torch.nn import functional as F
from .transformer import Transformer

class SupervoiceGPT(torch.nn.Module):
    def __init__(self, config):
        super(SupervoiceGPT, self).__init__()
        self.config = config
        self.n_special_tokens = 9
        self.n_tokens = self.n_special_tokens + len(config.tokenizer.phonemes) + len(config.tokenizer.input_tokens)

        # Input embedding
        self.input_embedding = torch.nn.Embedding(self.n_tokens, self.config.gpt.n_dim)
        torch.nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)

        # Transformer
        self.transformer = Transformer(
            n_heads = self.config.gpt.n_heads,
            n_layers = self.config.gpt.n_layers,
            n_dim = self.config.gpt.n_dim,
            n_dim_head = self.config.gpt.n_dim_head,
            n_dim_ffn = self.config.gpt.n_dim_ffn,
            n_non_bias_tokens = 0,
            enable_skip_connections = False,
            casual = True,
            att_dropout = 0,
            ffn_dropout = 0.1
        )

        # Prediction head
        self.prediction_head = torch.nn.Linear(self.config.gpt.n_dim, self.n_tokens, bias=False)

        # Weight sharing
        self.input_embedding.weight = self.prediction_head.weight

    def forward(self, x, target = None):

        # Input embeddings
        x = self.input_embedding(x)

        # Run a transformer
        x = self.transformer(x)

        # Run prediction head
        x = self.prediction_head(x)

        # If target is not None, compute loss
        if target is not None:
            # Compute loss over flatten batches and sequences
            loss = F.cross_entropy(x.view(-1, x.size(-1)), target.view(-1), ignore_index = 0) # Zero token is a padding token
            return x, loss

        return x