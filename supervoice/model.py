import torch
from torch.nn import functional as F

class SupervoiceGPT(torch.nn.Module):
    def __init__(self, config):
        super(SupervoiceGPT, self).__init__()
        self.config = config
        self.n_special_tokens = 9
        self.n_input_tokens = len(config.tokenizer.input_tokens) + len(config.tokenizer.phonemes) + self.n_special_tokens
        self.n_output_tokens = len(config.tokenizer.phonemes) + self.n_special_tokens

        # Input embedding
        self.input_embedding = torch.nn.Embedding(self.n_input_tokens, self.config.gpt.n_dim)

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
        # NOTE: No weight tying since we have a very different token sets for input and output
        self.prediction_head = torch.nn.Linear(self.config.n_dim, self.n_output_tokens)

        # Init weights
        torch.nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.prediction_head.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.prediction_head.bias)

    def forward(self, x, target = None):

        # Input embeddings
        x = self.input_embedding(x)

        # Run a transformer
        x = self.transformer(x)

        # Run prediction head
        x = self.prediction_head(x)

        # If target is not None, compute loss
        if target is not None:
            # Compute loss
            loss = F.cross_entropy(x, target, ignore_index = 0) # Zero token is a padding token
            return x, loss

        return x