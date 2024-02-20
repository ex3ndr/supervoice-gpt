import torch
from torch.nn import functional as F

class SupervoiceGPT(torch.nn.Module):
    def __init__(self, config):
        super(SupervoiceGPT, self).__init__()
        self.config = config

        # Input embedding
        self.input_embedding = torch.nn.Embedding(self.n_input_tokens, self.config.n_dim)

        # Transformer
        self.transformer = Transformer(
            n_heads = self.config.n_heads,
            n_layers = self.config.n_layers,
            n_dim = self.config.n_dim,
            n_dim_head = self.config.n_dim_head,
            n_dim_ffn = self.config.n_dim_ffn,
            n_non_bias_tokens = 0,
            enable_skip_connections = False,
            casual = True,
            att_dropout = 0,
            ffn_dropout = 0.1
        )

        # Prediction head
        # NOTE: No weight tying since we have a very different token sets for input and output
        self.prediction_head = torch.nn.Linear(self.config.n_dim, self.n_output_tokens)

    def forward(self, x, target = None):

        # Input embeddings
        x = self.input_embedding(x)

        # Run a transformer
        x = self.transformer(x)

        # Run prediction head
        x = self.prediction_head(x)

        # Apply softmax
        x = F.softmax(x, dim = -1)

        if target is not None:
            # Compute loss
            loss = self.compute_loss(x, target)
            return x, loss

        return x