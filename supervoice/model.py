import torch
from torch.nn import functional as F
from .transformer import Transformer

class SupervoiceGPT(torch.nn.Module):
    def __init__(self, config):
        super(SupervoiceGPT, self).__init__()
        self.config = config
        self.n_tokens = config.tokenizer.vocab_size

        # Input embedding
        self.input_embedding = torch.nn.Embedding(self.n_tokens, self.config.gpt.n_dim)
        torch.nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)

        # Transformer
        self.transformer = Transformer(

            # Architecture
            n_heads = self.config.gpt.n_heads,
            n_layers = self.config.gpt.n_layers,
            n_dim = self.config.gpt.n_dim,
            n_dim_head = self.config.gpt.n_dim_head,
            n_dim_ffn = self.config.gpt.n_dim_ffn,

            # Masking
            casual = True,

            # Dropout
            att_dropout = 0,
            ffn_dropout = 0.1,

            # Positional embedding
            position_embedding = 'alibi'
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
            loss = F.cross_entropy(x.view(-1, x.size(-1)), target.view(-1), ignore_index = -1) # We expect -1 to be a mask value that excludes loss
            return x, loss

        return x

    @torch.no_grad()
    def generate(self, input, max_new_tokens, temperature=1.0, top_k=None, stop_tokens = None, deterministic = False):
        ctx = input
        for _ in range(max_new_tokens):
            
            # Forward the model to get the logits for the index in the sequence
            logits = self(ctx)
            
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # Truncate the logits to only having generate tokens
            # logits = logits[:, :self.n_generate_tokens]
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            if deterministic:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append Context
            ctx = torch.cat((ctx, idx_next), dim=1)

            # Stop Tokens
            if idx_next in stop_tokens:
                break

        return ctx

    def predict_next(self, input, top_k, trim_generated = True):

        # Predict next token
        logits = self(input.unsqueeze(0)).squeeze(0)
        logits = logits[-1, :]
        if trim_generated:
            logits = logits[:self.n_generate_tokens]

        # Probabilities
        probs = F.softmax(logits, dim=-1)

        # Get top k
        probs, indices = torch.topk(probs, top_k)
        
        return probs, indices