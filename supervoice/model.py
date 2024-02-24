import torch
from torch.nn import functional as F
from .transformer import Transformer, TransformerAdvanced

class SupervoiceGPT(torch.nn.Module):
    def __init__(self, config):
        super(SupervoiceGPT, self).__init__()
        self.config = config
        self.n_input_tokens = config.tokenizer.vocab_size
        self.n_output_tokens = config.tokenizer.vocab_size_output
        self.n_durations = config.gpt.max_durations - config.gpt.min_durations + 1

        # Embeddings
        self.input_embedding = torch.nn.Embedding(self.n_input_tokens, self.config.gpt.n_dim)
        torch.nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)
        self.output_embedding_token = torch.nn.Embedding(self.n_output_tokens, self.config.gpt.n_dim)
        torch.nn.init.normal_(self.output_embedding_token.weight, mean=0.0, std=0.02)
        self.output_embedding_duration = torch.nn.Embedding(self.n_durations, self.config.gpt.n_dim)
        torch.nn.init.normal_(self.output_embedding_duration.weight, mean=0.0, std=0.02)

        # Encoder Transformer
        self.encoder = Transformer(

            # Architecture
            n_heads = self.config.gpt.n_heads,
            n_layers = self.config.gpt.n_layers,
            n_dim = self.config.gpt.n_dim,
            n_dim_head = self.config.gpt.n_dim_head,
            n_dim_ffn = self.config.gpt.n_dim_ffn,

            # Dropout
            att_dropout = 0,
            ffn_dropout = 0.1,

            # Positional embedding
            position_embedding = 'alibi'
        )

        # Decoder Transformer
        self.decoder = TransformerAdvanced(
            
            # Architecture
            n_heads = self.config.gpt.n_heads,
            n_layers = self.config.gpt.n_layers,
            n_dim = self.config.gpt.n_dim,
            n_dim_head = self.config.gpt.n_dim_head,
            n_dim_ffn = self.config.gpt.n_dim_ffn,

            # Dropout
            att_dropout = 0,
            ffn_dropout = 0.1,

            # Positional embedding
            position_embedding = 'alibi'
        )

        # Prediction heads
        self.prediction_head_token = torch.nn.Linear(self.config.gpt.n_dim, self.n_output_tokens, bias=False)
        self.prediction_head_duration = torch.nn.Linear(self.config.gpt.n_dim, self.n_durations, bias=False)

        # Weight sharing
        self.output_embedding_token.weight = self.prediction_head_token.weight
        self.output_embedding_duration.weight = self.prediction_head_duration.weight

    def forward(self, *,
        input, 
        input_lengths = None, 
        output, 
        output_lengths = None, 
        target_phonemes = None, 
        target_durations = None
    ):

        # Create input mask for cross-attention which is useful for training on variable length sequences
        input_mask = None
        if input_lengths is not None:
            input_mask = padded_square_mask(input_lengths, input.size(1), casual = False, device = input.device)

        # Create output mask for self-attention which is useful for training on variable length sequences
        output_mask = None
        if input_lengths is not None or output_lengths is not None:
            input_mask = padded_square_mask(input_lengths, input.size(1), casual = True, device = input.device)

        # Input embeddings
        input_embedded = self.input_embedding(input)

        # Output embeddings
        output_embedded = self.output_embedding_token(output)

        # Run an encoder
        latents = self.encoder(input_embedded, mask = input_mask)

        # Run an decoder
        decoded = self.decoder(latents, output_embedded, x_mask = input_mask, y_mask = output_mask)

        # Run prediction head
        predicted_token = self.prediction_head_token(decoded)
        preducted_duration = self.prediction_head_duration(decoded)

        # Compute loss if targets are provided
        if target_phonemes is not None and target_durations is not None:
            loss_duration = F.cross_entropy(preducted_duration.view(-1, preducted_duration.size(-1)), target_durations.view(-1), ignore_index = -1)
            loss_token = F.cross_entropy(predicted_token.view(-1, predicted_token.size(-1)), target_phonemes.view(-1), ignore_index = -1)
            loss = loss_token + loss_duration
            return predicted_token, preducted_duration, loss

        return predicted_token, preducted_duration

    # @torch.no_grad()
    # def generate(self, input, max_new_tokens, temperature=1.0, top_k=None, stop_tokens = None, deterministic = False):
    #     ctx = input
    #     for _ in range(max_new_tokens):
            
    #         # Forward the model to get the logits for the index in the sequence
    #         logits = self(ctx)
            
    #         # Pluck the logits at the final step and scale by desired temperature
    #         logits = logits[:, -1, :] / temperature

    #         # Truncate the logits to only having generate tokens
    #         # logits = logits[:, :self.n_generate_tokens]
            
    #         # Optionally crop the logits to only the top k options
    #         if top_k is not None:
    #             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #             logits[logits < v[:, [-1]]] = -float('Inf')
            
    #         # Apply softmax to convert logits to (normalized) probabilities
    #         probs = F.softmax(logits, dim=-1)
            
    #         # Sample from the distribution
    #         if deterministic:
    #             idx_next = torch.argmax(probs, dim=-1, keepdim=True)
    #         else:
    #             idx_next = torch.multinomial(probs, num_samples=1)
            
    #         # Append Context
    #         ctx = torch.cat((ctx, idx_next), dim=1)

    #         # Stop Tokens
    #         if idx_next in stop_tokens:
    #             break

    #     return ctx

    # def predict_next(self, input, top_k, trim_generated = True):

    #     # Predict next token
    #     logits = self(input.unsqueeze(0)).squeeze(0)
    #     logits = logits[-1, :]
    #     if trim_generated:
    #         logits = logits[:self.n_generate_tokens]

    #     # Probabilities
    #     probs = F.softmax(logits, dim=-1)

    #     # Get top k
    #     probs, indices = torch.topk(probs, top_k)
        
    #     return probs, indices

def padded_square_mask(lengths, max_length, casual, device):
    batch_size = lengths.size(0)
    mask = torch.zeros(batch_size, max_length, max_length, device = device, dtype = torch.bool)
    for i in range(batch_size):
        mask[i, :lengths[i], :lengths[i]] = True
    mask = torch.where(mask, 0, float('-inf'))
    if casual:
        mask = mask + torch.triu(torch.full((max_length, max_length), float('-inf'), device = device), diagonal = 1)
    return mask