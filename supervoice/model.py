import torch
from torch.nn import functional as F
from .transformer import Transformer, TransformerAdvanced
from .masks import create_padding_mask, create_padding_casual_mask, create_padding_rectangle_mask

class SupervoiceGPT(torch.nn.Module):
    def __init__(self, config):
        super(SupervoiceGPT, self).__init__()
        self.config = config
        self.n_input_tokens = config.tokenizer.vocab_size
        self.n_output_tokens = len(config.tokenizer.vocab_output)
        self.n_durations = (config.gpt.max_duration + 1) + 1 # +1 Padding

        # Embeddings
        self.input_embedding = torch.nn.Embedding(self.n_input_tokens, self.config.gpt.n_dim)
        torch.nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)
        self.output_embedding_token = torch.nn.Embedding(self.n_output_tokens, self.config.gpt.n_dim - self.config.gpt.n_dim_duration)
        torch.nn.init.normal_(self.output_embedding_token.weight, mean=0.0, std=0.02)
        self.output_embedding_duration = torch.nn.Embedding(self.n_durations, self.config.gpt.n_dim_duration)
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
        self.prediction_head_token = torch.nn.Linear(self.config.gpt.n_dim - self.config.gpt.n_dim_duration, self.n_output_tokens, bias=False)
        self.prediction_head_duration = torch.nn.Linear(self.config.gpt.n_dim_duration, self.n_durations, bias=False)

        # Weight sharing
        self.output_embedding_token.weight = self.prediction_head_token.weight
        self.output_embedding_duration.weight = self.prediction_head_duration.weight

    def forward(self, *,
        input, 
        input_lengths = None, 
        output_tokens, 
        output_durations,
        output_lengths = None, 
        target_tokens = None, 
        target_durations = None
    ):

        # Check input
        assert len(input.size()) == 2, 'Input tensor shape should be [batch_size, sequence_length]'
        assert len(output_tokens.size()) == 2, 'Input tensor shape should be [batch_size, sequence_length]'
        assert len(output_durations.size()) == 2, 'Input tensor shape should be [batch_size, sequence_length]'
        assert input.size(0) == output_tokens.size(0), 'Input and output batch size should be the same'
        assert output_tokens.size(0) == output_durations.size(0), 'Output batch sizes should be the same'
        assert output_tokens.size(1) == output_durations.size(1), 'Output sequence lengths should be the same'

        # Create input mask for self-attention which is useful for training on variable length sequences
        if input_lengths is None:
            input_lengths = torch.tensor([input.size(1)] * input.size(0), device = input.device)
        input_mask = create_padding_mask(input_lengths, input.size(1), device = input.device).unsqueeze(1)

        # Create output mask for self-attention which is useful for training on variable length sequences
        if output_lengths is None:
            output_lengths = torch.tensor([output_tokens.size(1)] * output_tokens.size(0), device = output_tokens.device)
        output_mask = create_padding_casual_mask(output_lengths, output_tokens.size(1), device = output_tokens.device).unsqueeze(1)

        # Create input-output masks for cross-attention which is useful for training on variable length sequences
        input_output_mask = create_padding_rectangle_mask(output_lengths, input_lengths, output_tokens.size(1), input.size(1), device = input.device).unsqueeze(1)

        # Embeddings
        input_embedded = self.input_embedding(input)
        output_tokens_embedded = self.output_embedding_token(output_tokens)
        output_durations_embedded = self.output_embedding_duration(output_durations)
        output_embedded = torch.cat((output_tokens_embedded, output_durations_embedded), dim = -1)

        # Run an encoder
        latents = self.encoder(input_embedded, mask = input_mask)

        # Run an decoder
        decoded = self.decoder(latents, output_embedded, x_mask = input_mask, y_mask = output_mask, xy_mask = input_output_mask)

        # Split the output into token and duration
        decoded_token = decoded[:, :, :self.config.gpt.n_dim - self.config.gpt.n_dim_duration]
        decoded_duration = decoded[:, :, self.config.gpt.n_dim - self.config.gpt.n_dim_duration:]

        # Run prediction head
        predicted_token = self.prediction_head_token(decoded_token)
        predicted_duration = self.prediction_head_duration(decoded_duration)

        # Compute loss if targets are provided
        if target_tokens is not None and target_durations is not None:
            loss_duration = F.cross_entropy(predicted_duration.view(-1, predicted_duration.size(-1)), target_durations.view(-1), ignore_index = 0)
            loss_token = F.cross_entropy(predicted_token.view(-1, predicted_token.size(-1)), target_tokens.view(-1), ignore_index = 0)
            loss = loss_token + loss_duration
            return predicted_token, predicted_duration, loss

        return predicted_token, predicted_duration

    @torch.no_grad()
    def generate(self, input, tokenizer, max_new_tokens = 128, temperature=1.0, top_k=None, deterministic = False, device="cpu"):
        ctx_input = torch.tensor([tokenizer.sequence_begin_token_id] + tokenizer.encode(input) + [tokenizer.sequence_end_token_id], device = device).unsqueeze(0)
        ctx_output_tokens = torch.tensor([tokenizer.sequence_begin_token_id], device = device).unsqueeze(0)
        ctx_output_durations = torch.tensor([0], device = device).unsqueeze(0)
        valid_exit = False
        for _ in range(max_new_tokens):
            
            # Forward the model to get the logits for the index in the sequence
            logits_token, logits_duration = self(input = ctx_input, output_tokens = ctx_output_tokens, output_durations = ctx_output_durations)
            
            # Pluck the logits at the final step and scale by desired temperature
            logits_token = logits_token[:, -1, :] / temperature
            logits_duration = logits_duration[:, -1, :] / temperature

            # Truncate the logits to only having generate tokens
            # logits = logits[:, :self.n_generate_tokens]
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits_token, min(top_k, logits_token.size(-1)))
                logits_token[logits_token < v[:, [-1]]] = -float('Inf')
                v, _ = torch.topk(logits_duration, min(top_k, logits_duration.size(-1)))
                logits_duration[logits_duration < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to convert logits to (normalized) probabilities
            probs_token = F.softmax(logits_token, dim=-1)
            probs_duration = F.softmax(logits_duration, dim=-1)
            
            # Sample from the distribution
            if deterministic:
                idx_next_token = torch.argmax(probs_token, dim=-1, keepdim=True)
                idx_next_duration = torch.argmax(probs_duration, dim=-1, keepdim=True)
            else:
                idx_next_token = torch.multinomial(probs_token, num_samples=1)
                idx_next_duration = torch.multinomial(probs_duration, num_samples=1)
            
            # Append Context
            ctx_output_tokens = torch.cat((ctx_output_tokens, idx_next_token), dim=1)
            ctx_output_durations = torch.cat((ctx_output_durations, idx_next_duration), dim=1)

            # Stop Tokens
            if idx_next_token == tokenizer.sequence_end_token_id:
                valid_exit = True
                break

        tokens = tokenizer.decode_phonemes(ctx_output_tokens.squeeze(0).cpu().tolist())
        durations = (ctx_output_durations.squeeze(0).cpu() - 1).tolist()
        tokens = tokens[1:]
        durations = durations[1:]
        if valid_exit:
            tokens = tokens[:-1]
            durations = durations[:-1]
        return list(zip(tokens, durations))

    def predict_next(self, input, output_tokens, output_durations, tokenizer, top_k = 10, device = "cpu"):

        # Context
        ctx_input = torch.tensor([tokenizer.sequence_begin_token_id] + tokenizer.encode(input) + [tokenizer.sequence_end_token_id], device = device).unsqueeze(0)
        ctx_output_tokens = torch.tensor([tokenizer.sequence_begin_token_id] + tokenizer.encode_phonemes(output_tokens), device = device).unsqueeze(0)
        ctx_output_durations = torch.tensor([0] + output_durations, device = device).unsqueeze(0)

        # Predict next token
        logits_token, logits_duration = self(input = ctx_input, output_tokens = ctx_output_tokens, output_durations = ctx_output_durations)
        logits_token.squeeze_(0)
        logits_duration.squeeze_(0)
        logits_token = logits_token[-1, :]
        logits_duration = logits_duration[-1, :]

        # Probabilities
        probs_token = F.softmax(logits_token, dim=-1)

        # Get top k
        probs_token, indices = torch.topk(probs_token, top_k)
        
        return probs_token.cpu().tolist(), tokenizer.decode_phonemes(indices.cpu().tolist())