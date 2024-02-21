import torch

# Some garbage tokens that could be safely ignored
ignored = [    
    '́', '€', '≡', '京', '先', '大', '奔', '尚', '时', '熊', '生', '都', '阪', 'ﬂ', '՚',
    'נ', 'ע', '~', '§', '¯', 'æ'
]

# Mapping keys
mapped_keys = {

    # Various dashes
    '‑': '-', 
    '–': '-', 
    '—': '-', 
    '−': '-',
    '→': '-',

    # Various quotes
    '"': '\'',
    '`': '\'',
    '´': '\'',
    '‘': '\'',
    '’': '\'',
    '“': '\'',
    '”': '\'',
    '„': '\'',
    '«': '\'',
    '»': '\'',
    'ʻ': '\'',
}

class Tokenizer:
    def __init__(self, config):

        # Tokens
        self.pad_token = config.tokenizer.pad_token
        self.silence_token = config.tokenizer.silence_token
        self.sequence_begin_token = config.tokenizer.sequence_begin_token
        self.sequence_end_token = config.tokenizer.sequence_end_token
        self.text_begin_token = config.tokenizer.text_begin_token
        self.text_end_token = config.tokenizer.text_end_token
        self.phonemes_begin_token = config.tokenizer.phonemes_begin_token
        self.phonemes_end_token = config.tokenizer.phonemes_end_token
        self.unknown_token = config.tokenizer.unknown_token

        # Special tokens
        self.special_tokens = [
            self.pad_token, 
            self.silence_token, 
            self.unknown_token,
            self.sequence_begin_token, 
            self.sequence_end_token, 
            self.text_begin_token,
            self.text_end_token,
            self.phonemes_begin_token,
            self.phonemes_end_token
        ]

        # Input tokens
        self.all_tokens = self.special_tokens + config.tokenizer.phonemes + config.tokenizer.input_tokens
        self.input_tokens = self.all_tokens
        self.output_tokens = self.special_tokens + config.tokenizer.phonemes # Truncated all tokens

        # Map
        self.token_to_id = {token: i for i, token in enumerate(self.all_tokens)}
        self.id_to_token = {i: token for i, token in enumerate(self.all_tokens)}

        # Add mapped keys
        for key, value in mapped_keys.items():
            self.token_to_id[key] = self.token_to_id[value]

        # IDs
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.silence_token_id = self.token_to_id[self.silence_token]
        self.unknown_token_id = self.token_to_id[self.unknown_token]
        self.sequence_begin_token_id = self.token_to_id[self.sequence_begin_token]
        self.sequence_end_token_id = self.token_to_id[self.sequence_end_token]
        self.text_begin_token_id = self.token_to_id[self.text_begin_token]
        self.text_end_token_id = self.token_to_id[self.text_end_token]
        self.phonemes_begin_token_id = self.token_to_id[self.phonemes_begin_token]
        self.phonemes_end_token_id = self.token_to_id[self.phonemes_end_token]

    def normalize(self, tokens):
        return [mapped_keys[token] if token in mapped_keys else token for token in tokens if token not in ignored]

    def reverse(self, tokens):
        return [self.id_to_token[token] for token in tokens]

    def __call__(self, tokens, force = False):
        if force:
            return torch.tensor([(self.token_to_id[token] if token in self.token_to_id else self.unknown_token_id) for token in tokens])
        else:
            missing = []
            for token in tokens:
                if token not in self.token_to_id:
                    missing.append(token)
            if missing:
                raise ValueError(f"Tokens not found: {missing}")
            return torch.tensor([self.token_to_id[token] for token in tokens])