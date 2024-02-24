import torch
import sentencepiece as spm
from .normalize import normalize

class Tokenizer:
    def __init__(self, config, path):
        self.vocab_size = config.tokenizer.vocab_size

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

        # Load processor
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(path)

        # IDs
        self.pad_token_id = self.sp.piece_to_id(self.pad_token)
        self.silence_token_id = self.sp.piece_to_id(self.silence_token)
        self.unknown_token_id = self.sp.piece_to_id(self.unknown_token)
        self.sequence_begin_token_id = self.sp.piece_to_id(self.sequence_begin_token)
        self.sequence_end_token_id = self.sp.piece_to_id(self.sequence_end_token)
        self.text_begin_token_id = self.sp.piece_to_id(self.text_begin_token)
        self.text_end_token_id = self.sp.piece_to_id(self.text_end_token)
        self.phonemes_begin_token_id = self.sp.piece_to_id(self.phonemes_begin_token)
        self.phonemes_end_token_id = self.sp.piece_to_id(self.phonemes_end_token)

    def encode(self, text):

        # Normalize first
        text = normalize(text).lower()

        # Encode
        return self.sp.encode(text)
    
    def encode_to_str(self, text):
        
        # Normalize first
        text = normalize(text).lower()

        # Encode
        return self.sp.encode(text, out_type=str)

    def decode(self, tokens):
        return self.sp.decode(tokens)