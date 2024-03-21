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
        self.unknown_token = config.tokenizer.unknown_token

        # Load processor
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(path)

        # Phoneme map
        self.phoneme_to_id = {}
        self.id_to_phoneme = {}
        for p in range(len(config.tokenizer.vocab_output)):
            self.phoneme_to_id[config.tokenizer.vocab_output[p]] = p
            self.id_to_phoneme[p] = config.tokenizer.vocab_output[p]

        # IDs
        self.silence_token_id = self.phoneme_to_id[self.silence_token]
        self.sequence_begin_token_id = self.sp.piece_to_id(self.sequence_begin_token)
        self.sequence_end_token_id = self.sp.piece_to_id(self.sequence_end_token)

    def encode(self, text):

        # Normalize first
        text = normalize(text).lower()

        # Encode
        return self.sp.encode(text)

    def encode_sample(self, text):

        # Normalize first
        text = normalize(text).lower()

        # Encode
        return self.sp.encode(text, enable_sampling=True, alpha=0.1, nbest_size=-1)
    
    def encode_to_str(self, text):
        
        # Normalize first
        text = normalize(text).lower()

        # Encode
        return self.sp.encode(text, out_type=str)

    def encode_phonemes(self, phonemes):
        return [self.phoneme_to_id[p] for p in phonemes]

    def decode_phonemes(self, phonemes):
        return [self.id_to_phoneme[p] for p in phonemes]

    def decode(self, tokens):
        return self.sp.decode(tokens)