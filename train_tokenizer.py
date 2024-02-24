from train_config import config
import sentencepiece as spm

print("Training text tokenizer")
spm.SentencePieceTrainer.train(
    input = "datasets/train_tokenizer_text.txt", 
    model_prefix = "tokenizer_text", 
    vocab_size = config.tokenizer.vocab_size, 
    character_coverage = 1.0, 
    num_threads = 32,
    # This avoid binding spaces to tokens since we want to use them as a separate tokens
    add_dummy_prefix = False,
    allow_whitespace_only_pieces = True,
    user_defined_symbols = '▁'
)

print("Training phonemes tokenizer")
spm.SentencePieceTrainer.train(
    input = "datasets/train_tokenizer_phonemes.txt", 
    model_prefix = "tokenizer_phonemes", 
    vocab_size = config.tokenizer.vocab_size, 
    character_coverage = 1.0, 
    num_threads = 32,
    max_sentencepiece_length = 64,
    split_by_unicode_script = False, # This is important for phonemes
    add_dummy_prefix = False # This removes "▁" from the beginning of tokens
)