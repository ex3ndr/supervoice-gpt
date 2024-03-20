from .misc import dict_to_object

config = dict_to_object({

    # Audio
    "audio": {
        # "token_duration": 256 / 24000 # 256 samples at 24kHz
        "token_duration": 0.01, # To match the 100Hz token duration or Montreal Forced Aligner
        "sample_rate": 24000,
        "hop_size": 256
    },

    # Architecture
    "gpt": {
        "n_embeddings": 512,
        "n_heads": 16,
        "n_layers": 16,
        "n_dim": 512,
        "n_dim_head": 32, # n_dim / n_heads
        "n_dim_ffn": 32 * 4, # 4 * n_dim_head

        # # n_dim_duration + n_dim_pitch should be less than n_dim
        # "n_dim_duration": 128,
        # "n_dim_pitch": 128,

        # Maximum duration values
        "max_duration": 100,
    },

    "tokenizer_style":{
        "pitch_min": -2,
        "pitch_max": 2,
        "tokens": 256
    },

    # Tokenizer
    "tokenizer": {
        "vocab_size": 4096,
        "vocab_output": ["<pad>","<s>","</s>","<SIL>","a","aj","aw","aː","b","bʲ","bː","c","cʰ","cʷ","cː","d","dzʲ","dʐː","dʒ","dʲ","dʲː","d̪","d̪z̪","d̪z̪ː","d̪ː","e","ej","f","fʲ","fʲː","fː","h","i","iː","j","jː","k","kʰ","kʷ","kː","l","lː","m","mʲ","mʲː","mː","m̩","n","n̩","n̪","n̪ː","o","ow","p","pʰ","pʲ","pʲː","pʷ","pː","r","rʲ","rʲː","rː","s","sʲ","sʲː","s̪","s̪ː","t","tsʲ","tsʲː","tɕ","tɕː","tʂ","tʂː","tʃ","tʃʲ","tʃʲː","tʃː","tʰ","tʲ","tʲː","tʷ","t̪","t̪s̪","t̪s̪ː","t̪ː","u","v","vʲ","vʲː","vː","w","x","z","zʲ","zʲː","z̪","z̪ː","æ","ç","ð","ŋ","ɐ","ɑ","ɑː","ɒ","ɒː","ɔ","ɔj","ɕ","ɕː","ə","ɚ","ɛ","ɝ","ɟ","ɟʷ","ɟː","ɡ","ɡʷ","ɡː","ɣ","ɦ","ɦː","ɨ","ɪ","ɫ","ɫː","ɫ̩","ɱ","ɲ","ɲː","ɵ","ɹ","ɾ","ɾʲ","ɾː","ɾ̃","ʂ","ʂː","ʃ","ʃʲ","ʃʲː","ʉ","ʉː","ʊ","ʋ","ʋʲ","ʋʲː","ʋː","ʎ","ʎː","ʐ","ʐː","ʑː","ʒ","ʒʲ","ʒʲː","ʔ","ʝ","θ"],

        # Special tokens
        "pad_token": "</s>",
        "silence_token": "<SIL>",
        "sequence_begin_token": "<s>",
        "sequence_end_token": "</s>",
        "unknown_token": "<UNK>",
    }
})