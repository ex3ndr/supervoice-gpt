from utils.misc import dict_to_object

config = dict_to_object({

    # Audio
    "audio": {
        "token_duration": 256 / 24000 # 256 samples at 24kHz
    },

    # Architecture
    "gpt": {
        "n_embeddings": 512,
        "n_heads": 8,
        "n_layers": 8,
        "n_dim": 512,
        "n_dim_head": 64,
        "n_dim_ffn": 2048,

        # Minimum and maximum duration values
        "min_durations": 1,
        "max_durations": 100
    },

    # Tokenizer
    "tokenizer": {
        "vocab_size": 4096,

        "vocab_size_output": 103 + 2, # Phonemes/silence + <s> and </s> 

        "output_vocab": ['<s>', '</s>',  ],

        # Special tokens
        "pad_token": "<pad>",
        "silence_token": "•",
        "sequence_begin_token": "<s>",
        "sequence_end_token": "</s>",
        "text_begin_token": "<t>",
        "text_end_token": "</t>",
        "phonemes_begin_token": "<p>",
        "phonemes_end_token": "</p>",
        "unknown_token": "<unk>",
    }
})