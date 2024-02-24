# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Imports
import os
from supervoice.tokenizer import Tokenizer
from train_config import config

def main():
    print("Tokenizing train corpus...")
    tokenizer = Tokenizer(config, model_prefix + ".model")
    with open("datasets/train.txt", "r") as f:
        with open("datasets/train.bin", "w") as f2:
            for line in tqdm(f.readlines()):

                # Prepare prompt
                text, phonemes = line.split("｜")
                text = tokenizer.encode(text)
                phonemes = tokenizer.encode(phonemes)
                prompt = [tokenizer.sequence_begin_token_id, tokenizer.text_begin_token_id] \
                            + text \
                            + [tokenizer.text_end_token_id, tokenizer.phonemes_begin_token_id]\
                            + phonemes \
                            + [tokenizer.phonemes_end_token_id, tokenizer.sequence_end_token_id]

                # Write prompt
                np.array(prompt, dtype=np.uint16).tofile(f2)

if __name__ == "__main__":
    main()