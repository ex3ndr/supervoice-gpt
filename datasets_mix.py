# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Imports
import os
import glob
import textgrid
from tqdm import tqdm
from pathlib import Path
import numpy as np
from supervoice_gpt.normalize import normalize
from supervoice_gpt.alignment import compute_alignments
from supervoice_gpt import Tokenizer, config
import sentencepiece as spm
import multiprocessing
import torch
import math

def process_segments_async(file):
    parts = Path(file).parts
    collection = parts[1][0:-(len('-aligned'))]
    file = parts[2] + "/" + parts[3].split(".")[0]
    return process_segments(collection, file)

def process_segments(collection, path):

    # Load text
    with open('datasets/' + collection + "-prepared/" + path + ".txt", "r") as f:
        text = f.read()
    text = normalize(text.lower().strip())

    # Load TextGrid
    tg = textgrid.TextGrid.fromFile('datasets/' + collection + "-aligned/" + path + ".TextGrid")

    # Load style tensor
    style = torch.load('datasets/' + collection + "-prepared/" + path + ".f0.pt", map_location="cpu")
    
    # Compute alignments
    aligned_phonemes = compute_alignments(config, tg, style, style.shape[0])

    # Check for unknown tokens, or invalid durations
    for phoneme, duration, style in aligned_phonemes:
        if phoneme == config.tokenizer.unknown_token:
            return None
        if duration > 100: # ~1 second
            return None
    
    # Collect phonemes
    known_phonemes = set()
    for phoneme, duration, pitch in aligned_phonemes:
        known_phonemes.add(phoneme)

    # Trim silence
    while len(aligned_phonemes) > 0 and aligned_phonemes[0][0] == config.tokenizer.silence_token:
        aligned_phonemes.pop(0)
    while len(aligned_phonemes) > 0 and aligned_phonemes[-1][0] == config.tokenizer.silence_token:
        aligned_phonemes.pop(-1)

    # GPT training
    output_train = [text + '｜' + format_durations_train(aligned_phonemes)]

    # Tokenizer training
    output_train_text = [text]

    # Results    
    return output_train, output_train_text, known_phonemes

def format_durations_train(phonemes):
    output = []
    for phoneme, duration, pitch in phonemes:
        output += ["" + phoneme + "," + str(duration) + "," + str(pitch) + ""]
    return " ".join(output)

def main():

    # Corpus
    print("Starting assembling text training corpus...")
    files = [] 
    files += glob.glob("datasets/vctk-aligned/*/*.TextGrid")
    files += glob.glob("datasets/libritts-aligned/*/*.TextGrid")
    files += glob.glob("datasets/common-voice-en-aligned/*/*.TextGrid")
    files += glob.glob("datasets/common-voice-ru-aligned/*/*.TextGrid") 
    files += glob.glob("datasets/common-voice-uk-aligned/*/*.TextGrid")
    
    # Process files
    print("Processing files...")
    known_phonemes = set()
    file_text = open("datasets/train.txt", "w")
    file_tok_text = open("datasets/train_tokenizer_text.txt", "w")
    with multiprocessing.Manager() as manager:
        with multiprocessing.Pool(processes=16) as pool:
            for result in tqdm(pool.imap_unordered(process_segments_async, files, chunksize=32), total=len(files)):
                if result is None:
                    continue
                segments, segments_tokenizer_text, kp = result
                known_phonemes = known_phonemes.union(kp)
                for s in segments:
                    file_text.write(s + "\n")
                for s in segments_tokenizer_text:
                    file_tok_text.write(s + "\n")

    # Close files
    file_text.close()
    file_tok_text.close()

    # Write vocab
    with open("./datasets/tokenizer_phonemes.vocab", "w") as vc:

        # Collect tokens
        items = [ "<pad>", "<s>", "</s>"]
        phon = []
        for k in known_phonemes:
            phon.append(k)
        phon.sort()
        items += phon
        
        # Wrap in quotes
        items = [f'"{i}"' for i in items]

        # Write
        vc.write("[" + ",".join(items) + "]")
        

if __name__ == "__main__":
    main()