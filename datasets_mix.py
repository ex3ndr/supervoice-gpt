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
from supervoice.normalize import normalize
from supervoice.tokenizer import Tokenizer
from train_config import config
import sentencepiece as spm
import multiprocessing

def process_segments_async(file):
    parts = Path(file).parts
    collection = parts[1][0:-(len('-aligned'))]
    file = parts[2] + "/" + parts[3].split(".")[0]
    return process_segments(collection, file)

def process_segments(collection, path):

    # Extract phonemes and text
    data = extract_phonemes(collection, path)

    # Check for unknown tokens, or invalid durations
    for word in data['w']:
        if word['w'] is not None:
            for phoneme in word['p']:
                if phoneme['p'] == config.tokenizer.unknown_token:
                    return None
                if phoneme['p'] is not None and (phoneme['t'][1] - phoneme['t'][0] > 1):
                    return None

    # Phoenemes
    known_phonemes = {}

    # Text
    text = format_tokens(data['t'])

    # Phoneme tokens
    phoneme_with_durations = []
    for word in data['w']:
        if word['w'] is not None:
            for phoneme in word['p']:
                duration = min(phoneme['t'][1] - phoneme['t'][0], 1) # Clip duration to 1 second
                if phoneme['p'] is not None:
                    known_phonemes[phoneme['p']] = max(known_phonemes.get(phoneme['p'], 0), round(duration / config.audio.token_duration))
                    phoneme_with_durations.append((phoneme['p'], duration))
                else:
                    phoneme_with_durations.append((config.tokenizer.silence_token, duration))
        else:
            duration = min(word['t'][1] - word['t'][0], 1) # Clip duration to 1 second
            phoneme_with_durations.append((config.tokenizer.silence_token, duration))

    # Trim silence
    while len(phoneme_with_durations) > 0 and phoneme_with_durations[0][0] == config.tokenizer.silence_token:
        phoneme_with_durations.pop(0)
    while len(phoneme_with_durations) > 0 and phoneme_with_durations[-1][0] == config.tokenizer.silence_token:
        phoneme_with_durations.pop(-1)

    # GPT Training
    output_train = [text + '｜' + format_durations_compact(phoneme_with_durations)]

    # Tokenizer training
    phonemes = format_durations(phoneme_with_durations)
    output_train_text = [text]
    output_train_phonemes = [phonemes]
    output_tokenizer = [text, phonemes]

    # Check if unknown tokens are present
    if (config.tokenizer.unknown_token in text) or (config.tokenizer.unknown_token in phonemes):
        return None
    
    return output_train, output_train_text, output_train_phonemes, output_tokenizer, known_phonemes

def format_durations(phonemes):
    output = ''
    for phoneme, duration in phonemes:
        for i in range(round(duration / config.audio.token_duration)):
            output += phoneme
    return output

def format_durations_compact(phonemes):
    output = []
    for phoneme, duration in phonemes:
        output += ["" + phoneme + "," + str(round(duration / config.audio.token_duration)) + ""]
    return " ".join(output)

def format_tokens(src):
    src = normalize(src)
    src = src.replace('•', '⋅') # Silence token
    return src

# Extract phonemes and text
def extract_phonemes(collection, path):
    phonemes = []

    # Load text
    with open('datasets/' + collection + "-prepared/" + path + ".txt", "r") as f:
        text = f.read()

    # Normalize
    text = text.lower().strip()

    # Load textgrid
    tg = textgrid.TextGrid.fromFile('datasets/' + collection + "-aligned/" + path + ".TextGrid")
    words = tg[0]
    phones = tg[1]
    assert words.name == "words"
    assert phones.name == "phones"

    # Process words
    output_words = []
    last_word_time = 0
    duration = words.maxTime
    time_offset = 0

    # Skip silence in the beginning
    i = 0
    while i < len(words) and words[i].mark == "":
        time_offset = -words[i].maxTime # Update offset
        last_word_time = words[i].maxTime
        i += 1

    # Process words
    for word in words:
        if word.mark == "": # Ignore empty words
            continue

        # Add silence between words
        if word.minTime != last_word_time:
            output_words.append({'t': [last_word_time + time_offset, word.minTime + time_offset], 'w': None})

        # Add word
        word_phonemes = []
        last_phone_time = 0
        for phone in phones:
            if phone.minTime != last_phone_time:
                word_phonemes.append({'t': [last_phone_time + time_offset, phone.minTime + time_offset], 'p': None})
            if phone.minTime >= word.minTime and phone.maxTime <= word.maxTime and phone.mark != "":
                m = phone.mark
                if m == "spn":
                    m = config.tokenizer.unknown_token
                word_phonemes.append({'t': [phone.minTime + time_offset, phone.maxTime + time_offset], 'p': m})
            last_phone_time = phone.maxTime

        # Processed word
        output_words.append({'t': [word.minTime + time_offset, word.maxTime + time_offset], 'w': word.mark, 'p': word_phonemes})
        last_word_time = word.maxTime

    return { 't': text, 'w': output_words, 'd': last_word_time + time_offset }

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
    known_phonemes = {}
    file_text = open("datasets/train.txt", "w")
    file_tok = open("datasets/train_tokenizer.txt", "w")
    file_tok_text = open("datasets/train_tokenizer_text.txt", "w")
    file_tok_ph = open("datasets/train_tokenizer_phonemes.txt", "w")
    with multiprocessing.Manager() as manager:
        with multiprocessing.Pool(processes=16) as pool:
            for result in tqdm(pool.imap_unordered(process_segments_async, files, chunksize=32), total=len(files)):
                if result is None:
                    continue
                segments, segments_tokenizer_text, segments_tokenizer_ph, segments_tokenizer, kp = result
                for k, v in kp.items():
                    known_phonemes[k] = max(known_phonemes.get(k, 0), v)
                for s in segments:
                    file_text.write(s + "\n")
                for s in segments_tokenizer:
                    file_tok.write(s + "\n")
                for s in segments_tokenizer_text:
                    file_tok_text.write(s + "\n")
                for s in segments_tokenizer_ph:
                    file_tok_ph.write(s + "\n")
    with open("datasets/train_phonemes.txt", "w") as fp:
        for k, v in known_phonemes.items():
            fp.write(k + " " + str(v) + "\n")

    # Close files
    file_text.close()
    file_tok.close()
    file_tok_text.close()
    file_tok_ph.close()

    # Write vocab
    with open("./datasets/tokenizer_phonemes.vocab", "w") as vc:

        # Collect tokens
        items = [ "<pad>", "<s>", "</s>", config.tokenizer.silence_token]
        phon = []
        for k, v in known_phonemes.items():
            phon.append(k)
        phon.sort()
        items += phon
        
        # Wrap in quotes
        items = [f'"{i}"' for i in items]

        # Write
        vc.write("[" + ",".join(items) + "]")
        

if __name__ == "__main__":
    main()