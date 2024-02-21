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
from supervoice.tokenizer import Tokenizer
from train_config import config

tokenizer = Tokenizer(config)

def process_segments(collection, path):
    data = extract_phonemes(collection, path)
    output = []

    # Text tokens
    output += [tokenizer.text_begin_token] 
    output += format_tokens(data['t'])
    output += [tokenizer.text_end_token]

    # Phoneme tokens
    phoneme_with_durations = []
    for word in data['w']:
        if word['w'] is not None:
            for phoneme in word['p']:
                if phoneme['p'] is not None:
                    phoneme_with_durations.append((phoneme['p'], phoneme['t'][1] - phoneme['t'][0]))
                else:
                    phoneme_with_durations.append((tokenizer.silence_token, phoneme['t'][1] - phoneme['t'][0]))
        else:
            phoneme_with_durations.append((tokenizer.silence_token, word['t'][1] - word['t'][0]))

    # Trim silence
    while len(phoneme_with_durations) > 0 and phoneme_with_durations[0][0] == tokenizer.silence_token:
        phoneme_with_durations.pop(0)
    while len(phoneme_with_durations) > 0 and phoneme_with_durations[-1][0] == tokenizer.silence_token:
        phoneme_with_durations.pop(-1)

    # Add phonemes
    output += [tokenizer.phonemes_begin_token]
    output += format_durations(phoneme_with_durations)
    output += [tokenizer.phonemes_end_token]

    # Wrap
    output = [tokenizer.sequence_begin_token] + output + [tokenizer.sequence_end_token]

    # Check if unknown tokens are present
    if tokenizer.unknown_token in output:
        return ([], [])

    # Generate text and tokenized
    output_tokenized = tokenizer(output)
    output = ''.join(output)

    return ([output], [output_tokenized])

def format_durations(phonemes):
    output = []
    for phoneme, duration in phonemes:
        for i in range(round(duration / config.audio.token_duration)):
            output.append(phoneme)
    return output

def format_tokens(src):
    chars = list(src)
    chars = tokenizer.normalize(chars)
    return chars

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
                    m = '<UNK>'
                word_phonemes.append({'t': [phone.minTime + time_offset, phone.maxTime + time_offset], 'p': m})
            last_phone_time = phone.maxTime

        # Processed word
        output_words.append({'t': [word.minTime + time_offset, word.maxTime + time_offset], 'w': word.mark, 'p': word_phonemes})
        last_word_time = word.maxTime

    return { 't': text, 'w': output_words, 'd': last_word_time + time_offset }

def main():

    print("Loading files...")
    files = [] 
    files += glob.glob("datasets/vctk-aligned/*/*.TextGrid") 
    files += glob.glob("datasets/libritts-aligned/*/*.TextGrid")
    files += glob.glob("datasets/common-voice-en-aligned/*/*.TextGrid")
    files += glob.glob("datasets/common-voice-ru-aligned/*/*.TextGrid") 
    files += glob.glob("datasets/common-voice-uk-aligned/*/*.TextGrid")
    
    # Process files
    print("Processing files...")
    output_binary = []
    file_text = open("datasets/train.txt", "w")
    file_bin = open("datasets/train.bin", "w")
    for file in tqdm(files):
        parts = Path(file).parts
        collection = parts[1][0:-(len('-aligned'))]
        file = parts[2] + "/" + parts[3].split(".")[0]

        # Collect segments
        segments, segments_binary = process_segments(collection, file)

        # Write to text file
        for s in segments:
            file_text.write(s + "\n")

        # Write to binary file
        for s in segments_binary:
            np.array(s, dtype=np.uint16).tofile(file_bin)

    # Close files
    file_text.close()
    file_bin.close()
    

if __name__ == "__main__":
    main()