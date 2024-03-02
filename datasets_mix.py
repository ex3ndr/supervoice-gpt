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

    # Pitch
    max_pitch = 0.0
    max_pitch_2 = 0.0

    # Text
    text = format_tokens(data['t'])

    # Phoneme tokens
    phoneme_with_durations = []
    for word in data['w']:
        if word['w'] is not None:
            for phoneme in word['p']:

                # Extract phoneme
                duration = min(phoneme['t'][1] - phoneme['t'][0], 1) # Clip duration to 1 second (actually ignored in previous stage)
                p = phoneme['p'] if phoneme['p'] is not None else config.tokenizer.silence_token
                pitch = phoneme['pitch'] if phoneme['pitch'] is not None else 0
                max_pitch = max(max_pitch, pitch)
                pitch = math.log(pitch + 1)
                pitch =  max(config.audio.pitch_min, min(config.audio.pitch_max, pitch))
                max_pitch_2 = max(max_pitch_2, pitch)

                # Write phoneme
                phoneme_with_durations.append((p, duration, pitch))

                # Persist statistics
                if phoneme['p'] is not None:
                    known_phonemes[phoneme['p']] = max(known_phonemes.get(phoneme['p'], 0), round(duration / config.audio.token_duration))
        else:
            # Silence between words
            duration = min(word['t'][1] - word['t'][0], 1) # Clip duration to 1 second (actually ignored in previous stage)
            phoneme_with_durations.append((config.tokenizer.silence_token, duration, 0))

    # Trim silence
    while len(phoneme_with_durations) > 0 and phoneme_with_durations[0][0] == config.tokenizer.silence_token:
        phoneme_with_durations.pop(0)
    while len(phoneme_with_durations) > 0 and phoneme_with_durations[-1][0] == config.tokenizer.silence_token:
        phoneme_with_durations.pop(-1)


    # Check for unknown tokens
    if (config.tokenizer.unknown_token in text) or (config.tokenizer.unknown_token in format_durations(phoneme_with_durations)):
        return None

    # GPT Training
    output_train = [text + '｜' + format_durations_train(phoneme_with_durations)]

    # Tokenizer training
    output_train_text = [text]

    # Results    
    return output_train, output_train_text, known_phonemes, max_pitch, max_pitch_2

def format_durations(phonemes):
    output = ''
    for phoneme, duration, pitch in phonemes:
        for i in range(round(duration / config.audio.token_duration)):
            output += phoneme
    return output

def quntize_pitch(pitch):
    qp = ((pitch - config.audio.pitch_min) / (config.audio.pitch_max - config.audio.pitch_min)) * config.audio.pitch_buckets
    qp = max(0, min(config.audio.pitch_buckets - 1, int(qp)))
    return qp

def format_durations_train(phonemes):
    output = []
    for phoneme, duration, pitch in phonemes:
        output += ["" + phoneme + "," + str(round(duration / config.audio.token_duration)) + "," + str(quntize_pitch(pitch)) + ""]
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

    # Load pitch
    pitch = torch.load('datasets/' + collection + "-prepared/" + path + ".f0.pt", map_location="cpu")

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
    first = False
    for word in words:
        if word.mark == "": # Ignore empty words
            continue

        # Add silence between words
        if not first:
            output_words.append({'t': [last_word_time + time_offset, word.minTime + time_offset], 'w': None})
        first = False

        # Add word
        word_phonemes = []
        last_phone_time = 0
        for phone in phones:

            # Add missing silence
            if phone.minTime != last_phone_time:

                # Pitch
                start = round((phone.minTime + time_offset) / config.audio.token_duration)
                end = round((phone.maxTime + time_offset) / config.audio.token_duration)
                pt = pitch[start:end].median().item()

                word_phonemes.append({'t': [last_phone_time + time_offset, phone.minTime + time_offset], 'p': None, 'pitch': pt})
            
            # Add phone
            if phone.minTime >= word.minTime and phone.maxTime <= word.maxTime and phone.mark != "": # Ignore empty phones since we added them above

                # Pitch
                start = round((phone.minTime + time_offset) / config.audio.token_duration)
                end = round((phone.maxTime + time_offset) / config.audio.token_duration)
                pt = pitch[start:end].median().item()

                m = phone.mark
                if m == "spn":
                    m = config.tokenizer.unknown_token
                word_phonemes.append({'t': [phone.minTime + time_offset, phone.maxTime + time_offset], 'p': m, 'pitch': pt})
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
    max_pitch = 0.0
    max_pitch_2 = 0.0
    file_text = open("datasets/train.txt", "w")
    file_tok_text = open("datasets/train_tokenizer_text.txt", "w")
    with multiprocessing.Manager() as manager:
        with multiprocessing.Pool(processes=16) as pool:
            for result in tqdm(pool.imap_unordered(process_segments_async, files, chunksize=32), total=len(files)):
                if result is None:
                    continue
                segments, segments_tokenizer_text, kp, mp, mp2 = result
                for k, v in kp.items():
                    known_phonemes[k] = max(known_phonemes.get(k, 0), v)
                max_pitch = max(max_pitch, mp)
                max_pitch_2 = max(max_pitch_2, mp2)
                for s in segments:
                    file_text.write(s + "\n")
                for s in segments_tokenizer_text:
                    file_tok_text.write(s + "\n")
    with open("datasets/train_phonemes.txt", "w") as fp:
        for k, v in known_phonemes.items():
            fp.write(k + " " + str(v) + "\n")

    # Close files
    file_text.close()
    file_tok_text.close()

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

    print("Max Pitch: " + str(max_pitch) + ", " + str(max_pitch_2))
        

if __name__ == "__main__":
    main()