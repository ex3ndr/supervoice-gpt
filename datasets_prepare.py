# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import os
import multiprocessing
import glob
import torch
import torchaudio
import csv
from pathlib import Path
from tqdm import tqdm
from utils.audio import load_mono_audio, init_if_needed, trim_silence
from supervoice_gpt.model_style import export_style
from supervoice_gpt.config import config
import pyworld as pw
import shutil

#
# Parameters
#

PARAM_WORKERS = max(torch.cuda.device_count() * 4, 8)

CLEANUP_SYMBOLS = [
    "\n",
    "\r",
    "\t",
    "-",
    "\"",
    # "\'" # Breaks the text with apostrophes
]

#
# Execution
#

def speaker_directory(speaker):
    return str(speaker).zfill(8)

def execute_parallel(args):
    process_id = multiprocessing.current_process()._identity[0]
    files, collection_dir, index = args
    file, text, speaker, alignment = files[index]
    device = "cuda:" + str(process_id % torch.cuda.device_count())

    # Format filename from index (e.g. 000001)
    target_name = str(index).zfill(8)

    # Load audio
    waveform = load_mono_audio(file, config.audio.sample_rate, device=device)

    # Trim silence
    if not alignment:
        waveform = trim_silence(waveform, config.audio.sample_rate)

    # Move to CPU
    waveform = waveform.cpu()

    # Export style
    f0 = export_style(config, waveform)

    # Clean up text
    for symbol in CLEANUP_SYMBOLS:
        text = text.replace(symbol, " ")
    text = " ".join(text.split()) # Remove multiple spaces
    text = text.strip()

    # Save
    target_dir = os.path.join(collection_dir, speaker_directory(speaker))
    if not alignment:
        torchaudio.save(os.path.join(target_dir, target_name + ".wav"), waveform.unsqueeze(0), 16000)
    with open(os.path.join(target_dir, target_name + ".txt"), "w", encoding="utf-8") as f:
        f.write(text)
    torch.save(f0, os.path.join(target_dir, target_name + ".f0.pt"))

def copy_parallel(args):
    files, from_dir, to_dir, index = args
    file, speaker = files[index]
    target_name = str(index).zfill(8) + ".TextGrid"
    shutil.copy(os.path.join(from_dir, file), os.path.join(to_dir, speaker_directory(speaker), target_name))


def load_vctk_corpus():
    files = []
    speakers = {}

    # Load vctk corpus
    for file in glob.glob("external_datasets/vctk-corpus-0.92/**/*.flac"):
        directory, filename = os.path.split(file)
        _, speaker = os.path.split(directory)
        speaker = "vctk_" + speaker

        # Check if file is a valid audio file
        if file.endswith("_mic1.flac") or file.endswith("_mic2.flac"):

            # Load text
            base = filename[:-10]
            text_file = os.path.join(directory, base + ".txt")
            if os.path.exists(text_file):
                with open(text_file, "r") as f:
                    text = f.read()
            else:
                continue # not found

            # Extract speaker
            if speaker not in speakers:
                speakers[speaker] = len(speakers)
            speaker = speakers[speaker]

            # Append
            files.append((file, text, speaker, False))
        else:
            print("Strange filename:", filename)    

    return { 'files': files, 'speakers': speakers, 'preprocessed': [], 'preprocessed_base': None }

def load_libritts_corpus(collections):
    files = []
    speakers = {}

    # Ignore bad samples
    ignored = {}
    ignore_files = [
        "train-clean-100_bad_sample_list.txt", 
        "train-clean-360_bad_sample_list.txt", 
        "train-other-500_bad_sample_list.txt", 
        "test-clean_bad_sample_list.txt", 
        "test-other_bad_sample_list.txt",
        "dev-clean_bad_sample_list.txt",
        "dev-other_bad_sample_list.txt"
    ]
    for file_list_file in ignore_files:
        with open("external_datasets/libritts-r/failed/" + file_list_file, "r") as f:
            for line in f:
                line = line.replace("./train-clean-100/", "external_datasets/libritts-r-clean-100/")
                line = line.replace("./train-clean-360/", "external_datasets/libritts-r-clean-360/")
                line = line.replace("./train-other-500/", "external_datasets/libritts-r-other-500/")
                line = line.replace("./test-clean/", "external_datasets/libritts-r/test-clean/")
                line = line.replace("./test-other/", "external_datasets/libritts-r/test-other/")
                line = line.replace("./dev-clean/", "external_datasets/libritts-r/dev-clean/")
                line = line.replace("./dev-other/", "external_datasets/libritts-r/dev-other/")
                ignored[line.strip()] = True

    # Load vctk corpus
    files_index = []
    for c in collections:
        files_index += glob.glob(f"external_datasets/{c}/*/*/*.wav")
    for file in files_index:
        p = Path(file)
        filename = p.name
        directory = p.parent
        speaker = "libritts_" + p.parents[1].name

        # Check if file is a valid audio file
        if file in ignored:
            # print("Ignored file: " + file)
            continue

         # Load text
        base = filename[:-4]
        text_file = os.path.join(directory, base + ".normalized.txt")
        if os.path.exists(text_file):
            with open(text_file, "r") as f:
                    text = f.read()
        else:
            print("Text not found: " + text_file)
            continue # not found

        # Extract speaker
        if speaker not in speakers:
            speakers[speaker] = len(speakers)
        speaker = speakers[speaker]

        # Append
        files.append((file, text, speaker, False))

    return { 'files': files, 'speakers': speakers, 'preprocessed': [], 'preprocessed_base': None }

def load_common_voice_corpus(path):
    files = []
    speakers = {}

    # Load CSV
    with open(path + "/train.tsv") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        next(reader) # Skip header
        data = list(reader)

    # Extract files and speakers
    for row in data:
        file = path + "/clips/" + row[1]
        speaker = "cv_" + row[0]
        text = row[2]
        
        # Extract speaker
        if speaker not in speakers:
            speakers[speaker] = len(speakers)
        speaker = speakers[speaker]

        # Append
        files.append((file, text, speaker, False))

    return { 'files': files, 'speakers': speaker, 'preprocessed': [], 'preprocessed_base': None }

def load_preprocessed_corpus(path):
    files = []
    speakers = {}
    preprocessed = []

    # Read all lines from index
    with open(os.path.join(path, "files_valid.txt") , "r") as f:
        file_list = f.readlines()
        file_list = [x.strip() for x in file_list]

    # Extract files and speakers
    for row in file_list:
        file = path + row + ".flac"
        text_file = path + row + ".txt"
        with open(text_file, "r") as f:
            text = f.read()
        speaker = "pr_" + str(Path(row).parts[0])

        # Extract speaker
        if speaker not in speakers:
            speakers[speaker] = len(speakers)
        speaker = speakers[speaker]

        # Append
        files.append((file, text, speaker, True))
        preprocessed.append((row + ".TextGrid", speaker))

    return { 'files': files, 'speakers': speakers, 'preprocessed': preprocessed, 'preprocessed_base': path }

def execute_run():

    # Indexing files
    print("Build file index...")
    collections = {}
    collections['libritts'] = load_libritts_corpus(["libritts-r-clean-100", "libritts-r-clean-360"])
    collections['eval'] = load_libritts_corpus(["libritts-r/test-clean"])
    collections['vctk'] = load_vctk_corpus()
    collections['common-voice-en'] = load_common_voice_corpus("external_datasets/common-voice-16.0-en/en")
    collections['common-voice-ru'] = load_common_voice_corpus("external_datasets/common-voice-16.0-ru/ru")
    collections['common-voice-uk'] = load_common_voice_corpus("external_datasets/common-voice-16.0-uk/uk")
    collections['librilight'] = load_preprocessed_corpus("external_datasets/librilight-processed/")
    collections['librilight-medium'] = load_preprocessed_corpus("external_datasets/librilight-medium-processed/")
    init_if_needed()

    # Process collections
    for collection in collections:
        print(f"Processing collection {collection} with {len(collections[collection]['files'])} files")
        name = collection
        files = collections[collection]['files']
        speakers = collections[collection]['speakers']
        preprocessed = collections[collection]['preprocessed']
        preprocessed_base = collections[collection]['preprocessed_base']
        prepared_dir = "datasets/" + name + "-prepared/"
        aligned_dir = "datasets/" + name + "-aligned/"

        # Check if exists
        if Path(prepared_dir).exists():
            print(f"Collection {name} already prepared")
        else:

            # Creating directories
            for speaker in speakers:
                Path(prepared_dir + speaker_directory(speakers[speaker])).mkdir(parents=True, exist_ok=True)

            # Process files
            with multiprocessing.Manager() as manager:
                files = manager.list(files)
                args_list = [(files, prepared_dir, i) for i in range(len(files))]
                with multiprocessing.Pool(processes=PARAM_WORKERS) as pool:
                    for result in tqdm(pool.imap_unordered(execute_parallel, args_list, chunksize=32), total=len(files)):
                        pass

        # Copy alignment files
        if len(preprocessed) > 0:

            # Check if exists
            if Path(aligned_dir).exists():
                print(f"Collection alignments {name} already prepared")
            else:
                # Creating directories
                for speaker in speakers:
                    Path(aligned_dir + speaker_directory(speakers[speaker])).mkdir(parents=True, exist_ok=True)

                # Copy files
                with multiprocessing.Manager() as manager:
                    files = manager.list(preprocessed)
                    args_list = [(files, preprocessed_base, aligned_dir, i) for i in range(len(files))]
                    with multiprocessing.Pool(processes=PARAM_WORKERS) as pool:
                        for result in tqdm(pool.imap_unordered(copy_parallel, args_list, chunksize=32), total=len(args_list)):
                            pass

    # End
    print("Done")

if __name__ == "__main__":
    execute_run()