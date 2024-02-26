# ✨ Supervoice GPT

A GPT model that converts from text to phonemes with durations that is suitable to feed into voice synthesizer.

## How it works?

This model converts raw text to phonemes and their durations compatible with [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/).

This model converts string like `"Hey, Vera, what time is it?"` to list of tuples of phoneme and it's duration:
 `[('ç', 9), ('iː', 7), ('v', 7), ('ɛ', 8), ('ɹ', 8), ('i', 7), ('w', 6), ('ɐ', 5), ('ʔ', 3), ('tʰ', 8), ('aj', 11), ('m', 7), ('ɪ', 6), ('z', 7), ('ɪ', 6), ('ʔ', 8)]`

## Dataset

This module require extensive dataset preparation. To prepare all needed data next commands are required to be performed:

1. `datasets sync` to download datasets
2. `python ./datasets_prepare.py` to preprocess audio files and extract texts from datasets
3. `./datasets_align.sh` to generate alignments
4. `python ./datasets_mix.py` to mix all data together
5. `python ./train_tokenizer.py` to train tokenizer on alignments
6. `python ./datasets_tokenize.py` to tokenize datasets

## Training
To train network execute:

```bash
./train.sh
```

## License
MIT
