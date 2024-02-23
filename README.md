# ✨ Supervoice GPT

A GPT model that converts from text to phonemes with durations that is suitable to feed into voice synthesizer.

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
