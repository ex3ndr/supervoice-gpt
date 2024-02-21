from glob import glob
from tqdm import tqdm
import multiprocessing

# Some garbage tokens that could be safely ignored
ignored = [    
    '́', '€', '≡', '京', '先', '大', '奔', '尚', '时', '熊', '生', '都', '阪', 'ﬂ', '՚',
    'נ', 'ע', '~', '§', '¯', 'æ'
]

ignored_numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # Not logged

mapped_keys = {

    # Various dashes
    '‑': '-', 
    '–': '-', 
    '—': '-', 
    '−': '-',
    '→': '-',

    # Various quotes
    '"': '\'',
    '`': '\'',
    '´': '\'',
    '‘': '\'',
    '’': '\'',
    '“': '\'',
    '”': '\'',
    '„': '\'',
    '«': '\'',
    '»': '\'',
    'ʻ': '\'',
}

def get_tokens(path):
    with open(path, "r") as f:
        text = f.read().lower()
    r = set(list(text))
    # log ignored 
    for char in r:
        if char in ignored:
            print(f"Ignored: {char} in {path}")
    return r
    

def main():

    # Load source texts
    files = glob('./datasets/*-prepared/*/*.txt')

    # Reading all characters
    tokens = set()
    with multiprocessing.Manager() as manager:
        files = manager.list(files)
        with multiprocessing.Pool(processes=8) as pool:
                for result in tqdm(pool.imap_unordered(get_tokens, files, chunksize=8), total=len(files)):
                    tokens = tokens.union(result)

    # Remove ignored
    for char in ignored:
        tokens.discard(char)
    for char in ignored_numbers:
        tokens.discard(char)

    # Map keys
    for key in mapped_keys:
        tokens.discard(key)
        tokens.add(mapped_keys[key])

    # Prepare array
    tokens = list(tokens)
    tokens.sort()

    # Print results
    print(tokens)

if __name__ == "__main__":
    main()