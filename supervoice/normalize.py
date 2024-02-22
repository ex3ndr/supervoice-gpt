# Some garbage tokens that could be safely ignored
ignored = [    
    '́', '€', '≡', '京', '先', '大', '奔', '尚', '时', '熊', '生', '都', '阪', 'ﬂ', '՚',
    'נ', 'ע', '~', '§', '¯', 'æ'
]

# Mapping keys
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

    # Used for techincal things
    '｜': '|',
    # '•': '⋅'
}

def normalize(tokens):
    return ''.join([mapped_keys[token] if token in mapped_keys else token for token in tokens if token not in ignored])