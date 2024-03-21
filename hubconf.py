dependencies = ['torch', 'torchaudio']

def phonemizer():

    # Imports
    import torch
    import os
    from supervoice_gpt import SupervoiceGPT, Tokenizer, config

    # Model
    tokenizer = Tokenizer(config, os.path.join(os.path.dirname(__file__), "tokenizer_text.model"))
    model = SupervoiceGPT(tokenizer, config)
    checkpoint = torch.hub.load_state_dict_from_url("https://github.com/ex3ndr/supervoice-gpt/releases/download/v0.0.1/supervoice_gpt_pitch_255000.pt", map_location="cpu")
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model
            
