dependencies = ['torch', 'torchaudio']

def phonemizer():

    # Imports
    import torch
    import os
    from supervoice_gpt import SupervoiceGPT, Tokenizer, config

    # Model
    tokenizer = Tokenizer(config, os.path.join(os.path.dirname(__file__), tokenizer_text.model))
    model = SupervoiceGPT(config)
    checkpoint = torch.hub.load_state_dict_from_url("https://github.com/ex3ndr/supervoice-gpt/releases/download/v0.0.1/supervoice_gpt_pitch_255000.pt", map_location="cpu")
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Wrapper
    class SupervoicePhonemizer(torch.nn.Module):
        def __init__(self, model, tokenizer):
            super(SupervoicePhonemizer, self).__init__()
            self.model = model
            self.tokenizer = tokenizer
        def forward(self, input, max_new_tokens = 128, temperature=1.0, top_k=5, deterministic = False):
            device = next(self.model.parameters()).device
            return self.model.generate(input, self.tokenizer, max_new_tokens, temperature, top_k, deterministic, device=device)
    
    return SupervoicePhonemizer(model, tokenizer)
            
