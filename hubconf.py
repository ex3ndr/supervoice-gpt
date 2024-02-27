dependencies = ['torch', 'torchaudio']

def supervoice():

    # Imports
    import torch
    import os
    from supervoice_gpt import SupervoiceGPT, Tokenizer, config

    # Download
    hub_dir = torch.hub.get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints', 'supervoice-gpt-1-tokenizer.model')
    torch.hub.download_url_to_file("https://shared.korshakov.com/models/supervoice-gpt-1-tokenizer.model", model_dir)

    # Tokenizer
    tokenizer = Tokenizer(config, model_dir)

    # Model
    model = SupervoiceGPT(config)
    checkpoint = torch.hub.load_state_dict_from_url("https://shared.korshakov.com/models/supervoice-gpt-1.pt")
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Wrapper
    class SupervoicePhonemizer(torch.nn.Module):
        def __init__(self, model, tokenizer):
            super(SupervoicePhonemizer, self).__init__()
            self.model = model
            self.tokenizer = tokenizer
        def forward(self, input, tokenizer, max_new_tokens = 128, temperature=1.0, top_k=5, deterministic = False):
            device = next(self.model.parameters()).device
            return self.model.generate(input, tokenizer, max_new_tokens, temperature, top_k, deterministic, device=device)
    
    return SupervoicePhonemizer(model, tokenizer)
            