dependencies = ['torch', 'torchaudio']

def supervoice():

    # Imports
    import torch
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
    
    return model
            