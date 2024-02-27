import torch
from supervoice_gpt import Tokenizer, SupervoiceGPT
from train_config import config

# tokenizer = Tokenizer(config, "tokenizer_text.model")

# Load Model
model = SupervoiceGPT(config)
checkpoint = torch.load(f'./output/pre.pt', map_location="cpu")
model.load_state_dict(checkpoint['model'])
model.eval()

# Create ONNX Wrapper
class Encoder(torch.nn.Module):
    def __init__(self, model):
        super(Encoder, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model.encode(x)
encoder = Encoder(model)

# Export
input = torch.IntTensor([0,1,2,3,4])
torch.onnx.export(encoder, input, "supervoice-encoder.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=11)