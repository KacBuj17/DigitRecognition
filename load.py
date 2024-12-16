import torch
from model import SimpleCNN


def load_model(path, device):
    model = SimpleCNN()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.to(device)
    print(f"Model loaded from {path}")
    return model
