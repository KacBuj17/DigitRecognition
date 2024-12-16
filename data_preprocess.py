from torchvision import transforms
import numpy as np


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)


def tensor_to_numpy(tensor):
    image = tensor.squeeze().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return image
