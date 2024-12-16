import torch


def predict(model, image, device):
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

        probabilities = torch.nn.functional.softmax(output, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

        confidence_percentage = int(confidence.item() * 100)

        return predicted.item(), confidence_percentage
