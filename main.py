import torch
import cv2

from load import load_model
from data_preprocess import preprocess_image, tensor_to_numpy


def predict(model, image, device):
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.__version__)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    model_path = 'SimpleCNN_model.pth'
    model = load_model(model_path, device)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't load image from camera.")
            break

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted_gray_image = cv2.bitwise_not(gray_image)

        height, width = inverted_gray_image.shape
        box_size = 100
        center_x, center_y = width // 2, height // 2

        top_left = (center_x - box_size // 2, center_y - box_size // 2)
        bottom_right = (center_x + box_size // 2, center_y + box_size // 2)

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        cropped_image = inverted_gray_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        processed_image = preprocess_image(cropped_image)

        processed_image_np = cv2.resize(tensor_to_numpy(processed_image), (280, 280))

        cv2.imshow("Processed Image", processed_image_np)

        prediction = predict(model, processed_image, device)

        cv2.putText(frame, f"Predicted Digit: {prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
