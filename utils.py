import torch
import io
import base64
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

IMAGE_SIZE = (128, 128)
CLASS_NAMES = ['Center', 'Donut', 'Edge-loc', 'Edge-ring', 'Loc', 'Near-Full', 'Random', 'Scratch']

def get_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def transform_image(image_bytes):
    transform = get_transform()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0), np.array(image.resize(IMAGE_SIZE)) / 255.0

def get_prediction(model, image_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    return predicted_idx.item()

def generate_gradcam(model, image_tensor, original_image, predicted_class):
    target_layer = model.conv3
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(predicted_class)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

    # Convert visualization to a base64 string to send to the browser
    pil_img = Image.fromarray(visualization)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    base64_img = base64.b64encode(buff.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{base64_img}"