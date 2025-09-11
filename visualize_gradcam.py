# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

# def visualize_gradcam(model, target_layer, img_path, class_names, transform, device):
#     img = Image.open(img_path).convert("RGB")
#     input_tensor = transform(img).unsqueeze(0).to(device)

#     cam = GradCAM(model=model, target_layers=[target_layer])

#     model.eval()
#     with torch.no_grad():
#         outputs = model(input_tensor)
#         predicted_class = outputs.argmax(dim=1).item()
#         predicted_label = class_names[predicted_class]

#     targets = [ClassifierOutputTarget(predicted_class)]
#     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

#     rgb_img = np.array(img.resize((128, 128))) / 255.0
#     visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

#     plt.figure(figsize=(8, 4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.title(f"Original Image\nPredicted: {predicted_label}")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(visualization)
#     plt.title("Grad-CAM Heatmap")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()
