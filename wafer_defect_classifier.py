# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# from visualize_gradcam import visualize_gradcam
# import matplotlib.pyplot as plt

# #PARAMETERS
# DATA_DIR = "wafer_defect_dataset"
# BATCH_SIZE = 32
# EPOCHS = 25
# LR = 0.001
# IMAGE_SIZE = (128, 128)
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform = transforms.Compose([
#     transforms.Resize(IMAGE_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
# val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CLASS_NAMES = train_dataset.classes
# NUM_CLASSES = len(CLASS_NAMES)
# class WaferCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(WaferCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.fc1 = nn.Linear(128 * 16 * 16, 256)
#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))  # 64x64
#         x = self.pool(F.relu(self.conv2(x)))  # 32x32
#         x = self.pool(F.relu(self.conv3(x)))  # 16x16
#         x = x.view(-1, 128 * 16 * 16)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# model = WaferCNN(NUM_CLASSES).to(DEVICE)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)

# for epoch in range(EPOCHS):
#     model.train()
#     running_loss = 0
#     for images, labels in train_loader:
#         images, labels = images.to(DEVICE), labels.to(DEVICE)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"[Epoch {epoch+1}] Loss: {running_loss / len(train_loader):.4f}")

# model.eval()
# correct, total = 0, 0
# all_preds, all_labels = [], []

# with torch.no_grad():
#     for images, labels in val_loader:
#         images, labels = images.to(DEVICE), labels.to(DEVICE)
#         outputs = model(images)
#         _, preds = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (preds == labels).sum().item()
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

# print(f"Validation Accuracy: {100 * correct / total:.2f}%")
# print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# print("Finished Training!")

# # Define the path to save the model
# MODEL_SAVE_PATH = "wafer_defect_model.pth"

# # Save the model's state dictionary
# torch.save(model.state_dict(), MODEL_SAVE_PATH)

# print(f"Model saved to {MODEL_SAVE_PATH}")

# cm = confusion_matrix(all_labels, all_preds)
# cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  
# cm_percent_rounded = np.round(cm_percent, decimals=1)

# fig, ax = plt.subplots(figsize=(10, 8))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent_rounded, display_labels=CLASS_NAMES)
# disp.plot(
#     cmap=plt.cm.Blues, 
#     xticks_rotation=45, 
#     values_format='.1f',  # Shows decimals like 95.2
#     ax=ax
# )

# plt.title("Confusion Matrix (% per True Class)")
# plt.tight_layout()
# plt.show()

# target_layer = model.conv3
# img_path = "wafer_defect_dataset/test/Center/center_1732.jpg"

# visualize_gradcam(
#     model=model,
#     target_layer=target_layer,
#     img_path=img_path,
#     class_names=CLASS_NAMES,
#     transform=transform,
#     device=DEVICE
# )
