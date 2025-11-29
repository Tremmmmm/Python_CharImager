# File: train.py (Gộp cả xử lý dữ liệu và huấn luyện)
import os
import shutil
import json
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

######################################
# 1. CHUẨN BỊ DỮ LIỆU VÀ CHIA TẬP
######################################
def organize_and_split_by_filename(image_root, annotation_root, output_class_dir, output_split_dir, val_ratio=0.1, test_ratio=0.1):
    print("[STEP 1] Organizing images by label using JSON filenames...")
    os.makedirs(output_class_dir, exist_ok=True)
    count = 0

    for label_folder in os.listdir(annotation_root):
        label_path = os.path.join(annotation_root, label_folder)
        if not os.path.isdir(label_path):
            continue
        for json_file in os.listdir(label_path):
            if not json_file.endswith(".json"):
                continue
            image_file = json_file.replace(".json", ".jpg")
            src_img_path = os.path.join(image_root, label_folder, image_file)
            dst_label_dir = os.path.join(output_class_dir, label_folder)
            os.makedirs(dst_label_dir, exist_ok=True)
            dst_img_path = os.path.join(dst_label_dir, image_file)
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
                count += 1
                if count % 500 == 0:
                    print(f"  Copied {count} images...")
            else:
                print(f"[WARNING] Missing image: {src_img_path}")

    print(f"[DONE] Total organized images: {count}\n")

    print("[STEP 2] Splitting dataset into train/val/test...")
    class_names = os.listdir(output_class_dir)
    for class_name in class_names:
        class_path = os.path.join(output_class_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(('jpg', 'png', 'jpeg'))]
        print(f"Class {class_name}: {len(images)} images")
        if not images:
            continue
        train_imgs, valtest_imgs = train_test_split(images, test_size=val_ratio+test_ratio, random_state=42)
        val_imgs, test_imgs = train_test_split(valtest_imgs, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)

        for phase, img_list in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            dest_dir = os.path.join(output_split_dir, phase, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img in img_list:
                shutil.copy(os.path.join(class_path, img), os.path.join(dest_dir, img))
            print(f"  -> {phase.upper()} - {class_name}: {len(img_list)} images")

    print("[DONE] Dataset splitting completed.\n")

######################################
# 2. HÀM TẠO DATALOADER + TRANSFORM
######################################
def prepare_dataloaders(data_dir, batch_size=32, img_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_dataset.classes

######################################
# 3. ĐỊNH NGHĨA MÔ HÌNH CNN
######################################
class ChartCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ChartCNN, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

######################################
# 4. HÀM HUẤN LUYỆN MÔ HÌNH
######################################
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

def train_model(data_dir, num_epochs=10, batch_size=32, lr=0.001, img_size=224, save_path='best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, class_names = prepare_dataloaders(data_dir, batch_size, img_size)

    model = ChartCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f} Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("[INFO] Saved best model.")

######################################
# 5. CHẠY TOÀN BỘ QUY TRÌNH
######################################
if __name__ == '__main__':
    # BƯỚC 1: Tạo thư mục dữ liệu theo nhãn và chia tập (chạy 1 lần nếu chưa có)
    organize_and_split_by_filename(
        image_root='ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/images',
        annotation_root='ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/annotations_JSON',
        output_class_dir='data_by_class',
        output_split_dir='data_by_class_split'
    )

    # BƯỚC 2: Huấn luyện mô hình
    train_model(data_dir='data_by_class_split', num_epochs=10)
